import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from ipywidgets import interact, Dropdown
import ipywidgets as widgets
from typing import Dict, Any, Union, Optional, Tuple
import itertools
from tqdm.notebook import trange
from sklearn import linear_model
import cvxpy as cp

from icl.linear.lr_models import DiscreteMMSE, Ridge
from icl.linear.lr_task import *
from icl.models import apply_rotary_emb

"""
Legacy utility functions that were used in earlier experiments. 
These functions may be deprecated in future releases.
"""


def find_task_vector_with_baseline(
    model,
    train_task,
    task_idx=0,
    l0=0,
    num_epochs=3000,
    lr=1e-3,
    pad="mapsto",
    verbose=True,
    return_baseline=False,
    l2_reg=1e-2,
    init_vec = None,
    init_reg = 2, 
    min_cos_sim = 0.8
):
    """
    Optimize a task vector theta for a given TransformerLin model on a specified task,
    while logging the baseline (frozen model) prediction loss for comparison.

    Args:
        model: TransformerLin instance (frozen).
        train_task: task sampler with sample_from_task(task_pool[task_idx], epoch).
        task_idx: index of the task in the task pool.
        l0: injection layer index.
        num_epochs: optimization iterations.
        lr: learning rate for theta.
        verbose: whether to print progress.
        return_baseline: if True, returns baseline predictions for analysis.

    Returns:
        theta: optimized task vector.
        (optional) baseline_preds: baseline model predictions without injection.
    """

    # Freeze the model
    for p in model.parameters(): p.requires_grad = False

    # Create learnable task vector
    d_model = model.transformer.blocks[0].ln_1.normalized_shape[0]
    if init_vec is None:
        theta = nn.Parameter(torch.randn(d_model, device=model.device)) / (d_model ** 0.5)  # Initialize theta with small random values
    else:
        theta = nn.Parameter(
            init_vec +  torch.randn_like(init_vec) / (d_model ** 0.5)
            )

    optimizer = torch.optim.LBFGS([theta], lr=lr, max_iter=10, history_size=10)
    # optim.Adam([theta], lr=lr)
    loss_fn = nn.MSELoss()

    # Sample a reference batch for baseline computation
    data_ref, target_ref = train_task.sample_from_task(train_task.task_pool[task_idx], step=0)
    target_ref_device = target_ref.to(model.device)

    # Compute baseline (frozen model) predictions and loss
    with torch.no_grad():
        baseline_preds = model(data_ref, target_ref)
        baseline_loss = loss_fn(baseline_preds[:,-1], target_ref_device[:,-1]).item()

    if verbose:
        print(f"[Task {task_idx}] Baseline Loss (no injection): {baseline_loss:.6f}")

    # Hook for BOS injection
    def bos_injection_hook(module, input, output):
        if pad == "bos":
            output[:, 0, :] = theta # Inject theta into the BOS position, output shape: (batch, seq_len, d_model)
        else:
            output[:, 1, :] = theta
        return output

    # Register hook
    hook_handle = model.transformer.blocks[l0].register_forward_hook(bos_injection_hook)
    flag = False
    try:
        for epoch in range(num_epochs):
            data, target = train_task.sample_from_task(train_task.task_pool[task_idx], epoch)

            def closure():
                optimizer.zero_grad()
                preds = model(data, target)
                # print(preds.shape, target.shape)
                loss = loss_fn(preds[:, 0], target[:, 0].to(preds.device))
                if loss.item() <= baseline_loss:
                    flag = True
                    if epoch % 2 != 0 and verbose:
                        print(f"[Task {task_idx}] Epoch {epoch}, Loss: {loss.item():.6f} (Baseline: {baseline_loss:.6f})")
                    return loss

                l2_penalty = l2_reg * theta.pow(2).sum()

                cos_penalty = 0.0
                if init_vec is not None:
                    # Compute cosine similarity between theta and init_vec
                    cos_sim = F.cosine_similarity(theta, init_vec.to(theta.device), dim=0, eps=1e-8)
                    # Encourage cosine similarity to be at least min_cos_sim
                    cos_penalty = init_reg * torch.relu(min_cos_sim - cos_sim)
                
                total_loss = loss + l2_penalty + cos_penalty

                total_loss.backward()
                return loss

            loss = optimizer.step(closure)

            if verbose and epoch % 2 == 0:
                print(f"[Task {task_idx}] Epoch {epoch}, Loss: {loss.item():.6f} (Baseline: {baseline_loss:.6f})")

            if loss.item() <= baseline_loss:
                flag = True
                if epoch % 2 != 0 and verbose:
                    print(f"[Task {task_idx}] Epoch {epoch}, Loss: {loss.item():.6f} (Baseline: {baseline_loss:.6f})")
                break

    finally:
        hook_handle.remove()

    if return_baseline:
        return theta.detach().cpu(), baseline_preds.detach(), flag
    else:
        return theta.detach().cpu(), flag

def extract_weak_task_vector(
        model, demo_data, demo_target, l=0, task_pos=-1
    ):
    extracted_vector = {}

    def hook_fn(module, input, output):
        # output: (batch, seq_len, d_model)
        extracted_vector['diff'] = output[:, task_pos, :].detach().clone() - input[:, task_pos, :].detach().clone()
        extracted_vector['input'] = input[:, task_pos, :].detach().clone()

    hook_handle = model.transformer.blocks[l].attn_block.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(demo_data, demo_target)
    hook_handle.remove()

    # attn_map = get_attn_at_layer(model, demo_data, demo_target, l)

    return extracted_vector['vector']



def compute_mixed_hiddens(config,
                          model: torch.nn.Module,
                          train_task,
                          layer_index: int = 1):
    n_tasks = train_task.task_pool.shape[0]
    n_points = config.task.n_points
    batch_size = train_task.batch_size
    n_embd = config.model.n_embd

    task_pos = 3 * torch.arange(0, n_points) + 1  # Position of task tokens
    hiddens = torch.zeros((n_tasks, n_points, batch_size, n_embd), device=model.device)
        
    for k in range(n_tasks):
        demo_data, demo_labels, demo_target = train_task.sample_from_task(train_task.task_pool1[k], train_task.task_pool2[k], step=1)
        hidden = extract_hidden(
            model=model,
            demo_data=demo_data,
            demo_target=demo_target,
            l=layer_index,
            task_pos=task_pos
        )
        hiddens[k] = hidden.transpose(0, 1)

    return hiddens, demo_data

def extract_task_vector_diff(
        model, demo_data, demo_target, l=0, task_pos=-1, init_pos=1
    ):
    extracted_vector = {}

    def hook_fn(module, input, output):
        # output: (batch, seq_len, d_model)
        diff_vector = output[:, task_pos, :] - output[:, init_pos, :]
        extracted_vector['vector'] = diff_vector.mean(dim=0).detach().clone()

    hook_handle = model.transformer.blocks[l].register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(demo_data, demo_target)
    hook_handle.remove()

    return extracted_vector['vector']

def predict_with_task_vector_diff(
        model, query_data, query_target, 
        task_vector, l=0, lamda=0.1
    ):
    task_pos=1

    def inject_hook(module, input, output):
        output_norm = output[:, task_pos, :].norm(dim=-1, keepdim=True)
        output[:, task_pos, :] += lamda * task_vector
        # Normalize the output to maintain scale
        output[:, task_pos, :] = output[:, task_pos, :] / (output[:, task_pos, :].norm(dim=-1, keepdim=True) + 1e-8)
        output[:, task_pos, :] *= output_norm
        return output

    hook_handle = model.transformer.blocks[l].register_forward_hook(inject_hook)
    preds = model(query_data, query_target)
    hook_handle.remove()

    return preds




def weighted_average_favor_late_batched(
    x: torch.Tensor, 
    mode="linear", 
    alpha=0.1, 
    cap_threshold: Optional[int] = None
):
    """
    x: Tensor of shape (B, T, D)
    mode: "linear", "quadratic", "exp"
    alpha: controls steepness for "exp"
    cap_threshold: if specified, weights stop increasing after this t

    Returns:
        Tensor of shape (B, D) with weighted average favoring later positions
    """
    B, T, D = x.shape
    t = torch.arange(T, dtype=x.dtype, device=x.device)

    if mode == "linear":
        weights = t + 1
    elif mode == "quadratic":
        weights = (t + 1) ** 2
    elif mode == "exp":
        weights = torch.exp(alpha * t)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if cap_threshold is not None:
        # Compute the cap value at threshold index
        cap_index = min(cap_threshold, T - 1)  # prevent out-of-bounds
        cap_value = weights[cap_index]
        weights = torch.where(t <= cap_threshold, weights, cap_value)

    weighted_sum = (weights[None, :, None] * x).sum(dim=1)  # (B, D)
    weights_sum = weights.sum()  # scalar
    return weighted_sum / weights_sum  # (B, D)

def moving_l2_distance_from_mean_sumD(x: torch.Tensor, window_size: int, sqrt: bool = False) -> torch.Tensor:
    """
    Compute moving L2 distance around moving mean for x of shape (B, T, D),
    summing over D, output shape (B, T - window_size + 1).
    """
    B, T, D = x.shape
    x_perm = x.permute(0, 2, 1)  # (B, D, T)
    x_unfold = x_perm.unfold(dimension=2, size=window_size, step=1)  # (B, D, T-w+1, w)

    mean = x_unfold.mean(dim=-1, keepdim=True)  # (B, D, T-w+1, 1)
    sq_dev = ((x_unfold - mean) ** 2).mean(dim=-1)  # (B, D, T-w+1)

    reduced = sq_dev.sum(dim=1)  # sum over D → (B, T-w+1)

    if sqrt:
        reduced = reduced.sqrt()

    return reduced


#########################
# Plotting Functions    #
#########################

def plot_moving_distances(task_vectors, ws_dmmse, ws_re, window_size=10, cap=170, max_labels=10):
    """
    Plot moving L2 distances from mean for multiple datasets using Plotly.
    
    Parameters:
    -----------
    task_vectors : torch.Tensor
        Task vectors tensor
    ws_dmmse : torch.Tensor
        dMMSE tensor
    ws_re : torch.Tensor
        Ridge regression tensor
    window_size : int, default=10
        Window size for moving average calculation
    cap : int, default=170
        Maximum number of positions to plot
    max_labels : int, default=10
        Maximum number of tasks to include in legend
    
    Returns:
    --------
    list of plotly.graph_objects.Figure
        List containing the three generated plots
    """
    
    # Calculate means and moving distances
    tvs_means = task_vectors.mean(dim=-2)
    moving_dists = moving_l2_distance_from_mean_sumD(tvs_means, window_size=window_size, sqrt=True)
    
    ws_dmmse_means = ws_dmmse.mean(dim=2)
    ws_dmmse_moving_dists = moving_l2_distance_from_mean_sumD(ws_dmmse_means, window_size=window_size, sqrt=True)
    
    ws_re_means = ws_re.mean(dim=2)
    ws_re_moving_dists = moving_l2_distance_from_mean_sumD(ws_re_means, window_size=window_size, sqrt=True)
    
    # Create position array
    t = torch.arange(1, moving_dists.shape[1] + 1)
    
    # Define plot configurations
    plot_configs = [
        {
            'data': moving_dists,
            'title': f'Task Vectors - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'Task',
            'max_labels': max_labels
        },
        {
            'data': ws_dmmse_moving_dists,
            'title': f'dMMSE - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'dMMSE Task',
            'max_labels': 5
        },
        {
            'data': ws_re_moving_dists,
            'title': f'Ridge - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'Ridge Task',
            'max_labels': 5
        }
    ]
    
    figures = []
    
    for config in plot_configs:
        fig = go.Figure()
        
        data = config['data']
        
        # Add traces for each task
        for i in range(data.shape[0]):
            show_legend = i < config['max_labels']
            label = f"{config['label_prefix']} {i}" if show_legend else None
            
            fig.add_trace(go.Scatter(
                x=t[:cap].numpy(),
                y=data[i][:cap].numpy(),
                mode='lines',
                name=label,
                showlegend=show_legend,
                line=dict(width=1.5)
            ))
        
        # Update layout
        fig.update_layout(
            title=config['title'],
            xaxis=dict(
                title='Position',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            yaxis=dict(
                title='Distance (log scale)',
                type='log',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        figures.append(fig)
    
    return figures

def create_interactive_plot(task_vectors, ws_dmmse, ws_re, window_size=10, cap=170, max_labels=10):
    """
    Create an interactive plot with dropdown selection for different datasets.
    
    Parameters:
    -----------
    task_vectors : torch.Tensor
        Task vectors tensor
    ws_dmmse : torch.Tensor
        dMMSE tensor
    ws_re : torch.Tensor
        Ridge regression tensor
    window_size : int, default=10
        Window size for moving average calculation
    cap : int, default=170
        Maximum number of positions to plot
    max_labels : int, default=10
        Maximum number of tasks to include in legend
    """
    
    # Calculate means and moving distances
    tvs_means = task_vectors.mean(dim=-2)
    moving_dists = moving_l2_distance_from_mean_sumD(tvs_means, window_size=window_size, sqrt=True)
    
    ws_dmmse_means = ws_dmmse.mean(dim=2)
    ws_dmmse_moving_dists = moving_l2_distance_from_mean_sumD(ws_dmmse_means, window_size=window_size, sqrt=True)
    
    ws_re_means = ws_re.mean(dim=2)
    ws_re_moving_dists = moving_l2_distance_from_mean_sumD(ws_re_means, window_size=window_size, sqrt=True)
    
    # Create position array
    t = torch.arange(1, moving_dists.shape[1] + 1)
    
    # Define plot configurations
    plot_configs = {
        'Task Vectors': {
            'data': moving_dists,
            'title': f'Task Vectors - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'Task',
            'max_labels': max_labels
        },
        'dMMSE': {
            'data': ws_dmmse_moving_dists,
            'title': f'dMMSE - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'dMMSE Task',
            'max_labels': 5
        },
        'Ridge': {
            'data': ws_re_moving_dists,
            'title': f'Ridge - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': 'Ridge Task',
            'max_labels': 5
        },
        'All Combined': {
            'data': [moving_dists, ws_dmmse_moving_dists, ws_re_moving_dists],
            'title': f'All Methods - Moving Distance Around Moving Average (Window Size: {window_size})',
            'label_prefix': ['Task', 'dMMSE Task', 'Ridge Task'],
            'max_labels': [max_labels, 5, 5]
        }
    }
    
    def update_plot(plot_type):
        """Update the plot based on dropdown selection"""
        fig = go.Figure()
        config = plot_configs[plot_type]
        
        if plot_type == 'All Combined':
            # Handle combined plot
            data_list = config['data']
            label_prefixes = config['label_prefix']
            max_labels_list = config['max_labels']
            colors = ['blue', 'red', 'green']
            
            for idx, (data, label_prefix, max_labels_val) in enumerate(zip(data_list, label_prefixes, max_labels_list)):
                for i in range(data.shape[0]):
                    show_legend = i < max_labels_val
                    label = f"{label_prefix} {i}" if show_legend else None
                    
                    fig.add_trace(go.Scatter(
                        x=t[:cap].numpy(),
                        y=data[i][:cap].numpy(),
                        mode='lines',
                        name=label,
                        showlegend=show_legend,
                        line=dict(width=1.5, color=colors[idx]),
                        opacity=0.7
                    ))
        else:
            # Handle individual plots
            data = config['data']
            
            for i in range(data.shape[0]):
                show_legend = i < config['max_labels']
                label = f"{config['label_prefix']} {i}" if show_legend else None
                
                fig.add_trace(go.Scatter(
                    x=t[:cap].numpy(),
                    y=data[i][:cap].numpy(),
                    mode='lines',
                    name=label,
                    showlegend=show_legend,
                    line=dict(width=1.5)
                ))
        
        # Update layout
        fig.update_layout(
            title=config['title'],
            xaxis=dict(
                title='Position',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            yaxis=dict(
                title='Distance (log scale)',
                type='log',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            ),
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            ),
            width=900,
            height=600
        )
        
        fig.show()
    
    # Create dropdown widget
    dropdown = widgets.Dropdown(
        options=list(plot_configs.keys()),
        value='Task Vectors',
        description='Plot Type:',
        style={'description_width': 'initial'},
        layout={'width': '200px'}
    )
    
    # Create interactive widget
    interact(update_plot, plot_type=dropdown)
    
    return dropdown

# Example usage:
# 
# # Original function - returns all figures
# figures = plot_moving_distances(task_vectors, ws_dmmse, ws_re, window_size=10, cap=170)
# 
# # Display all plots sequentially
# for fig in figures:
#     fig.show()
#
# # Display individual plots
# figures[0].show()  # Task vectors plot
# figures[1].show()  # dMMSE plot  
# figures[2].show()  # Ridge plot
#
# # NEW: Interactive dropdown interface
# dropdown = create_interactive_plot(task_vectors, ws_dmmse, ws_re, window_size=10, cap=170)
# 
# # The dropdown will appear above the plot and allow you to switch between:
# # - Task Vectors
# # - dMMSE  
# # - Ridge
# # - All Combined (shows all three datasets on one plot with different colors)


def estimate_lambda_with_r2_fast(task_vecs, task_vecs_over_all_time, is_zero_mean=True):
    """
    Estimate lambda_{j', t}^{(j)} with constraints and return R² of the fit.

    Returns:
        lambdas: (k, seq_len, num_tasks)
        r2_scores: (k, seq_len) -- R² value per fit
    """
    k, seq_len, d = task_vecs_over_all_time.shape
    num_tasks = task_vecs.shape[0]
    lambdas = np.zeros((k, num_tasks))
    r2_scores = np.zeros((k,))

    X = task_vecs.T  # shape (d, num_tasks)
    if is_zero_mean:
        X = X[:, :-1]

    for j in range(k):
        model = linear_model.LinearRegression()
        y = task_vecs_over_all_time[j, -1, :]  # shape (d,)
        model.fit(X, y)
        lambda_hat = model.coef_  # shape (num_tasks-1,) or (num_tasks,)
        if is_zero_mean:
            last_lambda = (1 - np.sum(lambda_hat)) / num_tasks
            lambdas[j, :] = np.concatenate([lambda_hat, [0]]) + last_lambda  # Add the last lambda to make it sum to 1
        else:
            lambdas[j, :] = lambda_hat

        y = np.asarray(y)
        # Goodness of fit
        y_pred = X @ lambda_hat
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_scores[j] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return lambdas, r2_scores


def analyze_tasks(config: Dict[str, Any], 
                  model, 
                  data_type = torch.float32,
                  batch_size: int = 1024,
                  layer_index: int = 0, 
                  pos: int = 1,
                  return_data: bool = False):
    """
    Analyze tasks by extracting task vectors and computing dMMSE and Ridge evolution.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing task and model parameters
    model : 
        The model to extract task vectors from
    data_type : 
        Data type for computations
    batch_size : int, default=1024
        Batch size for processing
    layer_index : int, default=0
        Layer index for task vector extraction
    
    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - task_vectors: Shape (n_tasks, n_points-1, batch_size, n_embd)
        - ws_dmmse: Shape (n_tasks, n_points-1, batch_size, n_dims)
        - ws_re: Shape (n_tasks, n_points-1, batch_size, n_dims)
    """

    model = model.to(config.device)
    
    # Initialize task
    train_task = get_task(**config["task"], dtype=data_type)
    train_task.batch_size = batch_size
    
    # Pre-allocate tensors
    task_vectors = torch.zeros((config.task.n_tasks, 
                               config.task.n_points - 1, 
                               train_task.batch_size, 
                               config.model.n_embd))
    
    ws_dmmse = torch.zeros((config.task.n_tasks, 
                           config.task.n_points - 1, 
                           train_task.batch_size, 
                           config.task.n_dims))
    
    ws_re = torch.zeros((config.task.n_tasks, 
                        config.task.n_points - 1, 
                        train_task.batch_size, 
                        config.task.n_dims))
    
    # Initialize estimators
    dMMSE = DiscreteMMSE(config.task.noise_scale, 
                         train_task.task_pool,
                         data_type)
    
    RE = Ridge(config.task.noise_scale**2 / config.task.task_scale**2, 
               data_type)

    # Process each task
    for k in trange(config.task.n_tasks):
        # Sample data from current task
        demo_data, demo_target = train_task.sample_from_task(
            train_task.task_pool[k], 
            step=1
        )

        # Extract task vectors
        task_vectors[k] = extract_task_vector(
            model=model,
            demo_data=demo_data,
            demo_target=demo_target,
            l=layer_index,
            task_pos=3 * torch.arange(1, config.task.n_points) + pos
        ).cpu().transpose(0, 1)
        
        # Compute dMMSE evolution
        ws_dmmse[k] = dMMSE.evolve(demo_data, demo_target).transpose(2, 0).transpose(1, 2)
        
        # Compute Ridge evolution
        ws_re[k] = RE.evolve(demo_data, demo_target).transpose(2, 0).transpose(1, 2)
    
    if return_data:
        return task_vectors, ws_dmmse, ws_re, demo_data
    
    return task_vectors, ws_dmmse, ws_re



"""
# Example usage:
# 
# # Basic usage
# task_vectors, ws_dmmse, ws_re = analyze_tasks(
    config=config,
    model=model,
    data_type=data_type,
    batch_size=1024,
    layer_index=2
# )
"""


def batch_cosine_similarity(A: np.ndarray, B: np.ndarray, eps: float = 1e-8):
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    return dot_product / (norm_A * norm_B + eps)

def rolling_mean_l2_deviation(arr: torch.Tensor, window: int):
    B, T, D = arr.shape
    output = torch.empty(B, T - window, device=arr.device)

    for b in range(B):
        for t in range(T - window):
            window_slice = arr[b, t:t+window, :]  # (W, D)
            window_mean = window_slice.mean(dim=0)  # (D,)
            l2_distances = torch.norm(window_slice - window_mean, dim=1)  # (W,)
            output[b, t] = l2_distances.mean()

    return output