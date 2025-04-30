import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

from pprint import pprint

from itertools import product
from IPython.display import display
import os
import glob
import re
from ml_collections import ConfigDict
import json

from train import get_sampler
from models.base_models import Transformer
import pickle
from datetime import datetime

import linear_algebra_utils as lau

import hashlib

def hash_array(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

###############################
# Memory Probes
###############################

def get_high_order_memory(model, sampler, toks, mlp=False):
    toks = torch.tensor([int(ch) for ch in toks], device=sampler.device)
    batch, _, triggers, _ = sampler.generate(mode="testing")
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    batch_stride = torch.as_strided(batch, size=(1, SEQ_LEN-order, order), stride=(batch.stride(0), batch.stride(1), batch.stride(1))).squeeze(0)
    matches = (batch_stride == toks).all(dim=-1)
    indices = torch.nonzero(matches, as_tuple=False).squeeze()
    powers = VOC_SIZE ** torch.arange(order - 1, -1, -1, device=sampler.device)
    tok_ind = torch.sum(toks * powers)

    print(indices.ndim)
    
    while ((torch.isin(toks, triggers).any()) 
           or indices.ndim < 1
           or indices.size(0) < 2
           or (indices[-1]<2) 
           or (indices[-1] > SEQ_LEN-order)):
        batch, _, triggers, _ = sampler.generate(mode="testing")
        batch_stride = torch.as_strided(batch, size=(1, SEQ_LEN-order, order), stride=(batch.stride(0), batch.stride(1), batch.stride(1))).squeeze(0)
        matches = (batch_stride == toks).all(dim=-1)
        indices = torch.nonzero(matches, as_tuple=False).squeeze()
    
    pos = indices[-1]
    print("Position: ", pos.item())
    base_prob = nn.Softmax(dim=-1)(model(batch)[0])[0][pos+order-1].detach().cpu()

    embs = model.embed(batch)
    sa = model.layers[0].MHA(embs, False)[0]
    sa1_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos+order-1].detach().cpu()
    if mlp:
        sa_ffn = model.layers[1].mlp(sa)
        sa1_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos+order-1].detach().cpu()
    
    hidden = model.layers[0](embs)[0]
    sa = model.layers[1].MHA(hidden, False)[0]
    res_prob = nn.Softmax(dim=-1)(model.output_layer(hidden))[0][pos+order-1].detach().cpu()
    sa_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos+order-1].detach().cpu()
    if mlp:
        out_ffn = model.layers[1].mlp(hidden)
        out_ffn_res = model.layers[1].mlp(hidden) + hidden
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        out_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn_res))[0][pos+order-1].detach().cpu()
        sa_ffn = model.layers[1].mlp(sa)
        sa_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos+order-1].detach().cpu()
    sa_res = hidden+sa
    sa_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res))[0][pos+order-1].detach().cpu()
    if mlp:
        sa_res_ffn = model.layers[1].mlp(sa_res)
        sa_res_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn))[0][pos+order-1].detach().cpu()
        sa_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+sa))[0][pos+order-1].detach().cpu()
        sa_ffn_toks_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+hidden))[0][pos+order-1].detach().cpu()
    if mlp:
        df = pd.DataFrame({'base': base_prob, 'sa1': sa1_prob, 'ffn(sa1)': sa1_ffn_prob,
                           'sa2': sa_prob, 'out1': res_prob, 'ffn(out1)': out_ffn_prob, 
                           'sa2+out1': sa_res_prob, "ffn(out1)+out1": out_ffn_res_prob,
                           'ffn(sa2)': sa_ffn_prob, "ffn(sa2+out1)": sa_res_ffn_prob, 
                           'ffn(sa2+out1)+sa2': sa_ffn_res_prob, "ffn(sa2+out1)+out1": sa_ffn_toks_prob})
    else:
        df = pd.DataFrame({'base': base_prob, 'sa1': sa_prob, 
                           'sa2': sa_prob, 'out1': res_prob, 'sa2+out1': sa_res_prob})

    ground_truth = sampler.trans_mat[tok_ind].cpu()

    display(df)
    
    print("-"*50)
    for i, key in enumerate(df.keys()):
        kl = F.kl_div(torch.tensor(df[key].values).log(), ground_truth, reduction="sum")
        sig = ""
        if kl < 0.05:
            sig = " ***"
        elif kl < 0.15:
            sig = " **"
        elif kl < 0.3:
            sig = " *"
        ending = "   " if ((i+1) % 4) != 0 else "\n"
        print(f"{key} : {kl.item():.3f}", sig, end=ending)
    print("\n")
    print("-"*50)
    # Plot the heatmap
    sns.heatmap(df, cmap="viridis", annot=True, fmt=".2f", cbar=False)
    
    # Customize labels
    plt.title(f"Probability Heatmap")
    
    # Show plot
    plt.tight_layout()
    plt.show()


def high_order_memory_probe_df(model, sampler, to_probe="ff+res"):
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    perms = list(product(range(VOC_SIZE), repeat=order))
    perms = [''.join(map(str, p)) for p in perms]
    pos = SEQ_LEN - 10
    batch = sampler.generate(mode="probe")
    df = pd.DataFrame(0., index=perms, columns=[f"{i}" for i in range(VOC_SIZE)])
    
    for p in perms:
        toks = torch.tensor([int(ch) for ch in p], device=sampler.device)
        batch_copy = batch.clone()
        batch_copy[0][pos:pos+order] = toks[:]
    
        embs = model.embed(batch_copy)
        hidden = model.layers[0](embs)[0]
        out_ffn = model.layers[1].mlp(hidden)
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        out_prob = nn.Softmax(dim=-1)(model.output_layer(hidden))[0][pos+order-1].detach().cpu()
        out_ffn_res = model.layers[1].mlp(hidden) + hidden
        out_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn_res))[0][pos+order-1].detach().cpu()
        if to_probe == "ff":
            df.loc[p] = out_ffn_prob.numpy()
        elif to_probe == "res":
            df.loc[p] = out_prob.numpy()
        else:
            df.loc[p] = out_ffn_res_prob.numpy()

    print(f"KL divergence: {F.kl_div(torch.tensor(df.values).log(), sampler.trans_mat.cpu(), reduction="none").sum(axis=-1).mean():.4f}")
    return df

def high_order_memory_probe_ff_df(model, sampler):
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    perms = list(product(range(VOC_SIZE), repeat=order))
    perms = [''.join(map(str, p)) for p in perms]
    pos = SEQ_LEN - 10
    batch = sampler.generate(mode="probe")
    df = pd.DataFrame(0., index=perms, columns=[f"{i}" for i in range(VOC_SIZE)])
    
    for p in perms:
        toks = torch.tensor([int(ch) for ch in p], device=sampler.device)
        batch_copy = batch.clone()
        batch_copy[0][pos:pos+order] = toks[:]
    
        embs = model.embed(batch_copy)
        hidden = model.layers[0](embs)[0]
        out_ffn = model.layers[1](hidden)[0]
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        df.loc[p] = out_ffn_prob.numpy()

    print(f"KL divergence: {F.kl_div(torch.tensor(df.values).log(), sampler.trans_mat.cpu(), reduction="none").sum(axis=-1).mean():.4f}")
    return df







#################
# Load model
#################

def load_model(checkpoint_dir, config, step=None):
    device = config.device
    
    # Extract step number from each filename
    def extract_step(path):
        match = re.search(r"model_final_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1
    
    if step is not None:
        model_path = os.path.join(checkpoint_dir, f"model_{step}.pt")
    else:
        pattern = "model_final_*.pt"
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if len(files) == 0:
            raise ValueError(f"No model found in {checkpoint_dir} with pattern {pattern}")

        paths = sorted(files, key=extract_step)
        model_path = paths[-1]
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = Transformer(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model.to(device)

####################
# Load Config
####################

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = ConfigDict(config_dict)
    return config


# If you do not know the exact folder name, 
# you can use this function to get the list of folders 
# with the same number of transition matrices.

def get_config(total_trans=None, vocab_size=10, path=None, 
               seq_len=None, hidden_dim=None, alpha=None, 
               stationary=None):
    DEFAULT_PATH = os.path.join("results", "latent")
    if path is None:
        path = DEFAULT_PATH
    if total_trans is None:
        total_trans_list = []
        
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                flag = config.vocab_size == vocab_size
                if seq_len is not None:
                    flag = flag and (config.seq_len == seq_len)
                if hidden_dim is not None:
                    flag = flag and (config.model.emb_dim == hidden_dim)
                if alpha is not None:
                    flag = flag and (config.task.alpha == alpha)
                if stationary is not None:
                    if "stationary" in config.task:
                        flag = flag and (config.task.stationary == stationary)
                    else:
                        flag = flag and (stationary is False)
                if flag:
                    total_trans_list.append(config.task.total_trans)
        pprint(f"You can choose from the following number of total transition matrices: \n {sorted(total_trans_list)}")
    else:
        folder_list = []
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                flag = (config.task.total_trans == total_trans) and (config.vocab_size == vocab_size)
                if seq_len is not None:
                    flag = flag and (config.seq_len == seq_len)
                if hidden_dim is not None:
                    flag = flag and (config.model.emb_dim == hidden_dim)
                if alpha is not None:
                    flag = flag and (config.task.alpha == alpha)
                if stationary is not None:
                    if "stationary" in config.task:
                        flag = flag and (config.task.stationary == stationary)
                    else:
                        flag = flag and (stationary is False)
                if flag:
                    folder_list.append(folder)
        pprint(f"You can choose from the following folders: {folder_list}")
        
        return folder_list


####################
# Load Log
####################

def load_log(log_path):
    with open(log_path, "r") as f:
        log_data = json.load(f)
    return log_data

#####################
# Load Sampler
#####################

def load_sampler(sampler_path):
    sampler = pickle.load(open(sampler_path, "rb"))
    return sampler


def load_everything(task_name, train_folder):
    path_prefix = os.path.join("results", task_name)
    train_folder = train_folder
    checkpoint_dir = os.path.join(path_prefix, train_folder, "checkpoints")
    config_path = os.path.join(path_prefix, train_folder, "config.json")
    sampler_path = os.path.join(path_prefix, train_folder, "sampler.pkl")
    config = load_config(config_path)
    model = load_model(checkpoint_dir, config)
    sampler = load_sampler(sampler_path)
    return model, sampler, config


######################
# Task Vector
######################

def get_single_task_vector_trigger(config, sampler, model, task_id, ffn=True):
    assert task_id < config.task.total_trans , f"Task ID {task_id} out of range"
    batch, mask, q_toks, trans_random = sampler.generate(mode="testing", task=task_id, num_samples=16, return_triggers=True)

    indices = torch.nonzero(mask == 1).cpu()
    last_indices =  torch.zeros(batch.size(0), dtype=torch.long)
    for key, val in indices.tolist():  # convert to list of pairs
        if val+1 < config.seq_len:
            last_indices[key] = max(last_indices[key], val)
    
    embds = model.embed(batch)
    hidden = model.layers[0](embds)[0]
    if ffn:
        hidden = model.layers[1](hidden)[0]
    else:
        hidden = model.layers[1].MHA(hidden, False)[0]
    
    task_vec = hidden[torch.arange(hidden.size(0)), last_indices].mean(dim=0)

    return task_vec, trans_random[0].squeeze(0)


def get_task_vectors_trigger(config, sampler, model, ffn=True):
    task_vecs  = torch.zeros((config.task.total_trans, config.model.emb_dim), device=config.device)
    for task_id in range(config.task.total_trans):
        tv = get_single_task_vector_trigger(config, sampler, model, task_id, ffn)[0]
        task_vecs[task_id, :] = tv
    
    return task_vecs.detach().cpu()


def get_cos_sim_plot(x: torch.Tensor):
    x_normalized = x / x.norm(dim=1, keepdim=True)

    # Step 2: Compute cosine similarity matrix
    cos_sim = x_normalized @ x_normalized.T  # shape: (64, 64)

    # Step 3: Visualize
    plt.figure(figsize=(6, 5))
    sns.heatmap(cos_sim.cpu().numpy(), cmap='coolwarm', center=0, square=True)
    plt.title("Cosine Similarity Matrix")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.show()


def id_icl_single_error(config, sampler, model, task_id):
    batch, mask, q_toks, trans_random = sampler.generate(mode="testing", task=task_id, num_samples=16, return_triggers=True)

    indices = torch.nonzero(mask == 1).cpu()
    last_indices =  torch.zeros(batch.size(0), dtype=torch.long)
    for key, val in indices.tolist():  # convert to list of pairs
        if val+1 < config.seq_len:
            last_indices[key] = max(last_indices[key], val)
    
    logits = model(batch)[0]
    probs = F.softmax(logits[torch.arange(logits.size(0)), last_indices], dim=-1)
    
    kl = F.kl_div(probs.log(), trans_random[0].squeeze(0), reduction="none").sum(axis=-1).mean()
    return kl.detach().cpu().item()


def get_id_icl_error(config, sampler, model):
    kl_loss = torch.zeros(config.task.total_trans)
    for task_id in range(config.task.total_trans):
        kl_loss[task_id] = id_icl_single_error(config, sampler, model, task_id)
    return kl_loss


def eval_tv_id_error(tvs, sampler, model, config, ffn):
    if ffn:
        kl = F.kl_div(sampler.random_dist.task_pool.cuda().log(), 
            nn.Softmax(dim=-1)(model.output_layer(tvs.to(config.device))), reduction="none").sum(dim=-1)
    else:
        mlp_out = model.layers[1].mlp(tvs.to(config.device))
        kl = F.kl_div(sampler.random_dist.task_pool.cuda().log(), 
                nn.Softmax(dim=-1)(model.output_layer(mlp_out + tvs.to(config.device))), reduction="none").sum(dim=-1)
    return kl.detach().cpu()


def lighten(color, amount=0.5):
    """Blend color with white."""
    white = torch.tensor([1.0, 1.0, 1.0])
    base = torch.tensor(color[:3])
    blended = base + (white - base) * amount
    return (*blended.tolist(), color[3] if len(color) > 3 else 1.0)

def get_pos_loss(model, sampler, mode, folder, n_sumples=1):
    cmap = cm.get_cmap('tab10')  # or 'Set1', 'tab20', etc.

    NUM_SAMPLES = n_sumples

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vocab_size = sampler.num_states
    batch, mask, trigger, trans_random = sampler.generate(num_samples=NUM_SAMPLES, mode=mode, return_triggers=True)
    logits = model(batch)[0]
    preds = F.softmax(logits, dim=-1)

    losses = F.kl_div(preds.log(), trans_random, reduction="none").sum(dim=-1)
    closest = [float("inf")]*NUM_SAMPLES
    if sampler.random_dist.n_tasks > 0:
        if (sampler.random_dist.n_tasks > 4096):
            task_inds = torch.randperm(sampler.random_dist.n_tasks)[:4096].tolist()
        else:
            task_inds = torch.arange(sampler.random_dist.n_tasks).tolist()
        for i in task_inds:
            for b in range(NUM_SAMPLES):
                closest[b] = min(closest[b], F.kl_div(sampler.random_dist.task_pool[i].log(), trans_random[b].cpu(), reduction="none").sum()) 
    
    if sampler.order == 1:
        memory = [0] * NUM_SAMPLES
        for b in range(NUM_SAMPLES):
            memory[b] = F.kl_div(sampler.trans_mat[batch[0][trigger[0]]].log().cpu(), trans_random[b].cpu(), reduction="none").sum()
    
    seqs = torch.arange(batch.size(1))

    for b in range(NUM_SAMPLES):
        valid_indices = torch.nonzero(mask[b].cpu(), as_tuple=True)[0][:-1]
        target = batch[b][valid_indices+1]
        emp_counts = torch.ones(vocab_size)
        emp_losses = torch.zeros_like(valid_indices, dtype=torch.float32)
        for i in range(len(valid_indices)):
            emp_probs = emp_counts / emp_counts.sum()
            emp_losses[i] = F.kl_div(emp_probs.log(), trans_random[b].cpu(), reduction="none").sum()
            token = target[i]
            emp_counts[token] += 1.0
        
        base_color = cmap((2*b)%10)
        model_color = lighten(base_color, amount=0)
        empirical_color = lighten(base_color, amount=0.4)
        icl_color = lighten(cmap((2*b)%10), amount=0.2)
        memory_color = lighten(cmap((2*b)%10), amount=0.4)

        plt.plot(seqs[valid_indices], losses[b][valid_indices].detach().cpu(), marker='o', linestyle='-', 
                 label=f"Model {b}", markersize=3, alpha=0.6, color=model_color)
        
        plt.plot(seqs[valid_indices], emp_losses.detach().cpu(), marker='x', linestyle='dashdot', 
                 label=f"Empirical {b}", markersize=3, alpha=1, color=empirical_color)
        
        if closest[b] != float("inf"):
            plt.axhline(y=closest[b], color=icl_color, linestyle='--', linewidth=1, label=f"ICL {b}")
        if sampler.order == 1:
            plt.axhline(y=memory[b], color=memory_color, linestyle='dotted', linewidth=1, label=f"Memory {b}")
    
    plt.title("KL Divergence over Positions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    image_path = os.path.join(folder, f"loss_over_pos_{mode}_{timestamp}.png")
    plt.savefig(image_path)
    plt.show()
    

    return losses.detach().cpu(), mask.detach().cpu()



#######################
# Latent Markov Chain #
#######################

def get_empirical_transition(model, sampler, task, pos=400, num_samples=1024):
    assert task < sampler.total_trans, "task id out of range"
    assert pos < sampler.seq_len, "position out of range"

    model.eval()
    device = sampler.device

    if num_samples > 1:
        trans_mat_est = torch.zeros((sampler.num_states_order, sampler.num_states), device=device)
        batch, prob = sampler.generate(num_samples=num_samples, mode="testing", task=task)
        logits, _ = model(batch)
        preds = torch.softmax(logits, dim=-1)
        probs = preds[:, pos] # (B, N)
        states = batch[:, (pos-sampler.order+1):(pos+1)] # (B, O)
        states_indices = torch.sum(states * sampler.powers, dim=1)  # (B,)
        trans_mat_est = trans_mat_est.scatter_add(0, states_indices.unsqueeze(1).expand(-1, sampler.num_states), probs)
        counts = torch.bincount(states_indices, minlength=sampler.num_states_order).clamp(min=1)
        trans_mat_est /= counts.unsqueeze(1)  # Normalize by the counts
        return trans_mat_est.detach().cpu()

    else:
        perms = list(product(range(sampler.num_states), repeat=sampler.order))
        batch, prob = sampler.generate(num_samples=len(perms), mode="testing", task=task)
        perms = torch.tensor(perms, device=device)
        
        batch[:, (pos-sampler.order+1):(pos+1)] = perms 
        logits, _ = model(batch)
        preds = torch.softmax(logits, dim=-1)
        probs = preds[:, pos]
        
        return probs.detach().cpu()



def kl_div_ave(P: torch.Tensor, Q: torch.Tensor) -> float:
    """
    Compute the KL divergence between two transition matrices P and Q.
    Q is the true transition matrix, and P is the estimated transition matrix.
    P and Q should be 2D tensors of the same size.
    """
    assert P.size() == Q.size(), "P and Q must have the same size."
    P = P.to(Q.device)  # Ensure P is on the same device as Q
    mu = lau.get_stationary(Q)
    kl = F.kl_div(P.log(), Q, reduction="none").sum(dim=-1)
    return (kl * mu).sum(dim=-1).cpu().item() # Average over the rows


def compute_stationary_distributions(P_batch):
    """
    P_batch: Tensor of shape (M, n, n), batch of transition matrices
    Returns:
        pi_batch: Tensor of shape (M, n), batch of stationary distributions
    """
    M, n, _ = P_batch.shape
    pi_batch = []

    for m in range(M):
        P = P_batch[m]  # (n, n)
        # Transpose because we solve right eigenvector of P^T
        eigenvalues, eigenvectors = torch.linalg.eig(P.T)
        # Find the eigenvector corresponding to eigenvalue 1
        idx = torch.argmin(torch.abs(eigenvalues - 1))
        pi = eigenvectors[:, idx].real  # take real part, just in case
        pi = pi / pi.sum()  # normalize to sum to 1
        pi = torch.clamp(pi, min=0.0)   # Remove tiny negative values due to numerical error
        pi = pi / pi.sum()              # Normalize again after clamping
        pi_batch.append(pi)

    pi_batch = torch.stack(pi_batch, dim=0)
    return pi_batch

def pairwise_kl_divergence(P_batch, pi_batch=None):
    """
    P_batch: Tensor of shape (M, n, n), transition matrices
    pi_batch: Tensor of shape (M, n), stationary distributions
    Returns:
        KL matrix of shape (M, M) where (i,j) is KL(P[i] || P[j]) weighted by pi[i]
    """
    if pi_batch is None:
        pi_batch = compute_stationary_distributions(P_batch)

    M, n, _ = P_batch.shape
    log_P_batch = torch.log(P_batch + 1e-12)  # To avoid log(0)
    
    kl_matrix = torch.zeros(M, M)

    for i in range(M):
        for j in range(M):
            kl_per_row = F.kl_div(
                log_P_batch[i],  # log(P^{(i)}(i,j))
                P_batch[j],      # P^{(j)}(i,j)
                reduction='none'
            ).sum(dim=-1)  # sum over j for each i
            kl = (pi_batch[i] * kl_per_row).sum()  # weighted by pi^{(i)}(i)
            kl_matrix[i, j] = kl.item()
    
    return kl_matrix

def pairwise_kl_divergence_stationary(pi_batch, P_batch=None):
    """
    pi_batch: Tensor of shape (M, n), batch of stationary distributions
    Returns:
        KL matrix of shape (M, M) where (i,j) = KL(pi[i] || pi[j])
    """
    if pi_batch is None and P_batch is not None:
        pi_batch = compute_stationary_distributions(P_batch)
    M, n = pi_batch.shape
    log_pi = torch.log(pi_batch + 1e-12)  # shape (M, n), add epsilon for numerical stability

    kl_matrix = torch.zeros(M, M)

    for i in range(M):
        for j in range(M):
            kl = (pi_batch[i] * (log_pi[i] - log_pi[j])).sum()
            kl_matrix[i, j] = kl.item()

    return kl_matrix


##########################
# Phase Transition Plots #
##########################

def get_loss_lineplot(task_name, vocab_size=20, task_ids=None, 
                      alpha=1.0, seq_len=512, hidden_dim=64, stationary=False):
    
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    # Define normalization and colormap

    plot_ids = []
    plot_paths = []

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids): continue

        if config.vocab_size != vocab_size: continue

        if config.task.alpha != alpha: continue

        if config.seq_len != seq_len: continue

        if config.model.emb_dim != hidden_dim: continue

        if "stationary" in config.task:
            if config.task.stationary != stationary: continue
        elif stationary:
            continue
        
        plot_ids.append(config.task.total_trans)
        plot_paths.append(result_path)
    
    plot_ids = np.array(plot_ids)
    if task_ids is None:
        norm = mcolors.Normalize(vmin=np.log2(min(plot_ids)), vmax=np.log2(max(plot_ids)))
    else:
        norm = mcolors.Normalize(vmin=min(plot_ids), vmax=max(plot_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for path in plot_paths:
        log_path = os.path.join(path, "log.json")
        config_path = os.path.join(path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if task_ids is None:
            value = np.log2(config.task.total_trans)
        else:
            value = config.task.total_trans
        color = cmap(norm(value))
        
        # Plot 1: OOD Loss
        ax2.plot(log_data["eval/step"], log_data["eval/OODLoss"], color=color, alpha=0.6)
        
        # Plot 2: Some other metric (e.g., ID Loss)
        ax1.plot(log_data["eval/step"], log_data["eval/loss"], color=color, alpha=0.6)



    # Customize each subplot
    ax2.set_xscale("log")
    ax2.set_title("OOD Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=11)

    ax1.set_xscale("log")
    ax1.set_title("ID Loss", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Steps", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)

    plt.subplots_adjust(wspace=0.02)  # Try 0 for zero gap
    # Add colorbar to the whole figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical')
    if task_ids is None:
        cbar.set_label("Log Number of Mixtures", fontsize=12)
    else:
        cbar.set_label("Number of Mixtures", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"loss_lineplots_{vocab_size}_{alpha}.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"loss_lineplots_{hash_array(np.array(plot_ids))}_{vocab_size}_{alpha}.png"))
    plt.show()

# TODO: outdated!
def get_loss_heatmap_data(task_name, measure, task_ids=None):
    measure_name = {"ood": "OODLoss", "id": "loss", "ih": "ih_score", "pth": "pth_score"}
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Collect all steps and transitions first
    all_steps = set()
    all_trans = set()
    data_dict = {}

    for folder in folders:
        result_path = os.path.join(folder_path, folder)
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")

        log_data = load_log(log_path)
        config = load_config(config_path)

        trans = config.task.total_trans
        if (task_ids is not None) and (trans not in task_ids):
            continue

        steps = log_data["eval/step"]
        losses = log_data[f"eval/{measure_name[measure]}"]  # or "eval/loss" for ID

        for step, loss in zip(steps, losses):
            all_steps.add(step)
            all_trans.add(trans)
            data_dict[(trans, step)] = loss

    # Sort axes
    sorted_steps = sorted(all_steps)
    sorted_trans = sorted(all_trans)

    # Create heatmap matrix
    heatmap = np.full((len(sorted_trans), len(sorted_steps)), np.nan)

    trans_idx = {v: i for i, v in enumerate(sorted_trans)}
    step_idx = {v: i for i, v in enumerate(sorted_steps)}

    for (trans, step), loss in data_dict.items():
        i = trans_idx[trans]
        j = step_idx[step]
        heatmap[i, j] = loss
    return heatmap, sorted_steps, sorted_trans


# TODO: outdated!
def get_loss_heatmap(task_name, measure, task_ids=None, log_scale=False):
    measure_name = {"ood": "OODLoss", "id": "loss", "ih": "ih_score", "pth": "pth_score"}
    folder_path = os.path.join("results", task_name)
    measure_title = {"ood": "OOD Loss", "id": "ID Loss", "ih": "Induction Head Score", "pth": "Previous Token Head Score"}
    
    heatmap, sorted_steps, sorted_trans = get_loss_heatmap_data(task_name, measure, task_ids=task_ids)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("plasma")
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, origin='lower')

    if log_scale:
        ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Transitions")
    ax.set_xticks(np.arange(0, len(sorted_steps), 25))
    ax.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax.set_xticklabels(sorted_steps[::25], rotation=45)
    ax.set_yticklabels(sorted_trans[::5])
    ax.set_title(f"{measure_title[measure]} Heatmap")

    cbar = fig.colorbar(im, ax=ax)
    if measure in ["ih", "pth"]:
        cbar.set_label("Attention Score")
    else:
        cbar.set_label("Loss")

    plt.tight_layout()
    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"{measure_name[measure]}_heatmap.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"{measure_name[measure]}_heatmap_{hash_array(task_ids)}.png"))
    plt.show()


# TODO: outdated!
def get_loss_heatmap_dual(task_name, task_ids=None, log_scale=False):
    folder_path = os.path.join("results", task_name)
    heatmap_ood, sorted_steps, sorted_trans = get_loss_heatmap_data(task_name, "ood", task_ids=task_ids)
    heatmap_id, _, _ = get_loss_heatmap_data(task_name, "id", task_ids=task_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # Shared color scale: choose LogNorm or Normalize
    data_all = np.concatenate([heatmap_id, heatmap_ood])
    if log_scale:
        norm = LogNorm(vmin=np.nanmin(data_all[data_all > 0]), vmax=np.nanmax(data_all))
    else:
        norm = mcolors.Normalize(vmin=np.nanmin(data_all), vmax=np.nanmax(data_all))

    # Heatmap 1: ID Loss
    im1 = ax1.imshow(
        heatmap_id,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        norm=norm
    )


    ax1.set_title("ID Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Transitions")
    ax1.set_xticks(np.arange(0, len(sorted_steps), 50))
    ax1.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax1.set_xticklabels(sorted_steps[::50], rotation=45)
    ax1.set_yticklabels(sorted_trans[::5])
    ax1.invert_yaxis()

    # Heatmap 2: OOD Loss
    im2 = ax2.imshow(
        heatmap_ood,
        aspect='auto',
        origin='lower',
        cmap='plasma',
        norm=norm
    )

    ax2.set_title("OOD Loss")
    ax2.set_xlabel("Step")
    ax2.set_xticks(np.arange(0, len(sorted_steps), 50))
    ax2.set_yticks(np.arange(0, len(sorted_trans), 5))
    ax2.set_xticklabels(sorted_steps[::50], rotation=45)
    ax2.set_yticklabels(sorted_trans[::5])
    ax2.invert_yaxis()

    plt.subplots_adjust(wspace=0.02)

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label("Loss")

    

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"losses_heatmap.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"losses_heatmap_{hash_array(task_ids)}.png"))
    plt.show()



def get_attn_score_lineplot(task_name, vocab_size=20, task_ids=None,
                            alpha=1.0, seq_len=512, hidden_dim=64, stationary=False):
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    plot_ids = []
    plot_paths = []

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids): continue

        if config.vocab_size != vocab_size: continue

        if config.task.alpha != alpha: continue

        if config.seq_len != seq_len: continue

        if config.model.emb_dim != hidden_dim: continue

        if "stationary" in config.task:
            if config.task.stationary != stationary: continue
        elif stationary:
            continue
        
        plot_ids.append(config.task.total_trans)
        plot_paths.append(result_path)
    
    plot_ids = np.array(plot_ids)
    if task_ids is None:
        norm = mcolors.Normalize(vmin=np.log2(min(plot_ids)), vmax=np.log2(max(plot_ids)))
    else:
        norm = mcolors.Normalize(vmin=min(plot_ids), vmax=max(plot_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for path in plot_paths:
        log_path = os.path.join(path, "log.json")
        config_path = os.path.join(path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if task_ids is None:
            value = np.log2(config.task.total_trans)
        else:
            value = config.task.total_trans
        color = cmap(norm(value))
        
        # Plot 1: IH Score
        ax2.plot(log_data["eval/step"], log_data["eval/ih_score"], color=color, alpha=0.6)
        
        # Plot 2: PTH Score
        ax1.plot(log_data["eval/step"], log_data["eval/pth_score"], color=color, alpha=0.6)


    # Customize each subplot
    ax2.set_xscale("log")
    ax2.set_title("Induction Head Score vs Step")
    ax2.set_xlabel("Step")

    ax1.set_xscale("log")
    ax1.set_title("Previous Token Head Score vs Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Attention Score")

    plt.subplots_adjust(wspace=0.02)  # Try 0 for zero gap
    # Add colorbar to the whole figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label("Total Transitions")

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, f"attn_scores_lineplots_{vocab_size}_{alpha}.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"attn_scores_lineplots_{hash_array(plot_ids)}_{vocab_size}_{alpha}.png"))
    plt.show()



########################   
# Loss along positions #
########################

def kl_plot(model, sampler, task=None, num_samples=100, pos=None):
    if task == None:
        batch, _, tasks = sampler.generate(mode="testing", task=task, num_samples=num_samples)
        B,T = batch.shape
        K = sampler.num_states
        tasks_exp = tasks[:, None].expand(B, T)        # shape: (B, T)
        batch_exp = batch                              # shape: (B, T)
        
        # Flatten for advanced indexing
        flat_tasks = tasks_exp.reshape(-1)             # (B*T,)
        flat_batch = batch_exp.reshape(-1)             # (B*T,)
        
        # Gather rows from trans_mat using tasks and batch
        selected = sampler.trans_mat[flat_tasks, flat_batch]   # shape: (B*T, K)
        
        trans_probs = selected.view(B, T, K)
    else:
        batch, _ = sampler.generate(mode="testing", task=task, num_samples=num_samples)
        trans_probs = sampler.trans_mat[task][batch]
        
    kl_losses = F.kl_div(nn.Softmax(dim=-1)(model(batch)[0]).log(), trans_probs, reduction="none").sum(dim=-1).detach().cpu().numpy() 
    
    if pos is not None:
        kl_losses = kl_losses[:, :pos]

    # Create a 1D NumPy array
    arr = np.arange(kl_losses.shape[1])
    
    mean_losses = np.mean(kl_losses, axis=0)
    std_losses = np.std(kl_losses, axis=0)
    
    # Plot the array
    plt.plot(arr, mean_losses)
    plt.fill_between(arr, np.maximum(mean_losses - 2*std_losses, 0), mean_losses + 2*std_losses, color='blue', alpha=0.3, label="Mean ± 2 std")
    
    plt.grid()
    if task is not None:
        plt.title(f"Average KL-divergence for task {task}")
    else:
        plt.title("Average KL-divergence over all tasks")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()



def predictive_distribution_batched(x_seq_batch, transition_matrices):
    """
    Computes batched Pr(x_{t+1} | x_{1:t}) in log-space for multiple sequences.

    Args:
        x_seq_batch: LongTensor of shape (B, T), each row is a sequence x_{1:t}
        transition_matrices: FloatTensor of shape (K, N, N), each is a transition matrix P^{(k)}

    Returns:
        pred_probs: Tensor of shape (B, N), predictive distribution for x_{t+1}
    """
    B, T = x_seq_batch.shape
    K, N, _ = transition_matrices.shape

    # Step 1: Compute log_weights[b, k] = log-likelihood of x_{1:T} under model k
    # We'll gather P_{x_{tau-1}, x_tau}^{(k)} for each tau and compute log-product
    log_P = torch.log(torch.clamp(transition_matrices, min=1e-40))  # (K, N, N)

    # Expand for indexing
    x_prev = x_seq_batch[:, :-1]  # (B, T-1)
    x_curr = x_seq_batch[:, 1:]   # (B, T-1)

    # Gather transition probabilities for each model k
    # log_probs: (B, K, T-1)
    log_probs = []
    for k in range(K):
        # log_P_k[x_prev, x_curr] → (B, T-1)
        log_Pk = log_P[k]  # (N, N)
        log_prob_k = log_Pk[x_prev, x_curr]  # batch-wise indexing
        log_probs.append(log_prob_k.unsqueeze(1))  # shape (B, 1, T-1)

    log_probs = torch.cat(log_probs, dim=1)  # (B, K, T-1)
    log_weights = torch.sum(log_probs, dim=2)  # (B, K)

    # Normalize: log_softmax over K models
    log_weights = F.log_softmax(log_weights, dim=1)  # (B, K)

    # Step 2: Compute log predictive probabilities for each next state j
    x_t = x_seq_batch[:, -1]  # (B,)
    log_pred = torch.full((B, N), -float('inf'), device=x_seq_batch.device)

    for k in range(K):
        # log_P_k[x_t, :] shape: (B, N)
        log_Pk = log_P[k]  # (N, N)
        log_Pk_xt = log_Pk[x_t]  # (B, N)
        # Add log weight
        log_term = log_weights[:, k].unsqueeze(1) + log_Pk_xt  # (B, N)
        log_pred = torch.logaddexp(log_pred, log_term)

    # Final probability
    pred_probs = torch.exp(log_pred)  # (B, N)
    return pred_probs

#TODO: build a hashmap for faster access

def bayes_emp_plot(vocab_size=10, total_trans=10, file_path=None, 
                   alpha=0.5, seq_len=512, hidden_dim=64, stationary=False,
                   task=None, num_samples=2000, low=1, high=200, 
                   emp=True, bayes=True, unigram=True):
    assert emp or bayes or unigram, "At least one of emp or bayes or unigram should be True"

    if file_path is not None:
        folder_name = file_path
        _, sampler, _ = load_everything("latent", folder_name)
    else:
        folder_name = get_config(vocab_size=vocab_size, total_trans=total_trans, 
                                alpha=alpha, seq_len=seq_len, hidden_dim=hidden_dim, stationary=stationary)

        if len(folder_name) > 0:
            _, sampler, _ = load_everything("latent", folder_name[0])
        else:
            print("The configuration does not exist.")
            return

    if task is None:
        batch, _, latents = sampler.generate(mode="testing", num_samples=num_samples, task=task)
    else:
        batch, _ = sampler.generate(mode="testing", num_samples=num_samples, task=task)
        latents = task

    batch_size, seq_len = batch.shape
    trans_mat_est = torch.ones((batch_size, sampler.num_states, sampler.num_states), device=batch.device)
    uni_trans_mat_est = torch.ones((batch_size, sampler.num_states), device=batch.device)
    next_states = batch[:, 1:]  # (B, T-1)

    values = torch.ones(batch_size, dtype=torch.float, device=batch.device)  # Same size as positions

    emp_mean_losses = np.zeros(high-low)
    emp_std_losses = np.zeros(high-low)
    bayes_mean_losses = np.zeros(high-low)
    bayes_std_losses = np.zeros(high-low)

    if unigram:
        unigram_mean_losses = np.zeros(high-low)
        unigram_std_losses = np.zeros(high-low)
    
    for t in range(low):
        trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t], next_states[:,t]), values, accumulate=True)
        uni_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t]), values, accumulate=True)

    for pos in range(low, high):
        pred_probs = trans_mat_est / trans_mat_est.sum(dim=-1, keepdim=True)
        trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], next_states[:,pos]), values, accumulate=True)
        uni_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos]), values, accumulate=True)

        
        kl_div = F.kl_div(pred_probs[torch.arange(batch_size),
                                     batch[:,pos]].log(), sampler.trans_mat[latents, batch[:,pos]], reduction="none").sum(dim=-1)
        emp_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
        emp_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()

        if unigram:
            pred_probs = uni_trans_mat_est / uni_trans_mat_est.sum(dim=-1, keepdim=True)

            kl_div = F.kl_div(pred_probs[torch.arange(batch_size)].log(), 
                            sampler.stationary[latents], reduction="none").sum(dim=-1)
        
            unigram_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            unigram_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()

        pred_probs = predictive_distribution_batched(batch[:,:pos], sampler.trans_mat)
        kl_div = F.kl_div(pred_probs.log(), sampler.trans_mat[latents, batch[:,pos-1]], reduction="none").sum(dim=-1)
        bayes_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
        bayes_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()
    
    arr = np.arange(low, high)
    if emp:
        plt.plot(arr, emp_mean_losses)
        plt.fill_between(arr, np.maximum(emp_mean_losses - 2*emp_std_losses, 0), emp_mean_losses + 2*emp_std_losses, color='blue', alpha=0.3, label="Emp Mean ± 2 std")

    if bayes:
        plt.plot(arr, bayes_mean_losses)
        plt.fill_between(arr, np.maximum(bayes_mean_losses - 2*bayes_std_losses, 0), bayes_mean_losses + 2*bayes_mean_losses, color='orange', alpha=0.3, label="Bayes Mean ± 2 std")
    
    if unigram:
        plt.plot(arr, unigram_mean_losses)
        plt.fill_between(arr, np.maximum(unigram_mean_losses - 2*unigram_std_losses, 0), unigram_mean_losses + 2*unigram_mean_losses, color='green', alpha=0.3, label="Unigram Mean ± 2 std")
    
    plt.grid()
    if task is not None:
        plt.title(f"Average KL-divergence for task {task}")
    else:
        plt.title("Average KL-divergence over all tasks")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()


def bayes_emp_ood_plot(
        vocab_size, total_trans, 
        alpha=0.5, seq_len=512, hidden_dim=64, num_samples=2000, low=1, high=200, 
        emp=True, bayes=True):
    assert emp or bayes, "At least one of emp or bayes should be True"

    folder_name = get_config(vocab_size=vocab_size, total_trans=total_trans, 
                             alpha=alpha, seq_len=seq_len, hidden_dim=hidden_dim)

    if len(folder_name) > 0:
        _, sampler, _ = load_everything("latent", folder_name[0])
    else:
        print("The configuration does not exist.")
        return

    batch, _, trans_mat = sampler.generate(mode="ood", num_samples=num_samples, return_trans_mat=True)

    batch_size, seq_len = batch.shape
    trans_mat_est = torch.ones((batch_size, sampler.num_states, sampler.num_states), device=batch.device)
    next_states = batch[:, 1:]  # (B, T-1)

    values = torch.ones(batch_size, dtype=torch.float, device=batch.device)  # Same size as positions

    emp_mean_losses = np.zeros(high-low)
    emp_std_losses = np.zeros(high-low)
    bayes_mean_losses = np.zeros(high-low)
    bayes_std_losses = np.zeros(high-low)
    
    for t in range(low):
        trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t], next_states[:,t]), values, accumulate=True)

    for pos in range(low, high):
        pred_probs = trans_mat_est / trans_mat_est.sum(dim=-1, keepdim=True)
        trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], next_states[:,pos]), values, accumulate=True)
        
        kl_div = F.kl_div(pred_probs[torch.arange(batch_size),
                                     batch[:,pos]].log(), 
                                     trans_mat[torch.arange(batch_size), batch[:,pos]], reduction="none").sum(dim=-1)
        emp_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
        emp_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()

        pred_probs = predictive_distribution_batched(batch[:,:pos], sampler.trans_mat)
        kl_div = F.kl_div(pred_probs.log(), trans_mat[torch.arange(batch_size), batch[:,pos-1]], reduction="none").sum(dim=-1)
        bayes_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
        bayes_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()
    
    arr = np.arange(low, high)
    if emp:
        plt.plot(arr, emp_mean_losses)
        plt.fill_between(arr, np.maximum(emp_mean_losses - 2*emp_std_losses, 0), emp_mean_losses + 2*emp_std_losses, color='blue', alpha=0.3, label="Emp Mean ± 2 std")

    if bayes:
        plt.plot(arr, bayes_mean_losses)
        plt.fill_between(arr, np.maximum(bayes_mean_losses - 2*bayes_std_losses, 0), bayes_mean_losses + 2*bayes_mean_losses, color='orange', alpha=0.3, label="Bayes Mean ± 2 std")
    
    plt.grid()
    plt.title("Average KL-divergence over all tasks (OOD)")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()


def all_kl_plot(vocab_size=10, total_trans=10, file_path=None, task=None, 
                seq_len=512, alpha=0.5, hidden_dim=64, stationary=False,
                num_samples=2000, low=1, high=200, 
                unigram=True,
                truth=True, unif=True):
    if file_path is not None:
        model, sampler, _ = load_everything("latent", file_path)
    else:
        folder_name = get_config(vocab_size=vocab_size, total_trans=total_trans, 
                                alpha=alpha, seq_len=seq_len, hidden_dim=hidden_dim, stationary=stationary)
        if len(folder_name) > 0:
            model, sampler, _ = load_everything("latent", folder_name[0])
        else:
            print("The configuration does not exist.")
            return
    if task is None:
        batch, _, tasks = sampler.generate(mode="testing", task=task, num_samples=num_samples)
        B,T = batch.shape
        N = sampler.num_states
        tasks_exp = tasks[:, None].expand(B, T)        # shape: (B, T)
        batch_exp = batch                              # shape: (B, T)
        
        # Flatten for advanced indexing
        flat_tasks = tasks_exp.reshape(-1)             # (B*T,)
        flat_batch = batch_exp.reshape(-1)             # (B*T,)
        
        # Gather rows from trans_mat using tasks and batch
        selected = sampler.trans_mat[flat_tasks, flat_batch]   # shape: (B*T, N)
        
        trans_probs = selected.view(B, T, N)
    else:
        batch, _ = sampler.generate(mode="testing", num_samples=num_samples, task=task)
        tasks = task
        trans_probs = sampler.trans_mat[task][batch]
    
    cmap = plt.get_cmap('tab10')
        
    model_pred_probs = nn.Softmax(dim=-1)(model(batch)[0]) # (B, T, N)
    trans_kl_losses = F.kl_div(model_pred_probs.log(), trans_probs, reduction="none").sum(dim=-1).detach().cpu().numpy() # (B, T)
    unif_probs = torch.ones_like(model_pred_probs) / sampler.num_states # (B, T, N)
    unif_kl_losses = F.kl_div(model_pred_probs.log(), unif_probs, reduction="none").sum(dim=-1).detach().cpu().numpy() # (B, T)
    unif_kl_losses = unif_kl_losses[:, (low-1):(high-1)]
    
    trans_kl_losses = trans_kl_losses[:, (low-1):(high-1)]

    # Create a 1D NumPy array
    arr = np.arange(low, high)
    
    trans_mean_losses = np.mean(trans_kl_losses, axis=0)
    trans_std_losses = np.std(trans_kl_losses, axis=0)

    unif_mean_losses = np.mean(unif_kl_losses, axis=0)
    unif_std_losses = np.std(unif_kl_losses, axis=0)

    batch_size, seq_len = batch.shape
    emp_trans_mat_est = torch.ones((batch_size, sampler.num_states, sampler.num_states), device=batch.device)
    uni_trans_mat_est = torch.ones((batch_size, sampler.num_states), device=batch.device)
    next_states = batch[:, 1:]  # (B, T-1)

    values = torch.ones(batch_size, dtype=torch.float, device=batch.device)  # Same size as positions

    emp_mean_losses = np.zeros(high-low)
    emp_std_losses = np.zeros(high-low)

    unigram_mean_losses = np.zeros(high-low)
    unigram_std_losses = np.zeros(high-low)

    bayes_mean_losses = np.zeros(high-low)
    bayes_std_losses = np.zeros(high-low)
    
    for t in range(low):
        emp_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t], next_states[:,t]), values, accumulate=True)
        uni_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,t]), values, accumulate=True)


    for pos in range(low, high):
        emp_probs = emp_trans_mat_est / emp_trans_mat_est.sum(dim=-1, keepdim=True) 
        unigram_probs = uni_trans_mat_est / uni_trans_mat_est.sum(dim=-1, keepdim=True) 
        emp_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos], next_states[:,pos]), values, accumulate=True)
        uni_trans_mat_est.index_put_((torch.arange(batch_size), batch[:,pos]), values, accumulate=True)

        kl_div = F.kl_div(model_pred_probs[:, pos].log(), 
                          emp_probs[torch.arange(batch_size), batch[:,pos]],
                          reduction="none").sum(dim=-1).detach().cpu().numpy()
        
        emp_mean_losses[pos-low] = kl_div.mean()
        emp_std_losses[pos-low] = kl_div.std()


        kl_div = F.kl_div(model_pred_probs[:, pos].log(),
                          unigram_probs[torch.arange(batch_size)],
                          reduction="none").sum(dim=-1).detach().cpu().numpy()
        unigram_mean_losses[pos-low] = kl_div.mean()
        unigram_std_losses[pos-low] = kl_div.std()

        if (total_trans < 300) and truth:

            bayes_probs = predictive_distribution_batched(batch[:,:pos], sampler.trans_mat)
            kl_div = F.kl_div(model_pred_probs[:, pos-1].log(), 
                            bayes_probs, reduction="none").sum(dim=-1)
            bayes_mean_losses[pos-low] = kl_div.mean().detach().cpu().numpy()
            bayes_std_losses[pos-low] = kl_div.std().detach().cpu().numpy()
    
    arr = np.arange(low, high)

    plt.plot(arr, emp_mean_losses, color=cmap(0))
    plt.fill_between(arr, np.maximum(emp_mean_losses - emp_std_losses, 0), 
                     emp_mean_losses + emp_std_losses, color=cmap(0), 
                     alpha=0.3, label="Emp Mean ± std")

    if truth:
        plt.plot(arr, trans_mean_losses, color=cmap(1))
        plt.fill_between(arr, np.maximum(trans_mean_losses - trans_std_losses, 0), 
                        trans_mean_losses + trans_std_losses, color=cmap(1), 
                        alpha=0.3, label="Truth Mean ± std")
    
    if (total_trans < 300) and truth:
        plt.plot(arr, bayes_mean_losses, color=cmap(2))
        plt.fill_between(arr, np.maximum(bayes_mean_losses - bayes_std_losses, 0), 
                        bayes_mean_losses + bayes_std_losses, color=cmap(2), 
                        alpha=0.3, label="Bayes Mean ± std")
    if unif:
        plt.plot(arr, unif_mean_losses, color=cmap(3))
        plt.fill_between(arr, np.maximum(unif_mean_losses - unif_std_losses, 0),
                        unif_mean_losses + unif_std_losses, color=cmap(3),
                        alpha=0.3, label="Uniform Mean ± std")
    
    if unigram:
        plt.plot(arr, unigram_mean_losses, color=cmap(4))
        plt.fill_between(arr, np.maximum(unigram_mean_losses - unigram_std_losses, 0),
                        unigram_mean_losses + unigram_std_losses, color=cmap(4),
                        alpha=0.3, label="Unigram Mean ± std")
    
    plt.grid()
    if task is not None:
        plt.title(f"Average KL-divergence for task {task} (Total: {sampler.total_trans})")
    else:
        plt.title(f"Average KL-divergence over all tasks (Total: {sampler.total_trans})")
    plt.xlabel("Positions")
    plt.ylabel("KL-divergence")
    plt.legend()
    plt.tight_layout()
    
    plt.show()