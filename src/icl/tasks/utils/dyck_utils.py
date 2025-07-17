import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import os

from icl.tasks import DyckBayes
from icl.config import get_config_base
from icl.utils.notebook_utils import load_everything
from icl.utils.basic import get_hash
from icl.utils.train_utils import get_attn_at_layer_base

def get_probs(config, sampler, model, B=32, mode="ood", dyck_mask=None):
    torch.cuda.empty_cache()
    eps = 1e-8
    batch, mask = sampler.generate(mode=mode, num_samples=B, dyck_mask=dyck_mask)
    output = model(batch)
    model_probs = nn.Softmax(dim=-1)(output).clamp(min=eps)
    if "pad" in config.task and config.task.pad:
        batch = batch[:, ::2]
        mask = mask[:, ::2]
        output = output[:, 1::2, :]
        model_probs = model_probs[:, 1::2, :]
    elif "bos_pad" in config.task and config.task.bos_pad:
        batch = batch[:, 1:]
        mask = mask[:, 1:]
        output = output[:, 1:-1, :]
        model_probs = model_probs[:, 1:-1, :]
    else:
        model_probs = model_probs[:, :-1, :]
    valid_mask = (mask > 0)
    batch_idx, time_idx = torch.where(valid_mask)
    
    batch_idx = batch_idx.reshape(B, -1)
    time_idx = time_idx.reshape(B, -1)
    batch_idx = batch_idx[:, 1:].reshape(-1)
    time_idx = time_idx[:, 1:].reshape(-1)
    model_probs_masked = model_probs[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)
    
    bayes_ood = DyckBayes(config, sampler, flag=True)
    bayes_ood_probs = bayes_ood.pos_prob(batch).clamp(min=eps)
    bayes_ood_probs_masked = bayes_ood_probs[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)
    
    bayes_id = DyckBayes(config, sampler)
    bayes_id_probs = bayes_id.pos_prob(batch).clamp(min=eps)
    bayes_id_probs_masked = bayes_id_probs[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)

    return model_probs, model_probs_masked, bayes_ood_probs, bayes_ood_probs_masked, bayes_id_probs, bayes_id_probs_masked

def get_ces_masked(config, sampler, model, B=512, mode="ood", dyck_mask=None):
    torch.cuda.empty_cache()
    eps = 1e-8
    batch, mask = sampler.generate(mode=mode, num_samples=B, dyck_mask=dyck_mask)
    output = model(batch)
    if "pad" in config.task and config.task.pad:
        batch = batch[:, ::2]
        mask = mask[:, ::2]
        output = output[:, 1::2, :]
    elif "bos_pad" in config.task and config.task.bos_pad:
        batch = batch[:, 1:]
        mask = mask[:, 1:]
        output = output[:, 1:, :]
    model_probs = nn.Softmax(dim=-1)(output).clamp(min=eps)
    valid_mask = (mask > 0)
    batch_idx, time_idx = torch.where(valid_mask)
    
    batch_idx = batch_idx.reshape(B, -1)
    time_idx = time_idx.reshape(B, -1)
    batch_idx = batch_idx[:, 1:].reshape(-1)
    time_idx = time_idx[:, 1:].reshape(-1)
    target = batch[batch_idx, time_idx].reshape(B, -1)
    out_masked = output[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)
    model_ce = F.cross_entropy(out_masked.reshape(-1, config.vocab_size), 
                    target.reshape(-1), reduction="none").reshape(B, -1).mean(dim=0)
    
    bayes_ood = DyckBayes(config, sampler, flag=True)
    bayes_ood_probs = bayes_ood.pos_prob(batch).clamp(min=eps)
    bayes_ood_probs_masked = bayes_ood_probs[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)
    bayes_ood_ce = F.cross_entropy(bayes_ood_probs_masked.log().reshape(-1, config.vocab_size), 
                    target.reshape(-1), reduction="none").reshape(B, -1).mean(dim=0)
    
    bayes_id = DyckBayes(config, sampler)
    bayes_id_probs = bayes_id.pos_prob(batch).clamp(min=eps)
    bayes_id_probs_masked = bayes_id_probs[batch_idx, time_idx-1, :].reshape(B, -1, config.vocab_size)
    bayes_id_ce = F.cross_entropy(bayes_id_probs_masked.log().reshape(-1, config.vocab_size), 
                    target.reshape(-1), reduction="none").reshape(B, -1).mean(dim=0)

    return model_ce, bayes_id_ce, bayes_ood_ce

def get_ces(config, sampler, model, B=512, mode="ood", dyck_mask=None):
    torch.cuda.empty_cache()
    eps = 1e-8
    batch, mask = sampler.generate(mode=mode, num_samples=B, dyck_mask=dyck_mask)
    output = model(batch)
    
    if "pad" in config.task and config.task.pad:
        batch = batch[:, ::2]
        mask = mask[:, ::2]
        output = output[:, 1::2, :]
    elif "bos_pad" in config.task and config.task.bos_pad:
        batch = batch[:, 1:]
        mask = mask[:, 1:]
        output = output[:, 1:-1, :]
    else:
        output = output[:, :-1, :] 

    target = batch[:,1:].reshape(-1)
    model_ce = F.cross_entropy(output.reshape(-1, config.vocab_size), 
                    target, reduction="none").reshape(B, -1).mean(dim=0)
    
    bayes_ood = DyckBayes(config, sampler, flag=True)
    
    bayes_ood_probs = bayes_ood.pos_prob(batch).clamp(min=eps)
    bayes_ood_ce = F.cross_entropy(bayes_ood_probs.log().reshape(-1, config.vocab_size), 
                    target, reduction="none").reshape(B, -1).mean(dim=0)
    
    bayes_id = DyckBayes(config, sampler)
    bayes_id_probs = bayes_id.pos_prob(batch).clamp(min=eps)
    bayes_id_ce = F.cross_entropy(bayes_id_probs.log().reshape(-1, config.vocab_size), 
                    target, reduction="none").reshape(B, -1).mean(dim=0)

    return model_ce, bayes_id_ce, bayes_ood_ce

def get_bayes_kl(bayes_ood_probs, bayes_id_probs):
    kl_bayes = (bayes_ood_probs * (-bayes_id_probs.log() + bayes_ood_probs.log())).sum(dim=-1)  # shape (B, T)
    kl_bayes = kl_bayes.mean(dim=0).detach().cpu().numpy()  # shape (T,)
    
    # Plot
    plt.plot(kl_bayes, label="Bayes")
    plt.xlabel("Positions")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence over Positions")
    plt.legend()
    plt.grid(True)
    plt.show()


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_ces(model_ce, bayes_id_ce, bayes_ood_ce, window_size=5, pos=None):
    model_ce_np = model_ce.detach().cpu().numpy()
    bayes_id_ce_np = bayes_id_ce.cpu().numpy()
    bayes_ood_ce_np = bayes_ood_ce.cpu().numpy()

    if pos is not None:
        model_ce_np = model_ce_np[:pos]
        bayes_id_ce_np = bayes_id_ce_np[:pos]
        bayes_ood_ce_np = bayes_ood_ce_np[:pos]


    # Apply moving average
    model_ce_smooth = moving_average(model_ce_np, window_size)
    bayes_id_ce_smooth = moving_average(bayes_id_ce_np, window_size)
    bayes_ood_ce_smooth = moving_average(bayes_ood_ce_np, window_size)

    # x-axis for smoothed curves
    x_smooth = np.arange(len(model_ce_smooth))
    x = np.arange(len(model_ce_np))

    plt.figure(figsize=(9, 5))

    # Define consistent colors
    colors = {
        "OOD": "tab:blue",
        "ID": "tab:orange",
        "model": "tab:green"
    }

    # Plot raw curves (faint)
    plt.plot(x, bayes_ood_ce_np, label=None, color=colors["OOD"], alpha=0.2)
    plt.plot(x, bayes_id_ce_np, label=None, color=colors["ID"], alpha=0.2)
    plt.plot(x, model_ce_np, label=None, color=colors["model"], alpha=0.2)

    # Plot smoothed curves (prominent)
    plt.plot(x_smooth, bayes_ood_ce_smooth, label="OOD (smoothed)", color=colors["OOD"], alpha=0.9)
    plt.plot(x_smooth, bayes_id_ce_smooth, label="ID (smoothed)", color=colors["ID"], alpha=0.9)
    plt.plot(x_smooth, model_ce_smooth, label="model (smoothed)", color=colors["model"], alpha=0.9)

    plt.xlabel("Positions")
    plt.ylabel("Cross Entropy Loss")
    plt.title(f"Cross Entropy Loss over Positions\nSmoothed with Moving Average (window={window_size})")
    plt.legend()
    plt.grid(True)
    
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

def get_model_kl(model_probs, bayes_ood_probs, bayes_id_probs, k, pos=None):
    kl_ood = (model_probs * (model_probs.log() - bayes_ood_probs.log())).sum(dim=-1)  # shape (B, T)
    kl_id = (model_probs * (model_probs.log() - bayes_id_probs.log())).sum(dim=-1)  # shape (B, T)
    kl_ood = kl_ood.mean(dim=0).detach().cpu().numpy()  # shape (T,)
    kl_id = kl_id.mean(dim=0).detach().cpu().numpy()  # shape (T,)
    
    # Plot
    if pos is not None:
        kl_ood = kl_ood[:pos]
        kl_id = kl_id[:pos]

    plt.plot(kl_ood, label="OOD")
    plt.plot(kl_id, label="ID")
    plt.xlabel("Position")
    plt.ylabel("KL Divergence")
    plt.title(f"KL Divergence over Positions (Pool Size: {k})")
    
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    plt.show()

def get_and_plot_ces(config, sampler, model, masked, mode="eval", window_size=3, pos=None, dyck_mask=None):
    if masked:
        model_ce, bayes_id_ce, bayes_ood_ce = get_ces_masked(config, sampler, model, mode=mode, dyck_mask=dyck_mask)
        window_size = 1
    else:
        model_ce, bayes_id_ce, bayes_ood_ce = get_ces(config, sampler, model, mode=mode, dyck_mask=dyck_mask)
    plot_ces(model_ce, bayes_id_ce, bayes_ood_ce, window_size=window_size, pos=pos)

def get_and_plot_kl(config, sampler, model, masked, mode="eval", pos=None, dyck_mask=None):
    results = get_probs(config, sampler, model, mode=mode, dyck_mask=dyck_mask)
    model_probs, model_probs_masked, bayes_ood_probs, bayes_ood_probs_masked, bayes_id_probs, bayes_id_probs_masked = results
    k = config.task.total_trans
    if masked:
        get_model_kl(model_probs_masked, bayes_ood_probs_masked, bayes_id_probs_masked, k, pos=pos)
    else:
        get_model_kl(model_probs, bayes_ood_probs, bayes_id_probs, k, pos=pos)

def get_result(total_trans):
    config = get_config_base()
    config.task.total_trans = total_trans
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)  
    
    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        model, sampler, config, log_data = load_everything("dyck", exp_name, get_log=True)
        return model, sampler, config, log_data
    else:
        print("No result!")
        raise FileNotFoundError(f"Log file not found at {log_path}. Please run the training first.") 


def get_ood_plot(ks):
    config = get_config_base()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    cmap = cm.get_cmap('viridis', len(ks))  # choose your colormap

    for idx, k in enumerate(ks):
        config.task.total_trans = k
        _, _, _, log_data = get_result(k)
        color = cmap(idx)
        ax[0].plot(log_data['eval/step'], log_data['eval/IDLoss'], label=f"k={k}", color=color)
        ax[1].plot(log_data['eval/step'], log_data['eval/OODLoss'], label=f"k={k}", color=color)

    ax[0].set_title("Dyck ID Loss")
    ax[1].set_title("Dyck OOD Loss")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[1].set_xlabel("Step")

    # Optional: add legends with fewer labels to avoid clutter
    ax[0].legend(fontsize=10, loc='best', ncol=2)

    plt.tight_layout()
    path_prefix = os.path.join("results", "dyck")
    path = os.path.join(path_prefix, "dyck_ood_plot.png")
    plt.savefig(path, dpi=300)
    plt.show()


def rolling_var(X, window):
    B, T, D = X.shape
    X2 = X ** 2

    # Pad with zero at the start for clean indexing
    cumsum_X = torch.cat([
        torch.zeros(B, 1, D, device=X.device, dtype=X.dtype),
        torch.cumsum(X, dim=1)
    ], dim=1)  # shape (B, T+1, D)
    cumsum_X2 = torch.cat([
        torch.zeros(B, 1, D, device=X.device, dtype=X.dtype),
        torch.cumsum(X2, dim=1)
    ], dim=1)  # shape (B, T+1, D)

    sum_X = cumsum_X[:, window:] - cumsum_X[:, :-window]      # shape (B, T - window + 1, D)
    sum_X2 = cumsum_X2[:, window:] - cumsum_X2[:, :-window]   # shape (B, T - window + 1, D)

    mean = sum_X / window
    mean2 = sum_X2 / window

    var = mean2 - mean ** 2  # shape (B, T - window + 1, D)
    var = var.norm(dim=-1).mean(dim=0)
    return var  # shape (T - window + 1)




def get_dyck_mask(config, device):
    padded = False
    if "pad" in config.task:
        padded = config.task.pad
    seq_len = (config.seq_len+1)//2 if padded else config.seq_len
    mask = (torch.rand(seq_len, device=device) < config.task.repeat_prob).to(torch.uint8)
    # Ensure no consecutive 1's
    mask[1:] = mask[1:] & (1 - mask[:-1])
    mask[0] = 0
    cumsum_rows = torch.cumsum(mask, dim=0)
    cutoff_mask = cumsum_rows > (config.task.dyck_length * 2)
    mask = mask.masked_fill(cutoff_mask, 0)
    return mask

def symmetric_kl(p, q, eps=1e-12):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    kl_pq = torch.sum(p * (p.log() - q.log()), dim=-1)
    kl_qp = torch.sum(q * (q.log() - p.log()), dim=-1)

    return (kl_pq + kl_qp) / 2



def batch_generate_tokens(
    model, config, sampler, num_samples, max_new_tokens=20, temperature=1.0
):
    """
    Args:
        model: Transformer with forward returning logits
        prompt_ids: Tensor (B, T)
        max_new_tokens: int, number of tokens to generate
        temperature: float, sampling temperature
    Returns:
        Tensor (B, T)
    """
    generated, _ = sampler.generate(mode="eval", num_samples=num_samples)
    for t in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(generated)  # (B, T, V)
        next_token_logits = logits[:, t, :]  # (B, V)
        probs = torch.softmax(next_token_logits / temperature, dim=-1)  # (B, V)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)
        generated[:, t+1] = next_tokens
    return generated[:, :(max_new_tokens+1)]


def extract_subsequences(tensor: torch.Tensor, allowed, pos):
    """
    Args:
        tensor: torch.Tensor of shape (B, T)
        allowed: list, set, or torch.Tensor of allowed elements
    Returns:
        List[torch.Tensor] of shape (L_i,) for each subsequence
    """
    allowed_tensor = torch.tensor(list(allowed), device=tensor.device)
    mask = (tensor.unsqueeze(-1) == allowed_tensor).any(-1)  # (B, T)
    cumsum = mask.cumsum(dim=-1)
    mask = mask & (cumsum <= pos)
    subsequences = [row[mask_row] for row, mask_row in zip(tensor, mask)]
    return subsequences

def match_subsequences_to_task_pool_with_counts(task_pool, subsequences):
    """
    Args:
        task_pool: (N_tasks, T), torch.Tensor
        subsequences: List[torch.Tensor] of shape (L_i,)
    Returns:
        List[Tuple[int, int]]: (matched_idx, match_count) for each subsequence
    """
    matched_ids = []
    max_counts = []
    for subseq in subsequences:
        L = subseq.size(0)
        matches = (task_pool[:, :L] == subseq.unsqueeze(0))  # (N_tasks, L)
        match_counts = matches.sum(dim=1)  # (N_tasks,)
        # Find if any task has exact match
        is_exact_match = (match_counts == L)
        if is_exact_match.any():
            matched_idx = is_exact_match.nonzero(as_tuple=False).squeeze(1)[0].item()
        else:
            matched_idx = -1
        # max_count = match_counts.max().item()  # maximum tokens matched across all tasks
        matched_ids.append(matched_idx)
        max_counts.append(match_counts.cpu())
    return matched_ids, max_counts
