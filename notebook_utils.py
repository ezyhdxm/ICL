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

def get_config(total_trans=None, path=None):
    DEFAULT_PATH = os.path.join("results", "latent")
    if path is None:
        path = DEFAULT_PATH
    if total_trans is None:
        total_trans_list = []
        
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                total_trans_list.append(config.task.total_trans)
        pprint(f"You can choose from the following number of total transition matrices: \n {sorted(total_trans_list)}")
    else:
        folder_list = []
        for folder in os.listdir(path):
            if os.path.isdir(os.path.join(path, folder)):
                config = load_config(os.path.join(path, folder, "config.json"))
                if config.task.total_trans == total_trans:
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



##########################
# Phase Transition Plots #
##########################

def get_loss_lineplot(task_name, task_ids=None):
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    # Define normalization and colormap
    if task_ids is None:
        norm = mcolors.Normalize(vmin=0, vmax=13)
    else:
        norm = mcolors.Normalize(vmin=min(task_ids), vmax=max(task_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids):
            continue
        
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
    ax2.set_title("OOD Loss vs Step")
    ax2.set_xlabel("Step")

    ax1.set_xscale("log")
    ax1.set_title("ID Loss vs Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    plt.subplots_adjust(wspace=0.02)  # Try 0 for zero gap
    # Add colorbar to the whole figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label("Total Transitions")

    if task_ids is None:
        plt.savefig(os.path.join(folder_path, "loss_lineplots.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"loss_lineplots_{hash_array(task_ids)}.png"))
    plt.show()


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



def get_attn_score_lineplot(task_name, task_ids=None):
    folder_path = os.path.join("results", task_name)
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    # Create a 1x2 subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # adjust figsize as needed

    # Define normalization and colormap
    if task_ids is None:
        norm = mcolors.Normalize(vmin=0, vmax=13)
    else:
        norm = mcolors.Normalize(vmin=min(task_ids), vmax=max(task_ids))
    cmap = plt.get_cmap('plasma')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in range(len(folders)):
        result_path = os.path.join(folder_path, folders[i])
        log_path = os.path.join(result_path, "log.json")
        config_path = os.path.join(result_path, "config.json")
        log_data = load_log(log_path)
        config = load_config(config_path)

        if (task_ids is not None) and (config.task.total_trans not in task_ids):
            continue

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
        plt.savefig(os.path.join(folder_path, "attn_scores_lineplots.png"))
    else:
        plt.savefig(os.path.join(folder_path, f"attn_scores_lineplots_{hash_array(task_ids)}.png"))
    plt.show()


def kl_plot(model, sampler, task=None, num_samples=100):
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
    
    # Create a 1D NumPy array
    arr = np.arange(kl_losses.shape[1])
    
    mean_losses = np.mean(kl_losses, axis=0)
    std_losses = np.std(kl_losses, axis=0)
    
    # Plot the array
    plt.plot(arr, mean_losses)
    plt.fill_between(arr, np.maximum(mean_losses - std_losses, 0), mean_losses + 2*std_losses, color='blue', alpha=0.3, label="Mean ± 3 std")
    
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



