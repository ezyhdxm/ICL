import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

#####################
# Load Sampler
#####################

def load_sampler(sampler_path):
    sampler = pickle.load(open(sampler_path, "rb"))
    return sampler





######################
# Task Vector
######################

def get_single_task_vector(config, sampler, model, task_id, ffn=True):
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


def get_task_vectors(config, sampler, model, ffn=True):
    task_vecs  = torch.zeros((config.task.total_trans, config.model.emb_dim), device=config.device)
    for task_id in range(config.task.total_trans):
        tv = get_single_task_vector(config, sampler, model, task_id, ffn)[0]
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