import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import shutil
from PIL import Image
import torch
import numpy as np
import seaborn as sns
import stat
from datetime import datetime
from tasks.causal_graph import dag_to_adj
from tqdm.notebook import tqdm
from scipy.interpolate import make_interp_spline
from IPython.display import display, HTML

def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def get_loss_plots(config, train_results, folder="loss_plots", show=False, log=True, verbose=False):
    os.makedirs(folder, exist_ok=True)
    task_name = config.task_name
    train_losses, eval_losses, eval_steps = train_results["train_losses"], train_results["eval_losses"], train_results["eval_steps"]
    ngramLosses = train_results["ngramLosses"] if "ngramLosses" in train_results else []
    many_ngram_losses = train_results["many_ngram_losses"] if "many_ngram_losses" in train_results else []
    bayes_losses = train_results["bayes_losses"] if "bayes_losses" in train_results else []
    last_token_losses = train_results["last_token_losses"] if "last_token_losses" in train_results else []
    
    plt.figure(figsize=(8, 6))
    range_vec = range(1, config.num_epochs + 1)
    plt.plot(range_vec, train_losses, 
            linestyle='-', color='lightblue', label='Training Loss')
    if len(last_token_losses) >= 1:
        # spline = make_interp_spline(range_vec, last_token_losses, k=3)
        last_token_losses_smoothed = moving_average(last_token_losses)
        
        plt.plot(range_vec[:len(last_token_losses_smoothed)], last_token_losses_smoothed, 
                linestyle='-', color="#B39EB5", label='Last Token Training Loss', alpha=0.5)
    plt.plot(eval_steps, eval_losses, 
             linestyle='--', color='palevioletred', label='Validation Loss')

    if len(bayes_losses) >= 1:
        plt.plot(range_vec, bayes_losses, 
                linestyle='-', color='burlywood', label='Bayes Loss')
        
    cmap = cm.get_cmap('tab10') 
    for i in range(len(ngramLosses)):
        color = cmap(i)
        plt.axhline(y=ngramLosses[i], linestyle='-', label=f'{i+1}-gram Loss', color=color)
    
    if len(many_ngram_losses) >= 1:
        for i in range(len(many_ngram_losses)):
            color = cmap(i)
            plt.axhline(y=many_ngram_losses[i], linestyle='-', label=f'{i+1}-gram Loss', color=color, alpha=0.5)
    
    if log:
        plt.xscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    is_mlp = any(config.mlp)
    mlp = "no" if not is_mlp else "with"
    linear = "(linear)" if (is_mlp and not any(config.activation)) else "" 
    plt.title(f'{task_name}: {", ".join(map(str, config.num_heads))}-Heads {config.num_layers} Layers {mlp} MLP {linear} Loss ({config.pos_enc})')
    plt.legend()
    plt.grid()
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    image_path = f"{folder}/s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}{task_name}_{curr_time}.png"
    plt.savefig(image_path)
    if verbose:
        print("Loss plot saved at ", image_path)
    if show:
        plt.show()
    plt.close()


def plot_probes(train_results, config, folder="loss_plots", show=False, log=True):
    probes = train_results["probes"]
    if len(probes) == 0:
        return
    
    task_name = config.task_name
    plt.figure(figsize=(8, 6))
    for pkey in probes.keys():
        plt.plot(range(1, config.num_epochs + 1), probes[pkey], 
                linestyle='-', label=f'{pkey}')
    
    if log:
        plt.xscale('log')
    
    plt.xlabel('Epochs')
    if task_name == "bietti":
        plt.ylabel('Mempry Recall & KL divergence')
    else:
        plt.ylabel('KL divergence')
    is_mlp = any(config.mlp)
    mlp = "no" if not is_mlp else "with"
    linear = "(linear)" if is_mlp and not any(config.activation) else "" 
    plt.title(f'{",".join(map(str, config.num_heads))} Heads {config.num_layers} Layers {mlp} MLP {linear} Recall ({config.pos_enc})')
    plt.legend()
    plt.grid()
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    image_path = f"{folder}/probe_s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}{task_name}_{curr_time}.png"
    plt.savefig(image_path)
    if show:
        plt.show()
    plt.close()


def plot_bigram_icl_risk(config, train_results, folder="loss_plots", show=False, log=True):

    if len(train_results["bigram_losses"]) == 0:
        return 

    task_name = config.task_name
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs + 1), train_results["bigram_losses"], 
            linestyle='-', label='Bigram Risk')
    plt.plot(range(1, config.num_epochs + 1), train_results["icl_losses"], 
             linestyle='--', label='ICL Risk')
    if log:
        plt.xscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    mlp = "no" if config.mlp == False else "with"
    linear = "(linear)" if config.activation == False else "" 
    plt.title(f'{task_name}: {config.num_heads} Heads {config.num_layers} Layers {mlp} MLP {linear} Loss Over Epochs ({config.pos_enc})')
    plt.legend()
    plt.grid()
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_scale = "log" if log else ""
    image_path = f"{folder}/icl_s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}_{log_scale}_{task_name}_{curr_time}.png"
    plt.savefig(image_path)
    if show:
        plt.show()
    plt.close()



def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def remove_readonly(func, path, exc_info):
    """Handle read-only files while deleting"""
    os.chmod(path, stat.S_IWRITE)  # Change to writable
    func(path)  # Retry removal


def get_attn_gif(layer, head, train_results, config, dag=None, folder="attns", out_folder="attns_plot", show=False, log=True, verbose=False):
    task_name = config.task_name
    attn_maps = train_results["attn_maps"]
    image_paths = []
    if os.path.exists(folder):
        shutil.rmtree(folder, onerror=remove_readonly)  # Handle read-only files
        print(f"Deleted: {folder}")
    
    os.makedirs(folder)
    steps = 0
    n_layer, n_heads, n_voc = config.num_layers, config.num_heads[layer], config.vocab_size
    
    for i, attn in tqdm(attn_maps.items(), mininterval=1, desc="Creating images"):
        if i < steps:
            continue
        
        if steps < 3000:
            steps += config.get_attn
        elif steps < 6000:
            steps += max(500, config.get_attn)
        else:
            steps += max(1000, config.get_attn)

        if dag is None:
            if head != "all" or config.num_heads[layer]==1:
                head = 0
                plt.figure(figsize=(6, 6))
                sns.heatmap(attn[layer][head].cpu(), cmap="viridis", annot=False, cbar=False)
                plt.title(f"Layer {layer}, Head {head}, Epoch {i + 1}")
            else:
                h = config.num_heads[layer]
                fig, axes = plt.subplots(1, h, figsize=(6*h, 6))
                for j in range(h):
                    sns.heatmap(attn[layer][j].cpu(), ax=axes[j], cmap="viridis", annot=False, cbar=False)
                    axes[j].set_title(f"Head {j+1} at Epoch {i + 1}")
                plt.tight_layout()
        else:
            adj_mat = dag_to_adj(dag, task_name)
            if head != "both":
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

                sns.heatmap(attn[layer][head].cpu(), ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

                sns.heatmap(attn[layer][0].cpu(), ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Head 1 Attention Map at Epoch {i + 1}")
                sns.heatmap(attn[layer][1].cpu(), ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title(f"Head 2 Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[2], cmap="viridis", annot=False, linewidths=0.05)
                axes[2].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()


        # Save image
        image_path = f"{folder}/attn_l{n_layer}h{n_heads}v{n_voc}ep{i}_L{layer}H{head}{task_name}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)

    # Step 2: Combine images into a GIF
    frames = [Image.open(image_path) for image_path in image_paths]
    os.makedirs(out_folder, exist_ok=True)
    # Get current time
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_gif_path = f"{out_folder}/s{config.seq_len}p_{config.pos_enc}_l{n_layer}h{n_heads}v{n_voc}_L{layer}H{head}{task_name}_{curr_time}.gif"
    
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )
    
    if verbose:
        print(f"GIF saved at {output_gif_path}")
    shutil.rmtree(folder, onerror=remove_readonly)
    if verbose:
        print(f"Folder '{folder}' and its contents removed.")
    if show:
        display(HTML(f'<img src="{output_gif_path}" width="500px">'))
    return output_gif_path

def get_pos_sim(seq_len, model, device, pos_enc):
    
    if pos_enc == "abs":
        range_pos_toks = torch.arange(seq_len).to(device)
        pos_emb = model.positional_encoding(range_pos_toks)
    elif pos_enc == "rpe":
        pos_emb = model.layers[0].MHA.PEK.pe[:(seq_len+1)]
    

    similar = pos_emb @ pos_emb.t()
    similar = similar.detach().cpu()
    plt.imshow(np.abs(similar))
    plt.show()

def get_emb_sim(vocab_size, model, device):
    range_toks = torch.arange(vocab_size).to(device)
    toks = model.embed(range_toks)
    similar = toks @ toks.t()
    similar = similar.detach().cpu()
    plt.imshow(np.abs(similar))
    plt.show()

def plot_adj_heatmap(adj_mat):
    plt.figure(figsize=(6, 6))
    sns.heatmap(adj_mat, annot=False, cmap='viridis', fmt='.2f', linewidths=0.05, cbar=False)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.title('Matrix Heatmap')
    plt.show()