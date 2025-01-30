import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import torch
import numpy as np
import seaborn as sns
import stat
from datetime import datetime
from causal_graph import dag_to_adj

def get_loss_plots(config, train_results, task_name):
    train_losses, eval_losses, eval_steps = train_results["train_losses"], train_results["eval_losses"], train_results["eval_steps"]
    ngramLosses = train_results["ngramLosses"] if "ngramLosses" in train_results else []
    bayes_losses = train_results["bayes_losses"] if "bayes_losses" in train_results else []
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs + 1), train_losses, 
            linestyle='-', color='lightblue', label='Training Loss')
    plt.plot(eval_steps, eval_losses, 
             linestyle='--', color='palevioletred', label='Validation Loss')

    if len(bayes_losses) >= 1:
        plt.plot(range(1, config.num_epochs + 1), bayes_losses, 
                linestyle='-', color='burlywood', label='Bayes Loss')

    for i in range(len(ngramLosses)):
        plt.plot(range(1, config.num_epochs + 1), ngramLosses[i], 
                linestyle='-', label=f'{i+1}-gram Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    is_mlp = any(config.mlp)
    mlp = "no" if not is_mlp else "with"
    linear = "(linear)" if (is_mlp and not any(config.activation)) else "" 
    plt.title(f'{task_name}: {config.num_heads}-Heads {config.num_layers} Layers {mlp} MLP {linear} Loss ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()


def plot_probes(train_results, config):
    probes = train_results["probes"]
    plt.figure(figsize=(8, 6))
    for pkey in probes.keys():
        plt.plot(range(1, config.num_epochs + 1), probes[pkey], 
                linestyle='-', label=f'{pkey}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Mempry Recall & KL divergence')
    is_mlp = any(config.mlp)
    mlp = "no" if not is_mlp else "with"
    linear = "(linear)" if is_mlp and not any(config.activation) else "" 
    plt.title(f'{config.num_heads} Heads {config.num_layers} Layers {mlp} MLP {linear} Recall ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()


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
        os.chmod(path, sstat.S_IWRITE)
        func(path)
    else:
        raise

def get_attn_gif(layer, head, train_results, config, task_name, dag=None, folder="attns", out_folder="attns_plot"):
    attn_maps = train_results["attn_maps"]
    image_paths = []
    os.makedirs(folder)
    for i, attn in attn_maps.items():
        if dag is None:
            plt.figure(figsize=(6, 6))
            sns.heatmap(attn[layer][head], cmap="viridis", annot=False)
            plt.colorbar()
            plt.title(f"Epoch {i + 1}")
        else:
            adj_mat = dag_to_adj(dag, task_name)
            if head != "both":
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

                sns.heatmap(attn[layer][head], ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()
            else:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

                sns.heatmap(attn[layer][0], ax=axes[0], cmap="viridis", annot=False, linewidths=0.05)
                axes[0].set_title(f"Head 1 Attention Map at Epoch {i + 1}")
                sns.heatmap(attn[layer][1], ax=axes[1], cmap="viridis", annot=False, linewidths=0.05)
                axes[1].set_title(f"Head 2 Attention Map at Epoch {i + 1}")
                sns.heatmap(adj_mat, ax=axes[2], cmap="viridis", annot=False, linewidths=0.05)
                axes[2].set_title("Adjacency Matrix of DAG")
                plt.tight_layout()


        # Save image
        image_path = f"{folder}/attn_l{config.num_layers}h{config.num_heads[layer]}v{config.vocab_size}ep{i}_L{layer}H{head}{task_name}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)

    # Step 2: Combine images into a GIF
    frames = [Image.open(image_path) for image_path in image_paths]
    os.makedirs(out_folder, exist_ok=True)
    # Get current time
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_gif_path = f"{out_folder}/l{config.num_layers}h{config.num_heads[layer]}v{config.vocab_size}_L{layer}H{head}{task_name}_{curr_time}.gif"
    
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )
    
    print(f"GIF saved at {output_gif_path}")
    shutil.rmtree(folder, onerror=onerror)
    print(f"Folder '{folder}' and its contents removed.")

def get_pos_sim(config, model):
    range_pos_toks = torch.arange(config.seq_len).to(config.device)
    pos_emb = model.positional_encoding(range_pos_toks)
    similar = pos_emb @ pos_emb.t()
    similar = similar.detach().cpu()
    plt.imshow(np.abs(similar))
    plt.show()

def plot_bigram_icl_risk(config, train_results, task_name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs + 1), train_results["bigram_losses"], 
            linestyle='-', label='Bigram Risk')
    plt.plot(range(1, config.num_epochs + 1), train_results["icl_losses"], 
             linestyle='--', label='ICL Risk')
    plt.xscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    mlp = "no" if config.mlp == False else "with"
    linear = "(linear)" if config.activation == False else "" 
    plt.title(f'{task_name}: {config.num_heads} Heads {config.num_layers} Layers {mlp} MLP {linear} Loss Over Epochs ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()

def plot_adj_heatmap(adj_mat):
    plt.figure(figsize=(6, 6))
    sns.heatmap(adj_mat, annot=False, cmap='viridis', fmt='.2f', linewidths=0.05, cbar=False)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0) 
    plt.title('Matrix Heatmap')
    plt.show()