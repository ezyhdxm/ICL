import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

def get_loss_plots(config, train_losses, eval_losses, eval_steps, ngramLosses, task_name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs + 1), train_losses, 
            linestyle='-', color='lightblue', label='Training Loss')
    plt.plot(eval_steps, eval_losses, 
             linestyle='--', color='palevioletred', label='Validation Loss')

    if len(ngramLosses) >= 1:
        plt.plot(range(1, config.num_epochs + 1), ngramLosses[0], 
                linestyle='-', color='bisque', label='Unigram Loss')
    if len(ngramLosses) >= 2:
        plt.plot(range(1, config.num_epochs + 1), ngramLosses[1], 
                linestyle='-', color='khaki', label='Bigram Loss')
    if len(ngramLosses) >= 3:
        plt.plot(range(1, config.num_epochs + 1), ngramLosses[2], 
                linestyle='-', color='burlywood', label='Trigram Loss')
    for i in range(3, len(ngramLosses)):
        plt.plot(range(1, config.num_epochs + 1), ngramLosses[i], 
                linestyle='-', label=f'{i+1}-gram Loss')

    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    mlp = "no" if config.mlp == False else "with"
    linear = "(linear)" if config.activation == False else "" 
    plt.title(f'{task_name}: {config.num_heads} Heads {config.num_layers} Layers {mlp} MLP {linear} Loss Over Epochs ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()

def plot_probes(probes, config):
    plt.figure(figsize=(8, 6))
    for pkey in probes.keys():
        plt.plot(range(1, config.num_epochs + 1), probes[pkey], 
                linestyle='-', label=f'{pkey}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Mempry Recall')
    mlp = "no" if config.mlp == False else "with"
    linear = "(linear)" if config.activation == False else "" 
    plt.title(f'{config.num_heads} Heads {config.num_layers} Layers {mlp} MLP {linear} Recall Over Epochs ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()

def get_attn_gif(layer, head, attn_maps, config, folder="attns"):
    image_paths = []
    os.makedirs(folder)
    for i, attn in attn_maps.items():
        # Create heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(attn[layer][head], cmap="viridis", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Epoch {i + 1}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
    
        # Save image
        image_path = f"{folder}/attn_l{config.num_layers}h{config.num_heads}v{config.vocab_size}ep{i}_L{layer}H{head}.png"
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)
    
    # Step 2: Combine images into a GIF
    frames = [Image.open(image_path) for image_path in image_paths]
    output_gif_path = f"attnmaps_l{config.num_layers}h{config.num_heads}v{config.vocab_size}_L{layer}H{head}.gif"
    
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )
    
    print(f"GIF saved at {output_gif_path}")

    try:
        shutil.rmtree(folder)
        print(f"Folder '{folder}' and its contents removed.")
    except FileNotFoundError:
        print(f"Folder '{folder}' does not exist.")
    except OSError as e:
        print(f"Error removing folder '{folder}': {e}")