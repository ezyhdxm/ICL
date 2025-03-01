from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tasks.latent_utils import *

def ood_id_plot(alpha, path, id_results_icl, ood_results_icl, id_results, ood_results, check_results):
    folders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    with open(f'{folders[0]}/train_results.pkl', 'rb') as file:
        train_results_ckpt = pickle.load(file)

    alpha = 0.1
    data1 = id_results.T
    data1 = np.concatenate((data1, id_results_icl.T), axis=0)
    data2 = ood_results[alpha].T
    data2 = np.concatenate((data2, ood_results_icl[alpha].T), axis=0)
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    
    # Create side-by-side heatmaps
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # First heatmap
    sns.heatmap(data1, ax=ax[0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('ID Error')
    
    # Second heatmap
    sns.heatmap(data2, ax=ax[1], cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title(f'OOD Error, $\\alpha={alpha}$')
    
    ngrams = ngram_checker(train_results_ckpt, alpha=1, verbose=True)
    contour_levels1 = [ngrams["unigram_loss"]-0.005, ngrams["unigram_loss"]+0.005]
    
    # Overlay contour lines
    X, Y = np.meshgrid(np.arange(data1.shape[1]), np.arange(data1.shape[0]))  # Create grid
    ax[0].contour(X + 0.5, Y + 0.5, data1, levels=contour_levels1, colors="white", linewidths=1, alpha=0.7)
    
    ngrams = ngram_checker(train_results_ckpt, alpha=alpha, verbose=True)
    contour_levels1 = [ngrams["unigram_loss"]-0.005, ngrams["unigram_loss"]+0.005]
    
    # Overlay contour lines
    X, Y = np.meshgrid(np.arange(data2.shape[1]), np.arange(data2.shape[0]))  # Create grid
    ax[1].contour(X + 0.5, Y + 0.5, data2, levels=contour_levels1, colors="white", linewidths=1, alpha=0.7)
    
    keys = list(check_results.keys())
    
    x_labels = sorted(list(check_results[keys[0]].keys()))
    y_labels = sorted(keys)  # Custom labels for y-axis
    y_labels.append("$\\infty$")
    
    for i in range(2):
        ax[i].set_xticks(np.arange(len(x_labels))[::4] + 0.5)
        ax[i].set_xticklabels(x_labels[::4], rotation=45)
        ax[i].set_yticks(np.arange(len(y_labels))[1::3] + 0.5)
        ax[i].set_yticklabels(y_labels[1::3], rotation=30)
    
    plt.gca().invert_yaxis()
    
    # Adjust layout
    plt.tight_layout()
    
    legend_elements = [
        Line2D([0], [0], color="white", lw=1, label="Unigram Contour"),
    ]
    
    # Add legend to the plot
    ax[0].legend(handles=legend_elements, loc="upper right", frameon=True)
    ax[1].legend(handles=legend_elements, loc="upper right", frameon=True)

    formatted_alpha = str(alpha).replace(".", "_")

    plt.savefig(f"{path}ood_id_{formatted_alpha}_.png")
    # Show the plot
    plt.show()