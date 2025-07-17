import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

# Option 1: Enhanced matplotlib/seaborn version
def create_enhanced_matplotlib_heatmap(attention_data, title="Attention Heatmap"):
    """Enhanced static heatmap with better styling and annotations"""
    
    # Create figure with subplots for main heatmap and colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), 
                                   gridspec_kw={'width_ratios': [4, 1]})
    
    # Main heatmap
    im = ax1.imshow(attention_data, cmap='viridis', aspect='auto')
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Key Position', fontsize=12)
    ax1.set_ylabel('Query Position', fontsize=12)
    
    # Add grid
    ax1.set_xticks(range(attention_data.shape[1]))
    ax1.set_yticks(range(attention_data.shape[0]))
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations for values
    for i in range(attention_data.shape[0]):
        for j in range(attention_data.shape[1]):
            text = ax1.text(j, i, f'{attention_data[i, j]:.3f}',
                           ha="center", va="center", color="white", fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, cax=ax2)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Statistics subplot
    ax2.text(0.5, -0.15, f'Max: {attention_data.max():.3f}\nMin: {attention_data.min():.3f}\nMean: {attention_data.mean():.3f}', 
             transform=ax2.transAxes, ha='center', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    return fig

# Option 2: Interactive Plotly version
def create_interactive_plotly_heatmap(attention_data, title="Interactive Attention Heatmap", size=800):
    """Fully interactive heatmap with hover information and controls"""
    
    # Create hover text with detailed information
    hover_text = []
    for i in range(attention_data.shape[0]):
        hover_row = []
        for j in range(attention_data.shape[1]):
            hover_row.append(f'Query: {i}<br>Key: {j}<br>Attention: {attention_data[i,j]:.4f}')
        hover_text.append(hover_row)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_data,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text,
        colorscale='Viridis',
        colorbar=dict(
            title="Attention Weight",
            titleside="right"
        )
    ))
    
    # Update layout to be square
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=size,
        height=size,  # Make it square
        font=dict(size=12),
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure 1:1 aspect ratio
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig

# Option 3: Multi-layer attention visualization
def create_multihead_attention_plot(attention_tensor, num_layers):
    """Visualize multiple attention layers in a grid"""
    
    rows = int(np.ceil(np.sqrt(num_layers)))
    cols = int(np.ceil(num_layers / rows))
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Layer {i}' for i in range(num_layers)],
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    for layer in range(num_layers):
        row = layer // cols + 1
        col = layer % cols + 1
        
        fig.add_trace(
            go.Heatmap(
                z=attention_tensor[layer].cpu().numpy(),
                showscale=False,
                colorscale='Viridis'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Multi-Layer Attention Pattern",
        height=200*rows,
        width=200*cols
    )
    
    return fig

# Option 4: Animated attention over layers
def create_animated_attention_heatmap(attention_layers):
    """Create an animated heatmap showing attention across different layers"""
    
    frames = []
    for i, layer_attention in enumerate(attention_layers):
        frame = go.Frame(
            data=[go.Heatmap(
                z=layer_attention.cpu().numpy(),
                colorscale='Viridis',
                zmin=0,
                zmax=1
            )],
            name=f'Layer {i+1}'
        )
        frames.append(frame)
    
    fig = go.Figure(
        data=[go.Heatmap(
            z=attention_layers[0].cpu().numpy(),
            colorscale='Viridis'
        )],
        frames=frames
    )
    
    fig.update_layout(
        title="Attention Across Layers",
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 1000, 'redraw': True},
                                  'fromcurrent': True}]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}}]
                }
            ]
        }]
    )
    
    return fig

# Example usage for your specific case:
def plot_attention_interactive(attns, layer_idx=1):
    """Main function to plot your attention data interactively"""
    
    # Extract attention data from dictionary
    if layer_idx not in attns:
        print(f"Layer {layer_idx} not found. Available layers: {list(attns.keys())}")
        layer_idx = list(attns.keys())[0]  # Use first available layer
    
    attention_data = attns[layer_idx].squeeze(0).cpu().numpy()
    
    print("Choose visualization option:")
    print("1. Enhanced matplotlib heatmap")
    print("2. Interactive Plotly heatmap")
    print("3. Multi-layer visualization")
    print("4. Layer comparison")
    print("5. Animated across layers")
    print("6. Show all options")
    
    choice = input("Enter choice (1-6): ")
    
    if choice == '1':
        fig = create_enhanced_matplotlib_heatmap(attention_data, f"Attention Layer {layer_idx}")
        plt.show()
    
    elif choice == '2':
        fig = create_interactive_plotly_heatmap(attention_data, f"Interactive Attention Layer {layer_idx}")
        fig.show()
    
    elif choice == '3':
        # Convert dict to tensor for multi-layer viz
        layer_tensor = torch.stack([attns[k].squeeze(0) for k in sorted(attns.keys())])
        fig = create_multihead_attention_plot(layer_tensor, len(attns))
        fig.show()
    
    elif choice == '4':
        fig = create_layer_comparison(attns)
        fig.show()
    
    elif choice == '5':
        attention_layers = [attns[k].squeeze(0) for k in sorted(attns.keys())]
        fig = create_animated_attention_heatmap(attention_layers)
        fig.show()
    
    elif choice == '6':
        # Show all visualizations
        print(f"\n1. Enhanced matplotlib version (Layer {layer_idx}):")
        fig1 = create_enhanced_matplotlib_heatmap(attention_data, f"Attention Layer {layer_idx}")
        plt.show()
        
        print(f"\n2. Interactive Plotly version (Layer {layer_idx}):")
        fig2 = create_interactive_plotly_heatmap(attention_data, f"Interactive Attention Layer {layer_idx}")
        fig2.show()
        
        print("\n3. Multi-layer visualization:")
        layer_tensor = torch.stack([attns[k].squeeze(0) for k in sorted(attns.keys())])
        fig3 = create_multihead_attention_plot(layer_tensor, len(attns))
        fig3.show()

# Quick replacement for your original code:
def quick_interactive_plot(attns, layer_idx=1, size=800):
    """Direct replacement for your sns.heatmap line"""
    
    # Handle dictionary structure
    if layer_idx not in attns:
        print(f"Layer {layer_idx} not found. Available layers: {list(attns.keys())}")
        layer_idx = list(attns.keys())[0]  # Use first available layer
    
    attention_data = attns[layer_idx].squeeze(0).cpu().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=attention_data,
        colorscale='Viridis',
        hovertemplate=f'Layer {layer_idx}<br>Query: %{{y}}<br>Key: %{{x}}<br>Attention: %{{z:.4f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Attention Heatmap - Layer {layer_idx}",
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=size,
        height=size,  # Make it square
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure 1:1 aspect ratio
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    fig.show()

# New function for layer comparison
def create_layer_comparison(attns):
    """Compare attention patterns across different layers"""
    
    num_layers = len(attns)
    rows = int(np.ceil(np.sqrt(num_layers)))
    cols = int(np.ceil(num_layers / rows))
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Layer {k}' for k in sorted(attns.keys())],
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    for idx, layer_key in enumerate(sorted(attns.keys())):
        row = idx // cols + 1
        col = idx % cols + 1
        
        attention_data = attns[layer_key].squeeze(0).cpu().numpy()
        
        fig.add_trace(
            go.Heatmap(
                z=attention_data,
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorscale='Viridis',
                hovertemplate=f'Layer {layer_key}<br>Query: %{{y}}<br>Key: %{{x}}<br>Attention: %{{z:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Attention Patterns Across Layers",
        height=200*rows,
        width=200*cols
    )
    
    return fig

# Replace your original line with:
# quick_interactive_plot(attns, layer_idx=1)

# Or use the full interactive version:
# plot_attention_interactive(attns, layer_idx=1)

# Additional utility functions:
def explore_attention_dict(attns):
    """Explore the structure of your attention dictionary"""
    print("Attention Dictionary Structure:")
    print(f"Number of layers: {len(attns)}")
    print(f"Available layer keys: {list(attns.keys())}")
    
    for key, tensor in attns.items():
        print(f"Layer {key}: shape {tensor.shape}, dtype {tensor.dtype}")
        print(f"  Min: {tensor.min():.4f}, Max: {tensor.max():.4f}, Mean: {tensor.mean():.4f}")
    
    return attns

def quick_layer_stats(attns):
    """Get quick statistics for all layers"""
    stats = {}
    for layer_key, tensor in attns.items():
        attention_data = tensor.squeeze(0).cpu().numpy()
        stats[layer_key] = {
            'shape': attention_data.shape,
            'min': attention_data.min(),
            'max': attention_data.max(),
            'mean': attention_data.mean(),
            'std': attention_data.std()
        }
    return stats