import torch

# Adapted from https://github.com/jessevig/bertviz/tree/master
def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    max_heads = 0
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        max_heads = max(max_heads, layer_attention.shape[0])
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    for i in range(len(squeezed)):
        if squeezed[i].shape[0] < max_heads:
            padding = torch.zeros((max_heads - squeezed[i].shape[0], squeezed[i].shape[1], squeezed[i].shape[2]), device=squeezed[i].device)
            squeezed[i] = torch.cat((squeezed[i], padding))
            print(squeezed[i].shape)
    return torch.stack(squeezed)


def num_heads(attention):
    return attention[0][0].size(0)

