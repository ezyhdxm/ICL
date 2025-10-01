import torch
# from torch.utils.data import Dataset
# from tasks.causal_graph import *
from icl.tasks.markov import *
# from models.ngram_latent import *
from icl.tasks.markov_latent import *
from torchinfo import summary


def get_bayes_loss(bayes_prob, prob):
    return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()


def last_token_loss(logits, probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()



def get_train_result(**kwargs):
    return kwargs


    

def tabulate_model(model: torch.nn.Module, seq_len: int, batch_size: int, device: str) -> str:
    dummy_data = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    try:
        info = summary(model, 
                       input_data=dummy_data, 
                       depth=3, 
                       col_names=["input_size", "output_size", "num_params"])
        return str(info)
    except Exception as e:
        return f"Could not tabulate model: {e}"


def compute_cross_entropy(preds, samples, eps=1e-12):
    log_preds = torch.log(preds + eps)  # (B, T, N)
    indices = samples.unsqueeze(-1)  # (B, T, 1)
    log_probs_true = torch.gather(log_preds, dim=2, index=indices).squeeze(-1)  # (B, T)
    cross_entropy = -log_probs_true  # (B, T)

    return cross_entropy


#######################
# Attention utilities #
#######################

def get_attn_base(model, batch):
    attn_maps = {}

    num_layers = len(model.layers)

    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            attn_maps[layer_idx] = module.attn.detach().cpu()
    
        return hook_fn

    handles = []

    for l in range(num_layers):
        model.layers[l].attn_block.MHA.flash = False
        handles.append(model.layers[l].attn_block.MHA.register_forward_hook(create_hook_fn(l)))
    
    with torch.no_grad():
        _ = model(batch)
    
    for l in range(num_layers):
        handles[l].remove()
        model.layers[l].attn_block.MHA.flash = True
    
    return attn_maps

def get_attn_at_layer_base(model, batch, layer):
    model.layers[layer].attn_block.MHA.flash = False
    attn_maps = {}

    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            attn_maps[layer_idx] = module.attn.detach().cpu()
    
        return hook_fn

    handle = model.layers[layer].attn_block.MHA.register_forward_hook(create_hook_fn(layer))
    _ = model(batch)
    handle.remove()
    model.layers[layer].attn_block.MHA.flash = True
    return attn_maps[layer]

def pth_score(model, batch, layer=0):
    attn = get_attn_at_layer_base(model, batch, layer)
    attn = attn.squeeze(1)
    return attn.mean(dim=0).diagonal(offset=-1).mean().item()

def ih_score(model, batch, device, layer=1):
    attns = get_attn_at_layer_base(model, batch, layer)
    attns = attns.squeeze(1) # (B, H, T, T)
    B, T = batch.shape
    
    # Compare all pairs: (B, T, T), batch[b, i] == batch[b, t]
    matches = (batch.unsqueeze(2) == batch.unsqueeze(1))  # (B, T, T)
    
    # Mask out positions where i >= t (i.e., keep only i < t)
    tril_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)  # (T, T)
    valid_matches = matches & tril_mask  # broadcasted to (B, T, T)
    b_indices, t_indices, i_indices = valid_matches.nonzero(as_tuple=True)
    grouped_sums = torch.zeros((B, T), device=device)  # (B, T)

    # Accumulate values at corresponding (b, t) positions
    grouped_sums.index_put_((b_indices, t_indices), attns[b_indices, t_indices, i_indices+1], accumulate=True)
    return grouped_sums.mean(dim=0)[1:].mean().item()




    


























