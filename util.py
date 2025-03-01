import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_models import *

#################
# Bietti Probes #
#################

# See the construction on page 6 of the paper: https://arxiv.org/pdf/2306.00802
# This probe is to measure to what extend linear weights serve as associate memories.
def memory_recall_probe(num_tokens, model, to_probe, pos_enc, seq_len=None, device='cpu'):
    range_toks = torch.arange(num_tokens).to(device)
    if to_probe == "wk1": 
        toks_key = model.embed(range_toks)
        toks_key = model.layers[0].MHA.value(toks_key)
        toks_key = model.layers[0].MHA.out(toks_key)
        toks_key = model.layers[1].MHA.key(toks_key)

        toks_query = model.embed(range_toks)
        toks_query = model.layers[1].MHA.query(toks_query)
        return ((toks_query @ toks_key.t()).argmax(-1) == range_toks).float().mean().item()
    
    elif to_probe == "wo1":
        toks = model.embed(range_toks)
        toks = model.layers[1].MHA.value(toks) 
        toks = model.layers[1].MHA.out(toks) # if associated memory, then toks is close to the unembedded matrix
        toks = model.output_layer(toks)
        return (toks.argmax(-1) == range_toks).float().mean().item()
    
    elif to_probe == "wk0":
        if pos_enc != "abs":
            return 
        
        range_pos_toks = torch.arange(seq_len).to(device)
        if pos_enc == "abs":
            pe = model.positional_encoding(range_pos_toks) # (T, D)
            k = model.layers[0].MHA.key(pe[:-1,:]) # (T-1, D)
            q = model.layers[0].MHA.query(pe[1:,:]) # (T-1, D)
        return ((q@k.t()).argmax(-1)==range_pos_toks[:seq_len-1]).float().mean().item()

def output_probe(num_tokens, model, trans_mat, device='cpu', random_tokens=None):
    range_toks = torch.arange(num_tokens).to(device)
    if random_tokens is not None:
        mask = ~torch.isin(range_toks, random_tokens)
        range_toks = range_toks[mask]
    toks = model.embed(range_toks)
    toks = model.output_layer(toks)
    return F.kl_div(trans_mat[range_toks].log(), nn.Softmax(dim=1)(toks), reduction='batchmean').item()

def output_residual_probe(num_tokens, model, trans_mat, device='cpu', random_tokens=None):
    range_toks = torch.arange(num_tokens).to(device)
    if random_tokens is not None:
        mask = ~torch.isin(range_toks, random_tokens)
        range_toks = range_toks[mask]
    toks = model.embed(range_toks)
    res = model.layers[0].MHA.value(toks)
    res = model.layers[0].MHA.out(res)
    res = model.layers[1].MHA.value(res)
    res = model.layers[1].MHA.out(res)
    toks += res
    toks = model.output_layer(toks)
    return F.kl_div(trans_mat[range_toks].log(), nn.Softmax(dim=1)(toks), reduction='batchmean').item()

def feedforward_probe(num_tokens, model, trans_mat, device='cpu', random_tokens=None, layer=1):
    range_toks = torch.arange(num_tokens).to(device)
    if random_tokens is not None:
        mask = ~torch.isin(range_toks, random_tokens)
        range_toks = range_toks[mask]
    toks = model.embed(range_toks)
    mlp_out = model.layers[layer].mlp(toks)
    toks += mlp_out
    toks = model.output_layer(toks)
    return F.kl_div(trans_mat[range_toks].log(), nn.Softmax(dim=1)(toks), reduction='batchmean').item()


def feedforward_residual_probe(num_tokens, model, trans_mat, device='cpu', random_tokens=None):
    range_toks = torch.arange(num_tokens).to(device)
    if random_tokens is not None:
        mask = ~torch.isin(range_toks, random_tokens)
        range_toks = range_toks[mask]
    toks = model.embed(range_toks)
    res = model.layers[0].MHA.value(toks)
    res = model.layers[0].MHA.out(res)
    res = model.layers[1].MHA.value(res)
    res = model.layers[1].MHA.out(res)
    mlp_out = model.layers[1].mlp(toks+res)
    toks += mlp_out
    toks = model.output_layer(toks)
    return F.kl_div(trans_mat[range_toks].log(), nn.Softmax(dim=1)(toks), reduction='batchmean').item()
    
    
def activation_probe(num_tokens, model, device='cpu'):
    range_toks = torch.arange(num_tokens).to(device)
    toks = model.embed(range_toks)
    toks = model.layers[0].mlp[0](toks)
    acts = model.layers[0].mlp[1](toks)
    return toks, acts

def TV_dist(targets, probs):
    return torch.sum(torch.abs(targets - probs), axis=-1).mean().item()

def ff_icl_probe(num_tokens, model, device='cpu'):
    OV_2 = model.layers[1].MHA.value.weight.T @ model.layers[1].MHA.out.weight.T
    range_toks = torch.arange(num_tokens).to(device)
    tok = model.embed(range_toks)
    targets = torch.eye(num_tokens).to(device)
    probs = nn.Softmax(dim=-1)(model.output_layer(model.layers[1].mlp(tok @ OV_2)))
    return TV_dist(targets, probs)

def ff_memory_probe(num_tokens, model, trans_mat, device='cpu', weight="true"):
    # OV_1 = model.layers[0].MHA.value.weight.T @ model.layers[0].MHA.out.weight.T
    OV_2 = model.layers[1].MHA.value.weight.T @ model.layers[1].MHA.out.weight.T
    range_toks = torch.arange(num_tokens).to(device)
    toks = model.embed(range_toks) # (N, D)
    if weight == "uniform":
        avg_toks = toks.mean(0).unsqueeze(0) # uniform weights
    else:
        avg_toks = trans_mat[range_toks] @ toks # weighted by trans_mat, (N, N) @ (N, D) -> (N, D)
    mlp_out = model.layers[1].mlp(toks + avg_toks @ OV_2)
    probs = nn.Softmax(dim=-1)(model.output_layer(toks + avg_toks @ OV_2 + mlp_out))
    return F.kl_div(trans_mat[range_toks].log(), probs, reduction='batchmean').item()

def attn_icl_probe(num_tokens, model, device='cpu'):
    OV_2 = model.layers[1].MHA.value.weight.T @ model.layers[1].MHA.out.weight.T
    range_toks = torch.arange(num_tokens).to(device)
    tok = model.embed(range_toks)
    probs = nn.Softmax(dim=-1)(model.output_layer(tok @ OV_2))
    targets = torch.eye(num_tokens).to(device)
    return TV_dist(targets, probs)

def combined_icl_probe(num_tokens, model, device='cpu'):
    OV_2 = model.layers[1].MHA.value.weight.T @ model.layers[1].MHA.out.weight.T
    range_toks = torch.arange(num_tokens).to(device)
    tok = model.embed(range_toks)
    probs = nn.Softmax(dim=-1)(model.output_layer(tok @ OV_2 + model.layers[1].mlp(tok @ OV_2)))
    targets = torch.eye(num_tokens).to(device)
    return TV_dist(targets, probs)

#################
# Get Attention #
#################


def get_attn(batch, layer, model, seq_len, device="cpu"):
    toks = model.embed(batch)
    freqs_cis = precompute_freqs_cis(16, seq_len * 2).to(device)
    Q = model.layers[layer].MHA.query(toks).view(1, seq_len, 1, 16)
    K = model.layers[layer].MHA.key(toks).view(1, seq_len, 1, 16)
    Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:seq_len])
    Q, K = Q.transpose(1, 2), K.transpose(1, 2)
    QK = Q @ K.transpose(-1,-2) / model.layers[layer].MHA.scale
    QK = QK.masked_fill(model.layers[layer].MHA.mask==0, -float("inf"))
    A = F.softmax(QK, dim=-1)
    return A