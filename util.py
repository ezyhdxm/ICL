import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import print_once

#################
# Bietti Probes #
#################

# See the construction on page 6 of the paper: https://arxiv.org/pdf/2306.00802
# This probe is to measure to what extend linear weights serve as associate memories.
def memory_recall_probe(num_tokens, model, to_probe, seq_len=None, device='cpu'):
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
        range_pos_toks = torch.arange(seq_len).to(device)
        pe = model.positional_encoding(range_pos_toks) # (T, D)
        k = model.layers[0].MHA.key(pe[:-1,:]) # (T-1, D)
        q = model.layers[0].MHA.query(pe[1:,:]) # (T-1, D)
        return ((q@k.t()).argmax(-1)==range_pos_toks[:seq_len-1]).float().mean().item()
