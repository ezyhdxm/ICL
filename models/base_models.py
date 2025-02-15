from dataclasses import dataclass
import logging
import random
import math
import numpy as np
from collections import namedtuple
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
from models.pos_encoder import *
from models.attention import *


class TFBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.MHA = MultiHeadAttention(config, layer)
        self.ln1 = nn.LayerNorm(config.emb_dim) if config.layer_norm else nn.Identity()
        self.mlp = None
        self.dropout = nn.Dropout(config.dropout) if config.dropout else None

        if config.mlp[layer]:
            if config.activation[layer]:
                assert config.ff_dim is not None, "FeedForward dimension cannot be empty."
                self.mlp = nn.Sequential(
                    nn.Linear(config.emb_dim, config.ff_dim),
                    nn.ReLU(),
                    nn.Linear(config.ff_dim, config.emb_dim)
                )
            else:
                self.mlp = nn.Linear(config.emb_dim, config.emb_dim)
            self.ln2 = nn.LayerNorm(config.emb_dim) if config.layer_norm else nn.Identity()

    def forward(self, x, get_attn=False):
        x = self.ln1(x)
        attn_map = -1
        atten_out, attn_map = self.MHA(x, get_attn)
        x = x + self.dropout(atten_out) if self.dropout is not None else x + atten_out
        if self.mlp is not None:
            mlp_out = self.mlp(self.ln2(x))
            x = x + self.dropout(mlp_out) if self.dropout is not None else x + mlp_out
        return x, attn_map 
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim).to(config.device)
        self.pos_enc = config.pos_enc
        if config.pos_enc == "abs":
            self.positional_encoding = nn.Embedding(config.seq_len, config.emb_dim)
        self.layers = nn.ModuleList([TFBlock(config, layer) for layer in range(config.num_layers)])
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)
        self.atten_maps = {l: torch.zeros((config.num_heads[l], config.seq_len, config.seq_len), device=config.device) for l in range(config.num_layers)}

    def forward(self, x, get_attn=False):
        if self.pos_enc == "abs":
            x = self.embed(x) + self.positional_encoding(torch.arange(x.size(1), device=x.device).view(1, x.size(1)))
        else:
            x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x, attn_map = layer(x, get_attn)
            if torch.is_tensor(attn_map):
                self.atten_maps[i] = attn_map.mean(dim=0)
            
        logits = self.output_layer(x) # (batch_size, seq_len, vocab_size)
        return logits, self.atten_maps