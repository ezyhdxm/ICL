from dataclass import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple


class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_length):
        super().__init__()
        self.head_dim = head_dim
        self.max_length = max_length
        self.pe = nn.Parameter(torch.randn(2 * self.max_length - 1, head_dim) / head_dim ** 0.5)
        
    def forward(self, seq_len):
        distances = torch.arange(-seq_len+1, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.n_head = config.num_heads
        self.head_dim = self.emb_dim // self.n_head
        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        self.query = nn.Linear(self.emb_dim, self.emb_dim)
        self.key = nn.Linear(self.emb_dim, self.emb_dim)
        self.value = nn.Linear(self.emb_dim, self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.emb_dim)
        self.mask = torch.tril(torch.ones(config.seq_len, config.seq_len)).unsqueeze(0).unsqueeze(1)
        self.mask = self.mask.to(config.device)
        self.get_attn = config.get_attn

    def forward(self, x): # (B,T,C)
        batch_size, seq_len, embed_dim = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2)
        attn_score = Q @ K.transpose(-1,-2) / (self.head_dim ** 0.5)
        attn_score = attn_score.masked_fill(self.mask==0, -float("inf"))
        attn = F.softmax(attn_score, dim=-1) # (B,H,T,T)
        out = attn @ V # (B, H, T, D)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B, T, C)
        out = self.out(out)
        return out, attn.detach().cpu() if self.get_attn else out
        
        

class TFBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MHA = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.emb_dim) if config.layer_norm else None
        self.mlp = None
        self.dropout = None
        self.get_attn = config.get_attn

        if config.mlp:
            self.mlp = nn.Sequential(
                nn.Linear(config.emb_dim, config.ff_dim),
                nn.ReLU(),
                nn.Linear(config.ff_dim, config.emb_dim)
            )
            self.ln2 = nn.LayerNorm(config.emb_dim)
            
        if config.dropout:
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_map = None
        if self.get_attn:
            atten_out, attn_map = self.MHA(x)
        else:
            atten_out = self.MHA(x)
        x = x + self.dropout(atten_out) if self.dropout else x + atten_out
        if self.ln1 is not None:
            x = self.ln1(x)
        if self.mlp is not None:
            mlp_out = self.mlp(x)
            x = self.ln2(x + self.dropout(mlp_out))
        return x, attn_map
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim).to(config.device)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.seq_len, config.emb_dim)).to(config.device)
        self.layers = nn.ModuleList([TFBlock(config) for _ in range(config.num_layers)])
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)
        if config.get_attn:
            self.atten_maps = torch.zeros((config.num_layers, config.num_heads, config.seq_len, config.seq_len))

    def forward(self, x):
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        for i, layer in enumerate(self.layers):
            x, attn_map = layer(x)
            if config.get_attn:
                self.atten_maps[i] = attn_map.mean(dim=0)
            
        logits = self.output_layer(x)
        return logits, self.atten_maps if config.get_attn else logits