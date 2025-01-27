from dataclasses import dataclass
import itertools
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
from pos_encoder import *
from attention import *


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
        self.pos_enc = config.pos_enc
        self.seq_len = config.seq_len
        self.scale = self.head_dim ** 0.5
        self.flash = config.flash
        assert not (self.flash and self.pos_enc == "rpe"), "Flash Attention does not support RPE currently."  
        if self.flash:
            self.flashAttend = FlashAttend(config)
        if self.pos_enc == "rpe":
            self.PEK = RelativePositionalEncoding(self.head_dim, self.pos_max_len) # (T,T,D)
            self.PEV = RelativePositionalEncoding(self.head_dim, self.pos_max_len) # (T,T,D)
        elif self.pos_enc == "rotary":
            self.rotary_emb = RotaryPositionalEmbeddings(self.head_dim, self.pos_max_len)
        

    def forward(self, x): # x: (B,T,C)
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        if self.pos_enc == "rotary":
            Q = self.rotary_emb(Q)
            K = self.rotary_emb(K)
        if self.flash:
            assert self.get_attn == 0, "Flash Attention does not output attentions."
            return self.flashAttend(Q, K, V)
        
        attn_score = Q @ K.transpose(-1,-2) / self.scale # (B,H,T,T)
        if self.pos_enc == "rpe":
            Q2 = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(0,1) # (T,B,H,D)
            Q2 = Q2.contiguous().view(seq_len, batch_size*self.n_head, self.head_dim) # (T,BH,D)
            attn_score2 = torch.matmul(Q2, self.PEK(seq_len).transpose(1,2)) # (T,BH,D) @ (T,D,T) -> (T,BH,T)
            attn_score2 = attn_score2.view(seq_len, batch_size, self.n_head, seq_len).transpose(0,1).contiguous() # (B,H,T,T)
            attn_score += attn_score2 / self.scale
        attn_score = attn_score.masked_fill(self.mask==0, -float("inf"))
        attn = F.softmax(attn_score, dim=-1) # (B,H,T,T)
        out = attn @ V # (B,H,T,D)
        if self.pos_enc == "rpe":
            attn2 = attn.transpose(0,2).contiguous().view(seq_len, -1, seq_len) # (T,BH,T)
            out2 = torch.matmul(attn2, self.PEV(seq_len)) # (T,BH,T) @ (T,T,D) -> (T,BH,D)
            out2 = out2.view(seq_len, batch_size, -1, self.head_dim).transpose(0,2) # (B,H,T,D)
            out += out2
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
        out = self.out(out)
        if self.get_attn > 0:
            return out, attn.detach().cpu() 
        return out
        
        

class TFBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MHA = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.emb_dim) if config.layer_norm else None
        self.mlp = None
        self.dropout = None
        self.get_attn = config.get_attn

        if config.mlp:
            assert config.ff_dim is not None, "FeedForward dimension cannot be empty."
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
            if self.dropout is not None:
                x = self.ln2(x + self.dropout(mlp_out))
            else:
                x = self.ln2(x + mlp_out)
        if self.get_attn > 0:
            return x, attn_map 
        return x
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim).to(config.device)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.seq_len, config.emb_dim)).to(config.device)
        self.layers = nn.ModuleList([TFBlock(config) for _ in range(config.num_layers)])
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)
        self.get_attn = config.get_attn
        if config.get_attn > 0:
            self.atten_maps = torch.zeros((config.num_layers, config.num_heads, config.seq_len, config.seq_len))

    def forward(self, x):
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        for i, layer in enumerate(self.layers):
            if self.get_attn > 0:
                x, attn_map = layer(x)
                self.atten_maps[i] = attn_map.mean(dim=0)
            else:
                x = layer(x)
            
        logits = self.output_layer(x)
        if self.get_attn > 0: 
            return logits, self.atten_maps 
        return logits