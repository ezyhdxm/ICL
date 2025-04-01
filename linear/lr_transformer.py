import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from models.attention import apply_rotary_emb, precompute_freqs_cis

@dataclass
class GPT2Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    dtype: Any = torch.float32
    device: str = "cpu"


class GPT2SelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.emb_dim = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.emb_dim // self.n_head
        self.seq_len = config.block_size
        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)

        self.pos_enc = "rotary" 
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.freqs_cis = precompute_freqs_cis(self.head_dim, self.seq_len * 2, # config.rotary_theta,
                                              ).to(config.device)

    def forward(self, x): # x: (B,T,C)
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        # expected shape for apply_rotary_emb: (batch_size, max_seq_len, num_head, d_head)
        Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=self.freqs_cis[:seq_len])
        Q, K = Q.transpose(1, 2), K.transpose(1, 2)
            
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
        out = self.resid_dropout(self.out(out))
        return out


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = GPT2MLP(config)

    def forward(self, x, training=False):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(self, input_embds):
        x = input_embds
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x
