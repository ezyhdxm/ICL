from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from pos_encoder import *

# TODO: Add MLA, MQA, GQA

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.n_head = config.num_heads[layer]
        self.head_dim = self.emb_dim // self.n_head
        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        if config.identity_query:
            self.query = nn.Identity()
        else:
            self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        if config.freeze_value:
            self.value.weight.requires_grad_(False)
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=config.bias)
        if config.freeze_out:
            self.out.weight.requires_grad_(False)
        self.mask = torch.tril(torch.ones((config.seq_len, config.seq_len), device=config.device)).unsqueeze(0).unsqueeze(1) # TODO: make self.mask a register_buffer
        self.get_attn = config.get_attn
        self.pos_enc = config.pos_enc
        self.seq_len = config.seq_len
        self.scale = self.head_dim ** 0.5
        self.flash = config.flash
        self.dropout = config.dropout if config.dropout else 0.
        assert not (self.flash and self.pos_enc == "rpe"), "Flash Attention does not support RPE currently."  
        if self.pos_enc == "rpe":
            if not self.flash:
                self.PEV = RelativePositionalEncoding(self.head_dim, config.pos_max_len) # (T,T,D)
                self.PEK = RelativePositionalEncoding(self.head_dim, config.pos_max_len) # (T,T,D)
            elif config.device == "cuda":
                self.rpe = torch.randn((2*config.pos_max_len+1, self.head_dim), device=config.device) / (self.head_dim ** 0.5)
                
            else:
                raise ValueError("Flash Attention with RPE is currently only supported on CUDA devices.")
        
        elif self.pos_enc == "rotary":
            self.rotary_emb = RotaryPositionalEmbeddings(self.head_dim, config.pos_max_len)
        elif self.pos_enc == "alibi":
            self.alibi_emb = AliBiPositionalEncoding(self.n_head)
    

    def forward(self, x, get_attn): # x: (B,T,C)
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        if self.pos_enc == "rotary":
            Q = self.rotary_emb(Q)
            K = self.rotary_emb(K)
            
        if self.flash and (not get_attn):
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout, is_causal=True)
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            return out, -1
        else:
            attn_score = Q @ K.transpose(-1,-2) / self.scale # (B,H,T,T)
            if self.pos_enc == "rpe":
                Q2 = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(0,1) # (T,B,H,D)
                Q2 = Q2.contiguous().view(seq_len, batch_size*self.n_head, self.head_dim) # (T,BH,D)
                attn_score2 = torch.matmul(Q2, self.PEK(seq_len).transpose(1,2)) # (T,BH,D) @ (T,D,T) -> (T,BH,T)
                attn_score2 = attn_score2.view(seq_len, self.n_head, batch_size, seq_len).transpose(0,2).contiguous() # (B,H,T,T)
                attn_score += attn_score2 / self.scale
            elif self.pos_enc=="alibi":
                attn_score += self.alibi_emb(self.seq_len)

            attn_score = attn_score.masked_fill(self.mask==0, -float("inf"))
            attn = F.softmax(attn_score, dim=-1) # (B,H,T,T)
            out = attn @ V # (B,H,T,D)
            if self.pos_enc == "rpe":
                attn2 = attn.transpose(0,2).contiguous().view(seq_len, -1, seq_len) # (T,BH,T)
                out2 = torch.matmul(attn2, self.PEV(seq_len)) # (T,BH,T) @ (T,T,D) -> (T,BH,D)
                out2 = out2.view(seq_len, -1, batch_size, self.head_dim).transpose(0,2) # (B,H,T,D)
                out += out2
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            if get_attn:
                return out, attn.detach()
            else:
                return out, -1