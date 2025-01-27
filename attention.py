from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttend(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout if config.dropout else 0.
        self.attn_dropout = nn.Dropout(self.dropout)
        FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
        self.cuda_config = FlashAttentionConfig(False, True, True) # Non-A100 GPU

    def forward(self, q, k, v):
        batch_size, heads, seq_len, _ = q.shape # (B,H,T,D)
        device = q.device
        config = self.cuda_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p = self.dropout if self.training else 0.
            )
        return out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)