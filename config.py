from dataclasses import dataclass

@dataclass
class Config:
    # Model
    emb_dim: int = 128
    num_heads: int = 2
    ff_dim: int = None
    num_layers: int = 2
    seq_len: int = 32
    vocab_size: int = 5
    dropout: float = None
    mask: bool = True
    mlp: bool = False
    layer_norm: bool = True

    # Positional Encoding
    pos_enc: str = "rotary"
    pos_max_len: int = 32

    # Attention
    flash: bool = False

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    eval_iter: int = 50
    get_attn: int = 50
    device: str = "cuda"
    weight_decay: float = 1e-4
    
    # Data
    order: int = 2
    alpha: float = 1
    test_size: int = 2048