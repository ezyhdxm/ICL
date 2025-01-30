from dataclasses import dataclass
import torch
from typing import Tuple, Optional

@dataclass
class BaseConfig:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: Optional[int] = None

    # Training
    batch_size: int = 256
    test_size: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config(BaseConfig):
    # Model
    emb_dim: int = 128
    bias: bool = False
    ff_dim: Optional[int] = None
    num_layers: int = 2
    num_heads: Tuple[int] = (1,1)
    dropout: Optional[float] = None
    mask: bool = True
    mlp: Tuple[bool] = (False for _ in range(num_layers))
    layer_norm: bool = True
    activation: Tuple[bool] = (False for _ in range(num_layers))

    # Positional Encoding
    pos_enc: str = "rotary"
    pos_max_len: int = 32

    # Attention
    flash: bool = True

    # Training
    learning_rate: float = 3e-4
    num_epochs: int = 1000
    eval_iter: int = 50
    get_attn: int = 50
    weight_decay: float = 1e-2
    freeze_value: bool = False
    freeze_out: bool = False
    identity_query: bool = False

    # Scheduler
    scheduler: Optional[str] = None
    T_max: int = 20
    
    # N-Gram
    ngram: bool = True
    max_gram: int = 4


@dataclass
class MarkovSamplerConfig(BaseConfig):
    order: int = 2
    alpha: float = 1


@dataclass
class CausalGraphConfig(BaseConfig):
    alpha: float = 1
    dag: list = None

@dataclass
class BiettiSamplerConfig(BaseConfig):
    k: int = 2
    show_latents: bool = False
    marginal: torch.Tensor = None
    trans_mat: torch.Tensor = None
    show_mask: bool = False
    shakespeare: bool = False

    def __post_init__(self):
        self.marginal = torch.ones((self.vocab_size,)) / self.vocab_size
        if not self.shakespeare:
            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.vocab_size))
            self.trans_mat = dirichlet_dist.sample((self.vocab_size,))  # Shape: (num_states_order, num_states)
            self.trans_mat /= self.trans_mat.sum(dim=1, keepdim=True)
        else:
            raise NotImplementedError("Shakespeare not implemented yet")

@dataclass
class BBSamplerConfig(BaseConfig):
    k: int = 2
    marginal: torch.Tensor = None
    trans_mat: torch.Tensor = None
    show_mask: bool = False
    shakespeare: bool = False

    def __post_init__(self):
        self.marginal = torch.ones((self.vocab_size-1,)) / (self.vocab_size-1)
        if not self.shakespeare:
            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.vocab_size-1))
            self.trans_mat = dirichlet_dist.sample((self.vocab_size-1,))  # Shape: (num_states_order, num_states)
            self.trans_mat /= self.trans_mat.sum(dim=1, keepdim=True)
        else:
            raise NotImplementedError("Shakespeare not implemented yet")
        
    