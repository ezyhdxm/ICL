from dataclasses import dataclass
import torch

@dataclass
class BaseConfig:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: int = None

    # Training
    batch_size: int = 256
    test_size: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config(BaseConfig):
    # Model
    emb_dim: int = 128
    num_heads: int = 2
    bias: bool = False
    ff_dim: int = None
    num_layers: int = 2
    dropout: float = None
    mask: bool = True
    mlp: tuple = (False for _ in range(num_layers))
    layer_norm: bool = True
    activation: tuple = (False for _ in range(num_layers))

    # Positional Encoding
    pos_enc: str = "rotary"
    pos_max_len: int = 32

    # Attention
    flash: bool = False

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 1000
    eval_iter: int = 50
    get_attn: int = 50
    weight_decay: float = 1e-4
    freeze_value: bool = False
    freeze_out: bool = False
    identity_query: bool = False

    # Scheduler
    scheduler: str = None
    T_max: int = 20
    
    # N-Gram
    ngram: bool = True
    max_gram: int = 4


@dataclass
class MarkovSamplerConfig(BaseConfig):
    order: int = 2
    alpha: float = 1

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
        
    