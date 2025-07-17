@dataclass
class BiettiSamplerConfig:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: Optional[int] = None

    # Training
    batch_size: int = 256
    eval_size: int = 1024
    test_size: int = 4096
    task_name: str = "bietti"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    k: int = 2
    marginal: torch.Tensor = None
    trans_mat: torch.Tensor = None
    shakespeare: bool = False
    alpha: float = 1
    order: int = 1
    fixed: bool = False

    def update_from_yaml(self, yaml_path):
        """ Update class attributes from a YAML file """
        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        num_states = self.vocab_size if self.task_name == "bietti" else self.vocab_size - 1
        if not self.shakespeare:
            prior = torch.ones(num_states, device=self.device) * self.alpha
            dirichlet_dist = torch.distributions.Dirichlet(prior)
            self.trans_mat = dirichlet_dist.sample((num_states,))  # Shape: (vocab_size, vocab_size)
            self.trans_mat /= self.trans_mat.sum(dim=1, keepdim=True)
            
            def get_stationary(num_states:int, pi: torch.Tensor)->torch.Tensor:
                svd_input = pi.transpose(0, 1) - torch.eye(num_states, device=pi.device)
                _, _, v = torch.linalg.svd(svd_input)
                mu = torch.abs(v[-1, :])  # Last singular vector for each matrix
                return mu / mu.sum(dim=-1, keepdim=True)
            
            self.marginal = get_stationary(num_states, self.trans_mat)

        else:
            raise NotImplementedError("Shakespeare not implemented yet")
    

@dataclass
class LatentMarkovSamplerConfig:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: Optional[int] = None

    # Training
    batch_size: int = 256
    eval_size: int = 1024
    test_size: int = 4096
    task_name: str = "icl-mc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    order: int = 1
    alpha: float = 1
    total_trans: int = 2 

    def update_from_yaml(self, yaml_path):
        """ Update class attributes from a YAML file """
        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                

@dataclass
class MarkovSamplerConfig:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: Optional[int] = None

    # Training
    batch_size: int = 256
    test_size: int = 4096
    eval_size: int = 1024
    task_name: str = "icl-mc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    order: int = 1
    alpha: float = 1
    cardinality: int = 2
    random_alpha: float = 0.2
    dist_name: str = "dirichlet"
    random_order: int = 1
    total_trans: int = 2
    # dag: list = None
    k: int = 1
    rho: float = 0.5
    fixed: bool = False
    o_max: int = 4
    ood: bool = False

    def update_from_yaml(self, yaml_path):
        """ Update class attributes from a YAML file """
        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                

@dataclass
class Config:
    # Data
    seq_len: int = 32
    vocab_size: int = 5
    seed: Optional[int] = None

    # Training
    batch_size: int = 256
    eval_size: int = 1024
    test_size: int = 4096
    task_name: str = "icl-mc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    emb_dim: int = 128
    bias: bool = False
    mlp_bias: bool = True
    ff_dim: Optional[int] = None
    num_epochs: int = 1000
    num_layers: int = 2
    num_heads: Tuple[int, ...] = field(default_factory=lambda: (1, 1))
    dropout: Optional[float] = None
    mask: bool = True
    mlp: Tuple[bool, ...] = field(default_factory=lambda: (False, False))
    layer_norm: bool = True
    activation: Tuple[bool, ...] = field(default_factory=lambda: (False, False))

    # Positional Encoding
    pos_enc: str = "rotary"
    pos_max_len: int = 32

    # Attention
    flash: bool = True

    # Training
    learning_rate: float = 3e-4
    eval_iter: int = 50
    get_probes: int = 200
    get_attn: int = 50
    weight_decay: float = 1e-2
    freeze_value: bool = False
    freeze_out: bool = False
    identity_query: bool = False
    get_checkpoints: int = 100

    # Scheduler
    scheduler: bool = False
    T_max: int = 20
    
    # N-Gram
    ngram: int = 2

    def update_from_yaml(self, yaml_path):
        """ Update class attributes from a YAML file """
        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)

        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)