from dataclasses import dataclass, field
import torch
from typing import Tuple, Optional, Any
from ml_collections import ConfigDict
import os

def get_config() -> ConfigDict:
    config = ConfigDict()
    config.seq_len = 512
    config.vocab_size = 6
    config.seed = None
    config.batch_size = 128
    config.eval_size = 512
    config.test_size = 4096
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "frm")  # Specify working directory
    config.ngram = 4  # N-gram order for the task, default is 2
    config.wandb = ConfigDict()
    config.wandb.project = "ICL"  # Specify wandb project
    
    config.task = ConfigDict()
    config.task.name = "frm"
    config.task.order = 2  # Order of the Markov chain
    config.task.alpha = 1.0  # Dirichlet prior for the transition matrix
    config.task.cardinality = 2  # Number of random distributions
    config.task.random_alpha = 0.2  # Dirichlet prior for random transition
    config.task.dist_name = "finite_dirichlet"  # Distribution for the transition matrix
    config.task.random_order = 1  # Random order for the Markov chain
    config.task.total_trans = 2**8  # Total number of transitions to sample
    config.task.rho = 0.2  
    config.task.fixed = False  # Whether to fix the transition matrix
    config.task.ood = True  # Out-of-distribution flag
    
    config.model = ConfigDict()
    config.model.emb_dim = 64
    config.model.bias = False
    config.model.mlp_bias = True
    config.model.ff_dim = 2*64
    config.model.num_layers = 2
    config.model.num_heads = (1, 1)  # Tuple of number of heads for each layer
    config.model.dropout = None  # Dropout rate, None means no dropout
    config.model.mask = True  # Whether to use masking in attention
    config.model.mlp = (False, True)  # Tuple indicating whether to use MLP in each layer
    config.model.layer_norm = False  # Whether to use layer normalization
    config.model.activation = (False, True)  # Tuple indicating whether to use activation in each layer
    config.model.pos_enc = "rotary"  # Type of positional encoding
    config.model.pos_max_len = 256  # Maximum length for positional encoding
    config.model.flash = True  # Whether to use flash attention for faster computation
    
    
    config.training = ConfigDict()
    config.training.num_epochs = 60000
    config.training.learning_rate = 3e-4
    config.training.eval_iter = 1000
    config.training.get_probes = 100
    config.training.get_attn = 1000
    config.training.get_checkpoints = 1000
    config.training.weight_decay = 1e-2
    config.training.freeze_value = False
    config.training.freeze_out = False
    config.training.identity_query = False
    config.training.scheduler = False
    config.training.T_max = 20
    
    
    return config








        
    





