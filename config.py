from dataclasses import dataclass, field
import torch
from typing import Tuple, Optional, Any
from ml_collections import ConfigDict # DeepMind style config library
import os


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.seq_len = 512
    config.vocab_size = 10
    config.seed = None
    config.batch_size = 64
    config.eval_size = 512
    config.test_size = 4096
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "fuzzy")  # Specify working directory
    config.ngram = 3  # N-gram order for the n-gram learner
    config.wandb = ConfigDict()
    config.wandb.project = "ICL"  # Specify wandb project

    #####################  
    #    Tasks          #
    #####################
    
    config.task = ConfigDict()
    config.task.name = "fuzzy"
    config.task.order = 1  # Order of the Markov chain
    config.task.alpha = 1  # Dirichlet prior for the transition matrix
    config.task.ood = True  # Out-of-distribution flag
    config.task.total_trans = 2000  # Total number of transitions to sample
    if config.task.name == "latent":
        config.task.stationary = True # Whether to use sampled stationary distribution

    # configurations for random triggers
    elif config.task.name == "frm":
        config.task.cardinality = 1  # Number of random distributions, this is for the frm task
        config.task.random_alpha = 1  # Dirichlet prior for random transition
        config.task.dist_name = "finite_dirichlet"  # Distribution for the transition matrix
        config.task.random_order = 1  # Random order for the Markov chain
        config.task.rho = 0.2  
        config.task.fixed = False  # Whether to fix the triggers
    
    elif config.task.name in ["repetition", "reversion", "fuzzy"]:
        config.task.repeat_length = 8
        config.task.repeat_prob = 6/config.seq_len  # Probability of repeating the sequence
    
    elif config.task.name == "dyck":
        config.task.dyck_length = 8
    

    ######################
    #     Model          #
    ######################

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
    

    #######################
    #     Training        #
    #######################
    
    config.training = ConfigDict()
    config.training.num_epochs = 20000
    config.training.learning_rate = 6e-4
    config.training.eval_iter = 100
    config.training.get_attn = 1000
    config.training.get_checkpoints = 500
    config.training.weight_decay = 1e-2
    config.training.freeze_value = False
    config.training.freeze_out = False
    config.training.identity_query = False
    config.training.scheduler = False
    config.training.T_max = 20
    
    
    return config








        
    





