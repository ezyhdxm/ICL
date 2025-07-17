from dataclasses import dataclass, field
import torch
from typing import Tuple, Optional, Any
from ml_collections import ConfigDict # DeepMind style config library
import os


def get_config_base() -> ConfigDict:
    config = ConfigDict()
    config.seq_len = 200
    config.vocab_size = 3
    config.seed = 10086
    config.batch_size = 64
    config.eval_size = 128
    config.test_size = 512
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    TASKNAME = "latent"  # Default task name, can be overridden in config
    config.work_dir = os.path.join("results", TASKNAME)  # Specify working directory
    config.ngram = 3  # N-gram order for the n-gram learner
    config.wandb = ConfigDict()
    config.wandb.project = "ICL"  # Specify wandb project

    #####################  
    #    Tasks          #
    #####################
    
    config.task = ConfigDict()
    config.task.name = TASKNAME  # Name of the task, can be "latent", "frm", "repetition", "reversion", "fuzzy", or "dyck"
    config.task.order = 1  # Order of the Markov chain
    config.task.alpha = 1  # Dirichlet prior for the transition matrix
    config.task.ood = True  # Out-of-distribution flag
    config.task.total_trans = 3  # Total number of transitions to sample
    config.task.init_task_pool = None
    if config.task.name == "latent":
        config.task.stationary = False # Whether to use sampled stationary distribution
        config.task.pad = True  # Whether to pad the sequences with a special token

    # configurations for random triggers
    elif config.task.name == "frm":
        config.task.cardinality = 1  # Number of random distributions, this is for the frm task
        config.task.random_alpha = 1  # Dirichlet prior for random transition
        config.task.dist_name = "finite_dirichlet"  # Distribution for the transition matrix
        config.task.random_order = 1  # Random order for the Markov chain
        config.task.rho = 0.2  
        config.task.fixed = False  # Whether to fix the triggers
    

    elif config.task.name in ["repetition", "reversion", "fuzzy"]:
        config.task.repeat_length_low = 20
        config.task.repeat_length_high = 30
        config.task.length_ood = True  # Whether to generate length OOD samples
        config.task.repeat_prob = 8/config.seq_len  # Probability of repeating the sequence
    
    elif config.task.name == "dyck":
        config.task.dyck_length = 10
        config.task.repeat_prob = 0.3
    
        # config.task.pad = True  # Whether to pad the sequences with a special token
        config.task.bos_pad = True  # Whether to pad the sequences with a special token at the beginning
    

    ######################
    #     Model          #
    ######################

    NUM_LAYERS = 3 # Default number of layers, can be overridden in config

    config.model = ConfigDict()
    config.model.emb_dim = 128
    config.model.bias = False
    config.model.mlp_bias = True
    config.model.ff_dim = 4*128
    config.model.num_layers = NUM_LAYERS
    config.model.num_heads = tuple([1]*NUM_LAYERS)  # Tuple of number of heads for each layer
    config.model.dropout = None  # Dropout rate, None means no dropout
    config.model.mask = True  # Whether to use masking in attention
    config.model.mlp = tuple([False]*NUM_LAYERS)  # Tuple indicating whether to use MLP in each layer
    config.model.layer_norm = False  # Whether to use layer normalization
    config.model.activation = tuple([True]*NUM_LAYERS)  # Tuple indicating whether to use activation in each layer
    config.model.pos_enc = "rotary"  # Type of positional encoding
    config.model.pos_max_len = config.seq_len  # Maximum length for positional encoding
    config.model.flash = True  # Whether to use flash attention for faster computation
    

    #######################
    #     Training        #
    #######################

    config.training = ConfigDict()
    config.training.num_epochs = 30000
    config.training.learning_rate = 2e-4
    config.training.eval_iter = 50
    config.training.get_attn = 500
    config.training.get_checkpoints = 500
    config.training.weight_decay = 1e-2
    config.training.freeze_value = False
    config.training.freeze_out = False
    config.training.identity_query = False
    config.training.scheduler = False
    config.training.T_max = 20
    
    
    return config








        
    





