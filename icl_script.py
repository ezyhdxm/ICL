import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from dataclasses import dataclass
from collections import namedtuple, defaultdict
from markov import *
from base_models import *
from pos_encoder import *
from causal_graph import *
from config import *
import train
import plot
from util import memory_recall_probe, feedforward_probe
import seaborn as sns
from tqdm import trange
train.trange = trange

import sys
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no).")

class DynamicNargsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Fetch the value of another argument that determines nargs
        num_values = getattr(namespace, "num_layers", None)
        if num_values is None:
            parser.error("--num must be provided before --values")

        # Ensure values match the expected count
        if len(values) != num_values:
            parser.error(f"--values requires exactly {num_values} integers")

        setattr(namespace, self.dest, tuple(values))  # Store as tuple

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ICL Parser")
    parser.add_argument('--task_name', '-tn', type=str, choices=["icl-mc", "markov", "tree", "dag", "bietti", "bb"], required=True)
    parser.add_argument('--sequence_length', '-sl', type=int, required=True)  
    parser.add_argument('--vocab_size', '-vs', type=int, required=True) 
    parser.add_argument('--batch_size', '-bs', type=int, required=True)
    parser.add_argument('--embedding_dim', '-ed', type=int, default=16)
    parser.add_argument('--num_layers', '-nl', type=int, default=2)
    parser.add_argument('--num_heads', '-nh', type=int, nargs="+", default=(1,1), action=DynamicNargsAction, help="Number of heads for each layer, must be same length as num_layers")

    parser.add_argument('--num_epochs', '-n', type=int, default=2000)
    parser.add_argument('--eval_iter', '-ei', type=int, default=20)  
    parser.add_argument('--get_attention', '-ga', type=int, default=10) 

    parser.add_argument('--position_encoding', '-pe', type=str, choices=["abs", "rpe", "rotary", "alibi"], default='rpe')
    parser.add_argument('--position_max_len', '-pm', type=int, default=100) 
    
    parser.add_argument('--identity_query', '-iq', action="store_true") 
    parser.add_argument('--mlp', '-mlp', type=str2bool, nargs="+", default=("False", "False"), action=DynamicNargsAction, 
                        help="Whether to use a feedforward layer in each layer (True/False), must be same length as num_layers")
    parser.add_argument('--activation', '-ac', type=str2bool, nargs="+", default=("False", "False"), action=DynamicNargsAction, 
                        help="Whether to use an activation function in each layer (True/False), must be same length as num_layers")
    parser.add_argument('--flash', '-f', action='store_true')
    parser.add_argument('--feedforward_dim', '-ff', type=int, default=16)
    parser.add_argument('--layer_norm', '-ln', action='store_true')
    
    parser.add_argument('--ngram_learner', '-ng', action='store_true')
    parser.add_argument('--max_gram', '-mg', type=int, default=2)

    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-4)
    

    parser.add_argument('--order', '-o', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)
    
    config = Config(
            emb_dim=params['embedding_dim'],
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            identity_query=params['identity_query'],
            seq_len=params['sequence_length'],
            vocab_size=params['vocab_size'],
            batch_size=params['batch_size'],
            num_epochs=params['num_epochs'],
            eval_iter=params['eval_iter'],
            pos_enc=params['position_encoding'],
            pos_max_len=params['position_max_len'],
            get_attn=params['get_attention'],
            mlp=params['mlp'],
            activation=params['activation'],
            flash=params['flash'],
            ff_dim=params['feedforward_dim'],
            layer_norm=params['layer_norm'],
            ngram=params['ngram_learner'],
            learning_rate=params['learning_rate'],
            max_gram=params['max_gram'],
            task_name=params['task_name'],
        )

    task_name = params['task_name']
    if task_name in ["icl-mc", "markov"]:
        sampler_config = MarkovSamplerConfig(seq_len=params['sequence_length'],
                                            vocab_size=params['vocab_size'],
                                            batch_size=params['batch_size'],
                                            order=params['order'],
                                            task_name=params['task_name'],
                                            )
    elif task_name in ["dag", "tree"]:
        sampler_config = CausalGraphConfig(seq_len=params['sequence_length'],
                                           vocab_size=params['vocab_size'],
                                           batch_size=params['batch_size'],
                                           task_name=params['task_name'],
                                        )
    elif task_name=="bb":
        sampler_config = BBSamplerConfig(seq_len=params['sequence_length'],
                                         vocab_size=params['vocab_size'],
                                         batch_size=params['batch_size'],
                                         task_name=params['task_name'],
                                    )
    elif task_name=='bietti':
        sampler_config = BiettiSamplerConfig(seq_len=params['sequence_length'],
                                             vocab_size=params['vocab_size'],
                                             batch_size=params['batch_size'],
                                             task_name=params['task_name'],
                                            )
    else:
        raise ValueError("Task name not recognized")
        
        

    model = Transformer(config).to(config.device)
    train_results = train.train_model(model, config, sampler_config)
    plot.get_loss_plots(config, train_results)
    


if __name__ == "__main__":
    main()