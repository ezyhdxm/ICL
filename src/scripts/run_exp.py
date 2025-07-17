# from tasks.markov import *
from models.base_models import Transformer
from icl.config import get_config
from icl.utils.train import train_model_with_plot
from tqdm.notebook import trange, tqdm

def run_exp(show=True):
    config = get_config()
    # config.update_from_yaml(yaml_path)
    # sampler_config = MarkovSamplerConfig()
    # sampler_config.update_from_yaml(yaml_path)
    
    model = Transformer(config)
    model = model.to(config.device)

    train_results = train_model_with_plot(model, config, show=show)

    return train_results, model


def run_exp_with_trans(trans_low, trans_high, every=1, log_scale=False, base=2, 
                       vocab_size=None, seq_len=None, hidden_dim=None, alpha=None, 
                       stationary=None):
    config = get_config()
    if vocab_size is not None: config.vocab_size = vocab_size
    if seq_len is not None: config.seq_len = seq_len
    if hidden_dim is not None:
        config.model.emb_dim = hidden_dim
        config.model.ff_dim = 2 * hidden_dim
    if alpha is not None: config.task.alpha = alpha
    if stationary is not None: config.task.stationary = stationary

    for total_trans in trange(trans_low, trans_high, every, desc="Number of Transitions"):
        if log_scale: config.task.total_trans = int(base**total_trans)
        else: config.task.total_trans = total_trans
        
        model = Transformer(config)
        model = model.to(config.device)

        _ = train_model_with_plot(model, config, show=False)