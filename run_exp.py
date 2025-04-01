# from tasks.markov import *
from models.base_models import Transformer
from config import get_config
from train import train_model_with_plot


def run_exp():
    config = get_config()
    # config.update_from_yaml(yaml_path)
    # sampler_config = MarkovSamplerConfig()
    # sampler_config.update_from_yaml(yaml_path)
    
    model = Transformer(config)
    model = model.to(config.device)

    train_results = train_model_with_plot(model, config, show=True)

    return train_results, model