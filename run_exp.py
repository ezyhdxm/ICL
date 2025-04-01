# from tasks.markov import *
from models.base_models import *
from models.pos_encoder import *
from config import *
from train import *


def run_exp(yaml_path="config.yaml"):
    config = Config()
    config.update_from_yaml(yaml_path)
    sampler_config = MarkovSamplerConfig()
    sampler_config.update_from_yaml(yaml_path)
    
    model = Transformer(config)
    model = model.to(config.device)

    train_results = train_model_with_plot(model, config, sampler_config, show=True)

    return train_results, model