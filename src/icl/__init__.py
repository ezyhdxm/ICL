__version__ = "0.1.0"
__author__ = "Hao Yan"

# Core imports for convenience
from icl.config import get_config_base
from icl.models import Transformer
from icl.utils import (
    get_attn_base,
    visualize_attention,
    extract_task_vector_markov,
    predict_with_task_vector_markov,
    train_model_with_plot,
    train_model,
    load_everything,
    view_mask,
)
from icl.linear import (
    plot_task_vector_differences,
    plot_task_vector_variance_with_fit,
    plot_pairwise_task_vector_variance,
    extract_task_vector,
    predict_with_task_vector,
    get_config,
    get_attn,
    compute_task_vectors,
    DiscreteMMSE,
    Ridge
)
from icl.tasks import (
    DyckPathTask,
    RepetitionTask,
    FuzzyCopyTask,
    ReversedTask,
    CoinTask,
    CoinBayes,
    DyckBayes,
    LatentMarkov,
    LatentIDBayes,
    LatentOODBayes,
)

# Define what gets imported with "from icl import *"
__all__ = [
    "get_config_base",
    "Transformer",
    "get_attn_base",
    "visualize_attention",
    "extract_task_vector_markov",  
    "predict_with_task_vector_markov",
    "train_model_with_plot",
    "train_model",
    "load_everything",
    "view_mask",
    "plot_task_vector_differences",
    "plot_task_vector_variance_with_fit",
    "plot_pairwise_task_vector_variance",
    "extract_task_vector",
    "predict_with_task_vector",
    "get_attn",
    "compute_task_vectors",
]