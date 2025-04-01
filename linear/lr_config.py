from ml_collections import ConfigDict
import torch


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = "linear_results"  # Specify working directory

    config.task = ConfigDict()
    config.task.name = "noisy_linear_regression"
    config.task.n_tasks = 2**5
    config.task.n_dims = 8
    config.task.n_points = 16
    config.task.batch_size = 256
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 1.0
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.n_points = 16
    config.model.n_dims = 8
    config.model.n_layer = 2
    config.model.n_embd = 128
    config.model.n_head = 1
    config.model.seed = 100

    config.training = ConfigDict()
    config.training.optimizer = "adam"
    config.training.lr = 1e-3
    config.training.schedule = "triangle"
    config.training.warmup_steps = 50_000
    config.training.total_steps = 100_000

    config.eval = ConfigDict()
    config.eval.n_samples = 2**14
    config.eval.batch_size = 4_096
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 1000

    config.wandb = ConfigDict()
    config.wandb.project = "ICL"  # Specify wandb project

    return config