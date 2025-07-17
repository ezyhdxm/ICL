import torch
from torch import nn
from typing import Tuple, Callable, Dict, Any, Optional, Union
from torch import optim
import os
from absl import logging

import wandb

from ml_collections import ConfigDict
import hashlib
import json

# Import the task classes
from icl.linear.lr_models import get_model
from icl.linear.optimize import get_optimizer_and_lr_schedule
from icl.linear.lr_eval import get_model_preds, mse
from icl.linear.lr_eval import get_baseline_step, get_model_name
from icl.linear.lr_utils import tabulate_model
from icl.linear.mixture_task import DiscreteInputLinearRegression, MixtureOfGaussiansRegression

Preds = dict[str, dict[str, torch.Tensor]]

########################################################################################################################
# Task-specific functions                                                                                              #
########################################################################################################################

def get_task(name: str, **kwargs) -> Union[DiscreteInputLinearRegression, MixtureOfGaussiansRegression]:
    """Get task instance based on name."""
    if name == "discrete_input_regression":
        return DiscreteInputLinearRegression(**kwargs)
    elif name == "mixture_of_gaussians_regression":
        return MixtureOfGaussiansRegression(**kwargs)
    else:
        raise ValueError(f"Unknown task name: {name}")

def get_task_name(task: Union[DiscreteInputLinearRegression, MixtureOfGaussiansRegression]) -> str:
    """Get display name for task."""
    return "Latent" if task.name.endswith("(0)") else "Pretrain"

# Type alias for clarity
Task = Union[DiscreteInputLinearRegression, MixtureOfGaussiansRegression]

########################################################################################################################
# Enhanced samplers for mixture support                                                                                #
########################################################################################################################

class MixtureAwareSampler:
    """Wrapper that preserves component information for mixture tasks."""
    def __init__(self, task: MixtureOfGaussiansRegression):
        self.task = task
    
    def __call__(self, step: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns ((data, component_labels), (tasks1, tasks2), targets)"""
        data, component_labels = self.task.sample_data(step)
        tasks1, tasks2 = self.task.sample_tasks(step)
        targets = self.task.evaluate(data, tasks1, tasks2, component_labels, step)
        
        return (data, component_labels), (tasks1, tasks2), targets

def get_sharded_batch_sampler(task: Task) -> Callable:
    """Create a batch sampler that handles both task types."""
    n_devices = 1  # fallback to 1 if no CUDA
    
    # Check if this is a mixture task
    is_mixture = hasattr(task, 'centroid1')
    
    if is_mixture:
        base_sampler = MixtureAwareSampler(task)
    else:
        base_sampler = task.sample_batch

    def sample_batch(step: int) -> Tuple:
        if is_mixture:
            data_component_labels, all_tasks, targets = base_sampler(step)
            data, component_labels = data_component_labels
            tasks1, tasks2 = all_tasks
            # Store component labels in task object for oracle evaluation
            task._last_component_labels = component_labels
            task._last_step = step
            task._tasks = (tasks1, tasks2)
            tasks = (tasks1 + tasks2) // 2  # Combine tasks for mixture to keep the format consistent
        else:
            data, tasks, targets = task.sample_batch(step)
        
        batch_size = data.shape[0]
        assert batch_size % n_devices == 0, "Batch size must be divisible by number of devices"
        per_device = batch_size // n_devices

        def reshape(x):
            return x.view(n_devices, per_device, *x.shape[1:])

        return reshape(data), reshape(tasks), reshape(targets)

    return sample_batch

########################################################################################################################
# Modified baseline evaluation                                                                                         #
########################################################################################################################

def get_oracle_step(task: Task) -> Callable:
    """Get oracle evaluation function that handles both task types."""
    if isinstance(task, MixtureOfGaussiansRegression):
        def step(xs: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
            # Check if we have stored component labels
            if hasattr(task, '_last_component_labels') and hasattr(task, '_last_step'):
                tasks1, tasks2 = task._tasks
                component_labels = task._last_component_labels
                step = task._last_step

                T = xs.shape[1]

                # Expand tasks to shape (B, T, D)
                tasks1_exp = tasks1.squeeze(-1)[:, None, :].expand(-1, T, -1)
                tasks2_exp = tasks2.squeeze(-1)[:, None, :].expand(-1, T, -1)

                # Select appropriate task vector for each position
                tasks = torch.where(component_labels[..., None] == 0, tasks1_exp, tasks2_exp)

                # Element-wise product and sum across D (dot product): (B, T)
                targets = (xs * tasks).sum(dim=-1)
                
                return targets
            else:
                # Fallback to simple linear prediction
                return (xs @ ws).squeeze(-1)
        return step
    else:
        def step(xs: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
            return task.evaluate_oracle(xs, ws)
        return step

def get_bsln_preds(
    train_task: Task,
    batch_samplers: Dict[str, Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    n_samples: int,
    batch_size: int
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Get baseline predictions for both task types."""
    preds = {}

    # Oracle prediction function (fixed for task)
    oracle_fn = get_oracle_step(train_task)

    # Baseline models
    baseline_models = {
        get_model_name(model): get_baseline_step(model)
        for model in train_task.get_default_eval_models()
    }

    for task_name, sample_batch_fn in batch_samplers.items():

        preds[task_name] = {"True": []}
        for model_name in baseline_models:
            preds[task_name][model_name] = []

        for i in range(1, n_samples // batch_size + 1):
            xs, ws, ys = sample_batch_fn(i)
            _, _, n_points = ys.shape

            # Oracle predictions
            true_preds = oracle_fn(xs, ws).reshape(batch_size, n_points)
            preds[task_name]["True"].append(true_preds)

            # Baseline model predictions
            for model_name, model_fn in baseline_models.items():
                pred = model_fn(xs, ys).reshape(batch_size, n_points)
                preds[task_name][model_name].append(pred)

        # Concatenate all collected predictions along batch axis
        preds[task_name]["True"] = torch.cat(preds[task_name]["True"], dim=0)
        for model_name in baseline_models:
            preds[task_name][model_name] = torch.cat(preds[task_name][model_name], dim=0)

    return preds

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################

def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()

def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    """Initialize log dictionary for evaluation metrics."""
    log = {"train/step": [], "train/lr": []}
    for _task_name, _task_preds in bsln_preds.items():
        log[f"eval/{_task_name}"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}"][f"{_bsln_name} | True"] = _errs.tolist()
    return log

@torch.no_grad()
def eval_step(model, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    model.eval()
    data = data.to(model.device)
    targets = targets.to(model.device)
    preds = model(data, targets)
    return preds

########################################################################################################################
# Main training function                                                                                               #
########################################################################################################################

def train(config: ConfigDict, verbose=False) -> Tuple[nn.Module, dict]:
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   

    cur_dir = os.getcwd()
    if cur_dir.endswith("notebooks"):
        exp_dir = os.path.join("..", exp_dir)

    data_type = getattr(torch, config.dtype)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
        log_path = os.path.join(exp_dir, "log.json")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model = get_model(**config["model"], dtype=data_type)
        model.load_state_dict(checkpoint["model"])
        model = model.to(config.device)
        print(f"Loaded model from {checkpoint_path}")
        return model, json.load(open(log_path, "r"))
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Model, optimizer, schedule
    model = get_model(**config["model"], dtype=data_type)
    model = model.to(config.device)
    if verbose:
        print(tabulate_model(model, config["task"]["n_dims"], config["task"]["n_points"], config["task"]["batch_size"]))

    optimizer, scheduler = get_optimizer_and_lr_schedule(**config.training, params=model.parameters())
    
    if verbose:
        print("Initialized model, optimizer, and train state")

    # Data samplers
    train_task = get_task(**config["task"], dtype=data_type)
    sample_train_batch = get_sharded_batch_sampler(train_task)

    samplers_eval = {
        get_task_name(task): get_sharded_batch_sampler(task)
        for task in train_task.get_default_eval_tasks(**config["eval"])
    }
    if verbose:
        print(f"Initialized data samplers for task: {config.task.name}")

    # Evaluate baselines
    if verbose:
        print("Evaluating baselines...")
    bsln_preds = get_bsln_preds(train_task, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"])

    # Logging
    log = _init_log(bsln_preds, config["task"]["n_dims"])
    wandb.init(config=config, name=exp_name, **config["wandb"])
    step = 0

    scaler = torch.amp.GradScaler("cuda") if config.device == "cuda" else None

    # Training loop
    print("Start training...")
    for i in range(1, config["training"]["total_steps"] + 1):
        step += 1
        data, _, targets = sample_train_batch(i)
        data = data.to(config.device)
        targets = targets.to(config.device)
        model.train()
        optimizer.zero_grad()

        if config.device == "cuda" and scaler is not None:
            with torch.amp.autocast("cuda"):
                preds = model(data, targets)
                loss = torch.mean((preds - targets) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(data, targets)
            loss = torch.mean((preds - targets) ** 2)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        # Evaluation
        if i % config["eval"]["every"] == 0 or i == config["training"]["total_steps"]:
            log["train/step"].append(i)
            lr_val = scheduler.get_last_lr()[0]
            log["train/lr"].append(lr_val)
            wandb.log({"train/lr": lr_val}, step=i)

            eval_preds = get_model_preds(
                model, eval_step, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"]
            )

            for task_name, task_preds in bsln_preds.items():
                for bsln_name, bsln_target_preds in task_preds.items():
                    bsln_target_preds = bsln_target_preds.to(config.device)
                    errs = mse(eval_preds[task_name]["Transformer"], bsln_target_preds) / config["task"]["n_dims"]
                    log[f"eval/{task_name}"][f"Transformer | {bsln_name}"].append(errs.tolist())
                    wandb.log({f"eval/{task_name}/{bsln_name}": errs.mean().item()}, step=i)

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, os.path.join(exp_dir, "checkpoint.pt"))

    # Save logs
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")

    return model, log

# Configuration functions
def get_discrete_config():
    """Configuration for training with discrete input regression."""
    config = ConfigDict()

    NPOINTS = 64

    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "discrete_input_regression")

    config.task = ConfigDict()
    config.task.name = "discrete_input_regression"
    config.task.n_tasks = 3
    config.task.n_discrete_vectors = 20
    config.task.n_points = NPOINTS
    config.task.batch_size = 128
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 1.0
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5
    config.task.n_dims = 6

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.n_points = NPOINTS
    config.model.n_dims = 6
    config.model.n_layer = 3
    config.model.n_embd = 128
    config.model.n_head = 1
    config.model.seed = 100
    config.model.pad = "mapsto"

    config.training = ConfigDict()
    config.training.optimizer = "adamw"
    config.training.lr = 4e-4
    config.training.schedule = "triangle"
    config.training.weight_decay = 1e-2
    config.training.warmup_steps = 20_000
    config.training.total_steps = 40_000

    config.eval = ConfigDict()
    config.eval.n_samples = 2**12
    config.eval.batch_size = 512
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 100

    config.wandb = ConfigDict()
    config.wandb.project = "ICL"

    return config

def get_mixture_config():
    """Configuration for training with mixture of Gaussians regression."""
    config = ConfigDict()

    NPOINTS = 64

    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "mixture_of_gaussians_regression")

    config.task = ConfigDict()
    config.task.name = "mixture_of_gaussians_regression"
    config.task.n_tasks = 3
    config.task.n_points = NPOINTS
    config.task.batch_size = 128
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 0.5
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5
    config.task.mixture_prob = 0.5
    config.task.n_dims = 6

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.n_points = NPOINTS
    config.model.n_dims = 6
    config.model.n_layer = 3
    config.model.n_embd = 128
    config.model.n_head = 1
    config.model.seed = 100
    config.model.pad = "mapsto"

    config.training = ConfigDict()
    config.training.optimizer = "adamw"
    config.training.lr = 5e-4
    config.training.schedule = "triangle"
    config.training.weight_decay = 1e-2
    config.training.warmup_steps = 10_000
    config.training.total_steps = 20_000

    config.eval = ConfigDict()
    config.eval.n_samples = 2**10
    config.eval.batch_size = 128
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 100

    config.wandb = ConfigDict()
    config.wandb.project = "ICL"

    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on discrete or mixture regression tasks")
    parser.add_argument("--task", type=str, choices=["discrete", "mixture"], 
                        default="discrete", help="Which task to train on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.task == "discrete":
        config = get_discrete_config()
        print("Training on Discrete Input Regression task...")
    else:
        config = get_mixture_config()
        print("Training on Mixture of Gaussians Regression task...")
    
    model, log = train(config, verbose=args.verbose)