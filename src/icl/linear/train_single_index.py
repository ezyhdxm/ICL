import torch
from torch import nn
from typing import Tuple, Callable, Dict, Any, Optional
from torch import optim
import os
from absl import logging
import wandb
from ml_collections import ConfigDict
import hashlib
import json

from icl.linear.single_index_task import SingleIndexRegression
from icl.linear.lr_models import get_model
from icl.linear.optimize import get_optimizer_and_lr_schedule
from icl.linear.lr_eval import get_bsln_preds, get_model_preds, mse
from icl.linear.lr_utils import tabulate_model

Preds = dict[str, dict[str, torch.Tensor]]

########################################################################################################################
# Task-specific functions                                                                                              #
########################################################################################################################

def get_task(name: str, **kwargs) -> SingleIndexRegression:
    """Get task instance based on name."""
    if name == "single_index_regression":
        return SingleIndexRegression(**kwargs)
    else:
        raise ValueError(f"Unknown task name: {name}")

def get_task_name(task: SingleIndexRegression) -> str:
    """Get display name for task."""
    return "Latent" if task.name.endswith("(0)") else "Pretrain"

# Type alias for clarity
Task = SingleIndexRegression

# Adapted from https://github.com/mansheej/icl-task-diversity/blob/main/icl/train.py

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################



def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def get_sharded_batch_sampler(task: Task) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n_devices = 1 #torch.cuda.device_count() or 1  # fallback to 1 if no CUDA

    def sample_batch(step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, tasks, targets = task.sample_batch(step)
        batch_size = data.shape[0]

        assert batch_size % n_devices == 0, "Batch size must be divisible by number of devices"
        per_device = batch_size // n_devices

        def reshape(x):
            return x.view(n_devices, per_device, *x.shape[1:])

        return reshape(data), reshape(tasks), reshape(targets)

    return sample_batch

def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    Args:
        bsln_preds: baseline predictions
        n_dims: number of dimensions
    """
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


def train(config: ConfigDict, verbose=False) -> None:
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   
    # logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")

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
        print("Initialized data samplers")

    # Evaluate baselines
    if verbose:
       print("Evaluating baselines...")
    bsln_preds = get_bsln_preds(train_task, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"])


    # Logging
    log = _init_log(bsln_preds, config["task"]["n_dims"])
    wandb.init(config=config, name=exp_name, **config["wandb"])
    step = 0

    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    print("Start training...")
    for i in range(1, config["training"]["total_steps"] + 1):
        step += 1
        data, _, targets = sample_train_batch(i)
        data = data.to(config.device)
        targets = targets.to(config.device)
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            preds = model(data, targets)
            loss = torch.mean((preds - targets) ** 2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step > 1:
            scheduler.step()

        # Evaluation
        if i % config["eval"]["every"] == 0 or i == config["training"]["total_steps"]:
            # print(f"Step: {i}")
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















def get_config():
    """Configuration for training with single index functions."""
    config = ConfigDict()

    NDIMS = 6
    NPOINTS = 256

    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = os.path.join("results", "single_index_regression")

    config.task = ConfigDict()
    config.task.name = "single_index_regression"
    config.task.n_tasks = 3
    config.task.n_dims = NDIMS # Must be even for mixture Gaussian
    config.task.n_points = NPOINTS
    config.task.batch_size = 64
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 1.0
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5
    config.task.link_function = "chirp"  # Can be 'identity', 'sigmoid', etc.
    config.task.link_scale = 5.0  # Scale for the link function output

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.n_points = NPOINTS
    config.model.n_dims = NDIMS
    config.model.n_layer = 3
    config.model.n_embd = 128
    config.model.n_head = 1
    config.model.seed = 100
    config.model.pad = "mapsto"  # Padding strategy, can be "bos" or "mapsto"

    config.training = ConfigDict()
    config.training.optimizer = "adamw"
    config.training.lr = 4e-4
    config.training.schedule = "triangle"
    config.training.weight_decay = 1e-2
    config.training.warmup_steps = 20_000
    config.training.total_steps = 40_000

    config.eval = ConfigDict()
    config.eval.n_samples = 2**11
    config.eval.batch_size = 128
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 100

    config.wandb = ConfigDict()
    config.wandb.project = "ICL"  # Specify wandb project

    return config