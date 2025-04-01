import torch
from torch import nn
from typing import Tuple, Callable, Dict, Any, Optional
from linear.lr_task import *
from linear.lr_models import get_model
from linear.optimize import get_optimizer_and_lr_schedule
from linear.lr_eval import get_bsln_preds, get_model_preds, mse
from linear.lr_utils import tabulate_model
from torch import optim
import os
from absl import logging

import wandb

from ml_collections import ConfigDict
import hashlib
import json



Preds = dict[str, dict[str, torch.Tensor]]

# Adapted from https://github.com/mansheej/icl-task-diversity/blob/main/icl/train.py

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################



def get_hash(config: ConfigDict) -> str:
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()


def get_sharded_batch_sampler(task: Task) -> Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    n_devices = torch.cuda.device_count() or 1  # fallback to 1 if no CUDA

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


def train(config: ConfigDict) -> None:
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Model, optimizer, schedule
    model = get_model(**config["model"], dtype=torch.float32)
    model = model.to(config.device)
    print(tabulate_model(model, config["task"]["n_dims"], config["task"]["n_points"], config["task"]["batch_size"]))

    optimizer, scheduler = get_optimizer_and_lr_schedule(**config.training, params=model.parameters())
    
    print("Initialized model, optimizer, and train state")

    # Data samplers
    train_task = get_task(**config["task"], dtype=torch.float32)
    sample_train_batch = get_sharded_batch_sampler(train_task)

    samplers_eval = {
        get_task_name(task): get_sharded_batch_sampler(task)
        for task in train_task.get_default_eval_tasks(**config["eval"])
    }
    print("Initialized data samplers")

    # Evaluate baselines
    print("Evaluating baselines...")
    bsln_preds = get_bsln_preds(train_task, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"])

    # Logging
    log = _init_log(bsln_preds, config["task"]["n_dims"])
    # wandb.init(config=config, name=exp_name, **config["wandb"])
    step = 0

    # Training loop
    print("Start training...")
    for i in range(1, config["training"]["total_steps"] + 1):
        step += 1
        data, _, targets = sample_train_batch(i)
        data = data.to(config.device)
        targets = targets.to(config.device)
        model.train()
        optimizer.zero_grad()
        
        preds = model(data, targets)
        loss = torch.mean((preds - targets) ** 2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation
        if i % config["eval"]["every"] == 0 or i == config["training"]["total_steps"]:
            print(f"Step: {i}")
            log["train/step"].append(i)
            lr_val = scheduler.get_last_lr()[0]
            log["train/lr"].append(lr_val)
            # wandb.log({"train/lr": lr_val}, step=i)

            eval_preds = get_model_preds(
                model, eval_step, samplers_eval, config["eval"]["n_samples"], config["eval"]["batch_size"]
            )

            for task_name, task_preds in bsln_preds.items():
                for bsln_name, bsln_target_preds in task_preds.items():
                    bsln_target_preds = bsln_target_preds.to(config.device)
                    errs = mse(eval_preds[task_name]["Transformer"], bsln_target_preds) / config["task"]["n_dims"]
                    log[f"eval/{task_name}"][f"Transformer | {bsln_name}"].append(errs.tolist())
                    # wandb.log({f"eval/{task_name}/{bsln_name}": errs.mean().item()}, step=i)

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