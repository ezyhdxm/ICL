from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn  
from tasks.markov import *
from models.ngram_latent import *
from models.ngram_trigger import *
from collections import defaultdict
# from tasks.causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from tasks.repetition import RepetitionTask
from tasks.fuzzy_copy import FuzzyCopyTask
from tasks.reversion import ReversedTask
from probe_util import *
from absl import logging

import hashlib

# from torch.utils.data import DataLoader
from train_utils import *
from figures.plot import *
from IPython.display import display, HTML
from figures.head_view import *
import pickle

import wandb

def get_hash(config: ConfigDict) -> str:
    """
    Generate a hash for the given configuration.
    This is used to identify the experiment uniquely.
    Args:
        config (ConfigDict): The configuration object.
    Returns:
        str: The hash of the configuration.
    """
    return hashlib.md5(config.to_json(sort_keys=True).encode("utf-8")).hexdigest()

# Train model based on task
def train_model(model, config):
    if config.task.name in ["frm", "bietti"]:
        return train_trigger(model, config)
    elif config.task.name in ["icl-mc", "latent"]:
        return train_latent(model, config)
    elif config.task.name in ["repetition", "reversion", "fuzzy"]:
        return train_markov(model, config)
    
    raise NotImplementedError(f"Task '{config.task.name}' not implemented yet or is a legacy task. Please choose 'frm', 'bietti', 'icl-mc', or 'latent' for training.")

def get_sampler(config):
    task_samplers = {
        "markov": MarkovSampler,
        "icl-mc": ICLMarkovSampler,
        "frm": FRMarkovSampler,
        "latent": LatentMarkov,
        "repetition": RepetitionTask,
        "reversion": ReversedTask,
        "fuzzy": FuzzyCopyTask,
    }
    if config.task.name in task_samplers:
        return task_samplers[config.task.name](config)
    raise NotImplementedError(f"Task '{config.task.name}' not implemented yet.")

def _init_log() -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    """
    log = {"train/step": [], "train/lr": [], "train/loss": [], 
           "baseline": {},
           "eval/loss": [], "eval/step": [], 
           "eval/IDLoss": [], "eval/ICLLoss": [], "eval/OODLoss": [], "eval/CopyError": [],
           "eval/pth_score": [], "eval/ih_score": [], "train/task_loss": [], "eval/IDAcc": [], "eval/OODAcc": []
           }
    return log


def train_trigger(model, config, verbose=False):
    
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)   
    
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    
    # Specify the maximum number of epochs to generate in one pass to speedup data generation
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    
    log = _init_log()
    
    checkpoint_path = os.path.join(exp_dir, f"checkpoints")

    print(tabulate_model(model, 
                         config.seq_len, config.batch_size, config.device)) 
    
    sampler = get_sampler(config)
    random_tokens = None
    if hasattr(config.task, 'fixed') and config.task.fixed is True:
        random_tokens = sampler.q_toks

    layer = None
    if True in config.model.mlp:
        layer = config.model.mlp.index(True)
    # print(f"Layer: {layer}")

    # train_losses, eval_losses, eval_steps = [], [], []
    # last_token_losses = []
    attn_maps, probes = {}, defaultdict(list)
    # ood_losses = []
    # many_ngram_losses = {}
    # bayes_losses = []
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay) #torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.training.T_max) if config.training.scheduler is True else None
    
    test_data, test_mask = sampler.generate(mode="test")
    test_target = test_data[:, 1:].reshape(-1)
    
    if config.task.ood:
        ood_batch, ood_mask = sampler.generate(mode="ood")
        copy_batch, copy_mask = sampler.generate(mode="copy")
    else:
        ood_batch, ood_mask = None, None
        copy_batch, copy_mask = None, None
    eval_batch, eval_mask = sampler.generate(mode="eval")
    probe_batch = sampler.generate(mode="probe")
    
    # Collect ngram losses for baseline comparison
    if config.ngram > 0:
        print("Evaluating baselines...")
        ngramLearnerDict = {i:mixed_ngramLearner(config, i) for i in range(config.ngram)}

        for i, learner in ngramLearnerDict.items():
            learner.update(test_data, test_mask)
            ngram_loss = learner.loss(test_data, test_mask)
            log["baseline"][i] = ngram_loss.item()
    
    step = 0
    epochs = min(config.training.num_epochs, MAX_SIZE)
    while (config.training.num_epochs % epochs != 0) and (epochs > 0):
        epochs -= 1

    tot_iters = config.training.num_epochs // epochs
    
    wandb.init(config=config, name=exp_name, **config["wandb"])

    
    ##################
    # Start training #
    ##################
    print("Starting training...")
    for iters in range(tot_iters): #trange(tot_iters, leave=False):
        data = sampler.generate(epochs=epochs)
        sample, sample_mask = data
        #miniters = epochs // 50
        for i in range(epochs): #trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            # batch_mask = sample_mask[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            # get_attn_flag = (step < early_steps) or (step % early_steps == 0)

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0):
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            # last_token = outputs[:, -2, :].reshape(-1, config.vocab_size) # (B, V)
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            loss = criterion(outputs, targets)
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss.item()}, step=step)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            
            if config.training.get_checkpoints > 0 and step % config.training.get_checkpoints == 0:
                
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    }, os.path.join(checkpoint_path, f"model_{step}.pt"))
                    

            if (step % config.training.eval_iter == 0) or (step < min(config.training.eval_iter, 100) and step % 5 == 0):
                if verbose:
                    print(f"Step: {step}")
                log["train/step"].append(step)
                lr_val = scheduler.get_last_lr()[0] if scheduler else config.training.learning_rate
                log["train/lr"].append(lr_val)
                wandb.log({"train/lr": lr_val}, step=step)
                with torch.no_grad():
                    model.eval()
                    outputs, _ = model(test_data)
                    outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                    eval_loss = criterion(outputs, test_target)
                    log["eval/loss"].append(eval_loss.item())
                    wandb.log({"eval/loss": eval_loss.item()}, step=step)
                    log["eval/step"].append(step)

                    id_loss, icl_loss, ood_loss, copy_error = trigger_handler(model, eval_batch, eval_mask, probes, probe_batch, 
                                                                  config, sampler, random_tokens, layer, 
                                                                  ood_batch, ood_mask, copy_batch, copy_mask)
                    log["eval/IDLoss"].append(id_loss)
                    wandb.log({"eval/IDLoss": id_loss}, step=step)
                    log["eval/ICLLoss"].append(icl_loss)
                    wandb.log({"eval/ICLLoss": icl_loss}, step=step)
                    if ood_loss is not None:
                        log["eval/OODLoss"].append(ood_loss)
                        wandb.log({"eval/OODLoss": ood_loss}, step=step)
                    if copy_error is not None:
                        log["eval/CopyError"].append(copy_error)
                        wandb.log({"eval/CopyError": copy_error}, step=step)
    

    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "step": step,
        }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")

    return get_train_result(log=log, attn_maps=attn_maps, probes=probes, sampler=sampler, config=config)
                            # bayes_losses=bayes_losses, last_token_losses=last_token_losses, 
                            







def train_latent(model, config, verbose=False):

    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)  
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    
    # Specify the maximum number of epochs to generate in one pass to speedup data generation
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    
    log = _init_log()
    
    checkpoint_path = os.path.join(exp_dir, f"checkpoints")

    print(tabulate_model(model, 
                         config.seq_len, config.batch_size, config.device)) 
    
    sampler = get_sampler(config)


    # last_token_losses = []
    attn_maps, probes = {}, defaultdict(list)
    # many_ngram_losses = {}
    # bayes_losses = []
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay) #torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.training.T_max) if config.training.scheduler is True else None
    
    is_icl = "icl" in config.task.name
    
    test_data, test_info = sampler.generate(mode="test")
    test_target = test_data[:, 1:].reshape(-1)
    
    
    if config.task.ood:
        ood_batch, _ = sampler.generate(mode="ood")
        ood_target = ood_batch[:, 1:].reshape(-1)
    
    eval_batch, _ = sampler.generate(mode="eval")

    
    # Collect ngram losses for baseline comparison
    if config.ngram > 0:
        print("Evaluating baselines...")

        if config.task.name == "latent":
            many_ngramLearnersDict = {i:many_ngramLearners(config, i, sampler) for i in range(config.ngram)}
            for i, learner in many_ngramLearnersDict.items():
                many_ngram_loss = learner.loss()
                log['baseline'][i] = many_ngram_loss
        
        else:
            ngramLearnerDict = {i:ngramLearner(config, i, is_icl) for i in range(config.ngram)}

            for i, learner in ngramLearnerDict.items():
                learner.update(test_data)
                ngram_loss = learner.loss(test_data)
                log['baseline'][i] = ngram_loss.item()
    
    step = 0
    epochs = min(config.training.num_epochs, MAX_SIZE)
    while config.training.num_epochs % epochs != 0:
        epochs -= 1

    tot_iters = config.training.num_epochs // epochs

    wandb.init(config=config, name=exp_name, **config["wandb"])

    
    ##################
    # Start training #
    ##################

    print("Starting training...")
    for iters in range(tot_iters): #trange(tot_iters, leave=False):
        data = sampler.generate(epochs=epochs)
        sample, sample_info = data
        # miniters = epochs // 50
        for i in range(epochs): # trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            batch_info = sample_info[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            # get_attn_flag = (step < early_steps) or (step % early_steps == 0)

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0): # and get_attn_flag:
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            # last_token = outputs[:, -2, :].reshape(-1, config.vocab_size) # (B, V)
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            loss = criterion(outputs, targets)
            # if is_icl:
            #    last_token_losses.append(last_token_loss(last_token, batch_info).item())
            
            # with torch.no_grad():
            #    if task_handler:
            #        # collect probes etc.
            #        task_handler(model, batch, outputs, batch_info, criterion, bigram_losses, icl_losses, probes, config, sampler, random_tokens, layer)
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss.item()}, step=step)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            
            if config.training.get_checkpoints > 0 and ((step % config.training.get_checkpoints == 0) 
                                                        or (step < min(config.training.get_checkpoints, 200) and step % 5 == 0)):
                
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    }, os.path.join(checkpoint_path, f"model_{step}.pt"))


            if (step % config.training.eval_iter == 0) or (step < min(config.training.eval_iter, 100) and step % 5 == 0):
                if verbose:
                    print(f"Step: {step}")
                log["train/step"].append(step)
                lr_val = scheduler.get_last_lr()[0] if scheduler else config.training.learning_rate
                log["train/lr"].append(lr_val)
                wandb.log({"train/lr": lr_val}, step=step)
                with torch.no_grad():
                    model.eval()
                    outputs, _ = model(test_data)
                    outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                    eval_loss = criterion(outputs, test_target)
                    log["eval/loss"].append(eval_loss.item())
                    wandb.log({"eval/IDLoss": eval_loss.item()}, step=step)
                    log["eval/step"].append(step)
                    if config.task.ood:
                        ood_outputs, _ = model(ood_batch)
                        ood_outputs = ood_outputs[:, :-1, :].reshape(-1, config.vocab_size)
                        ood_loss = criterion(ood_outputs, ood_target)
                        log["eval/OODLoss"].append(ood_loss.item())
                        wandb.log({"eval/OODLoss": ood_loss.item()}, step=step)
                    
                    pth = pth_score(model, eval_batch)
                    log["eval/pth_score"].append(pth)
                    wandb.log({"eval/pth_score": pth}, step=step)
                    ih = ih_score(model, eval_batch, config.device)
                    log["eval/ih_score"].append(ih)
                    wandb.log({"eval/ih_score": ih}, step=step)
                    
    
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "step": step,
        }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")
            

    return get_train_result(log=log, config=config, sampler=sampler, attn_maps=attn_maps, probes=probes)







def train_markov(model, config, verbose=False):

    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)  
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    
    # Specify the maximum number of epochs to generate in one pass to speedup data generation
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    
    log = _init_log()
    
    checkpoint_path = os.path.join(exp_dir, f"checkpoints")

    print(tabulate_model(model, 
                         config.seq_len, config.batch_size, config.device)) 
    
    sampler = get_sampler(config)


    # last_token_losses = []
    attn_maps, probes = {}, defaultdict(list)
    # many_ngram_losses = {}
    # bayes_losses = []
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay) #torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.training.T_max) if config.training.scheduler is True else None
    
    
    test_data, test_info = sampler.generate(mode="test")
    test_info = test_info[:, 1:].reshape(-1)
    test_target = test_data[:, 1:].reshape(-1)
    
    if config.task.ood:
        ood_batch, ood_mask = sampler.generate(mode="ood")
        ood_mask = ood_mask[:, :-1].reshape(-1)
        ood_target = ood_batch[:, 1:].reshape(-1)
    
    eval_batch, eval_mask = sampler.generate(mode="eval")
    eval_mask = eval_mask[:, :-1].reshape(-1)
    
    step = 0
    epochs = min(config.training.num_epochs, MAX_SIZE)
    while config.training.num_epochs % epochs != 0:
        epochs -= 1

    tot_iters = config.training.num_epochs // epochs

    wandb.init(config=config, name=exp_name, **config["wandb"])

    
    ##################
    # Start training #
    ##################

    print("Starting training...")
    for iters in range(tot_iters): #trange(tot_iters, leave=False):
        data = sampler.generate(epochs=epochs)
        sample, sample_info = data
        # miniters = epochs // 50
        for i in range(epochs): # trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            batch_info = sample_info[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            # get_attn_flag = (step < early_steps) or (step % early_steps == 0)

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0): # and get_attn_flag:
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            # last_token = outputs[:, -2, :].reshape(-1, config.vocab_size) # (B, V)
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            batch_info = batch_info[:, :-1].reshape(-1)
            loss = criterion(outputs, targets)
            task_loss = criterion(outputs[batch_info > 0], targets[batch_info > 0])
            # if is_icl:
            #    last_token_losses.append(last_token_loss(last_token, batch_info).item())
            
            # with torch.no_grad():
            #    if task_handler:
            #        # collect probes etc.
            #        task_handler(model, batch, outputs, batch_info, criterion, bigram_losses, icl_losses, probes, config, sampler, random_tokens, layer)
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss.item()}, step=step)
            log["train/task_loss"].append(task_loss.item())
            wandb.log({"train/task_loss": task_loss.item()}, step=step)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            
            if config.training.get_checkpoints > 0 and ((step % config.training.get_checkpoints == 0) 
                                                        or (step < min(config.training.get_checkpoints, 200) and step % 5 == 0)):
                
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    }, os.path.join(checkpoint_path, f"model_{step}.pt"))


            if (step % config.training.eval_iter == 0) or (step < min(config.training.eval_iter, 100) and step % 5 == 0):
                if verbose:
                    print(f"Step: {step}")
                log["train/step"].append(step)
                lr_val = scheduler.get_last_lr()[0] if scheduler else config.training.learning_rate
                log["train/lr"].append(lr_val)
                wandb.log({"train/lr": lr_val}, step=step)
                with torch.no_grad():
                    model.eval()
                    outputs, _ = model(test_data)
                    outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                    
                    eval_loss = criterion(outputs, test_target)
                    log["eval/loss"].append(eval_loss.item())
                    wandb.log({"eval/loss": eval_loss.item()}, step=step)
                    eval_task_loss = criterion(outputs[test_info > 0], test_target[test_info > 0])
                    eval_task_acc = (outputs[test_info > 0].argmax(dim=-1) == test_target[test_info > 0]).float().mean().item()
                    log["eval/IDLoss"].append(eval_task_loss.item())
                    log["eval/IDAcc"].append(eval_task_acc)
                    wandb.log({"eval/IDLoss": eval_task_loss.item()}, step=step)
                    wandb.log({"eval/IDAcc": eval_task_acc}, step=step)
                    log["eval/step"].append(step)
                    if config.task.ood:
                        ood_outputs, _ = model(ood_batch)
                        ood_outputs = ood_outputs[:, :-1, :].reshape(-1, config.vocab_size)
                        ood_loss = criterion(ood_outputs[ood_mask > 0], ood_target[ood_mask > 0])
                        ood_acc = (ood_outputs[ood_mask > 0].argmax(dim=-1) == ood_target[ood_mask > 0]).float().mean().item()
                        log["eval/OODLoss"].append(ood_loss.item())
                        log["eval/OODAcc"].append(ood_acc)
                        wandb.log({"eval/OODLoss": ood_loss.item()}, step=step)
                        wandb.log({"eval/OODAcc": ood_acc}, step=step)
                    
                    pth = pth_score(model, eval_batch)
                    log["eval/pth_score"].append(pth)
                    wandb.log({"eval/pth_score": pth}, step=step)
                    ih = ih_score(model, eval_batch, config.device)
                    log["eval/ih_score"].append(ih)
                    wandb.log({"eval/ih_score": ih}, step=step)
                    
    
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "step": step,
        }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")
            

    return get_train_result(log=log, config=config, sampler=sampler, attn_maps=attn_maps, probes=probes)








def train_model_with_plot(model, config, show=False):
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)

    print("Experiment directory: ", exp_dir) 

    if os.path.exists(os.path.join(exp_dir, "log.json")):
        print(f"{exp_name} already completed")
        return
    

    train_results = train_model(model, config)
    
    plot_path = os.path.join(exp_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    get_loss_plots(config, train_results, folder=plot_path, show=show)
    plot_probes(train_results, config, folder=plot_path, show=True, log=False)
    plot_probes(train_results, config, folder=plot_path, show=True, log=True)
    plot_attn_scores(train_results, config, folder=plot_path, show=True, log=False)
    plot_attn_scores(train_results, config, folder=plot_path, show=True, log=True)
    if config.task.name not in ["repetition", "reversion", "fuzzy"]:
        plot_bigram_icl_risk(config, train_results, folder=plot_path, show=True)

    gif_paths = defaultdict(list)
    counts = 0
    attn_folder = os.path.join(plot_path, "attns_plot")
    os.makedirs(attn_folder, exist_ok=True)

    for layer in range(config.model.num_layers):
        gif_paths[layer].append(get_attn_gif(layer, "all", train_results, config, out_folder=attn_folder))
        counts += 1
    if show:
        if counts < 3:
            gifs = [item for sublist in gif_paths.values() for item in sublist]
            htmls = [f"<td><img src='{gif}' width='500'></td>" for gif in gifs]
            html_code = "<table><tr>" + "".join(htmls) + "</tr></table>"
            display(HTML(html_code))
        else:
            for layer, paths in gif_paths.items():
                gifs = [path for path in paths]
                htmls = [f"<td><img src='{gif}' width='500'></td>" for gif in gifs]
                html_code = "<table><tr>" + "".join(htmls) + "</tr></table>"
                display(HTML(html_code))

    if show:
        trunc = max(config.seq_len - 64, 0) # show the last 64 tokens
        get_head_view(model, train_results, config, trunc=trunc, action="view")
    
    
    html = get_head_view(model, train_results, config, trunc=0, action="return")
    
    html_file_name = os.path.join(attn_folder, "attn_view.html")
    with open(html_file_name, "w", encoding="utf-8") as file:
        file.write(html)
    

    last_key = sorted(list(train_results["attn_maps"].keys()))[-1]
    last_attn = train_results["attn_maps"][last_key]
    last_attn["steps"] = last_key
    train_results["attn_maps"] = last_attn

    result_file_name = os.path.join(exp_dir, "sampler.pkl")
    with open(result_file_name, "wb") as file:
        pickle.dump(train_results["sampler"], file)
    
    return train_results
    