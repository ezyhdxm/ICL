import torch
import torch.nn as nn  
# from models.ngram_trigger import *
from collections import defaultdict
# from tasks.causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from absl import logging
import pickle
import os
import wandb
import json

from icl.tasks.markov import *
from icl.models.ngram_latent import *
from icl.tasks import *

# from torch.utils.data import DataLoader
from .train_utils import get_attn_base, get_train_result, tabulate_model
from icl.figures.plot import get_loss_plots
# from IPython.display import display, HTML
# from icl.figures.head_view import get_head_view

from .basic import get_hash



def _init_log() -> dict:
    """
    Initialize log dictionary for evaluation metrics.
    """
    log = {"train/step": [], "train/lr": [], "train/loss": [], 
           "baseline": {},
           "eval/loss": [], "eval/step": [], 
           "eval/IDLoss": [], "eval/ICLLoss": [], "eval/OODLoss": [], "eval/CopyError": [],
           "eval/pth_score": [], "eval/ih_score": [], "eval/IDAcc": [], "eval/OODAcc": [],
           "eval/LengthLoss": [], "eval/LengthAcc": []}
    return log

class BaseTrainer: 
    def __init__(self, config):
        self.config = config
        self.exp_name = f"train_{get_hash(config)}"
        self.exp_dir = os.path.join(config.work_dir, self.exp_name)  
        logging.info(f"Train Experiment\nNAME: {self.exp_name}\nCONFIG:\n{config}")
        self.MAX_SIZE = 1024
        self.log_path = os.path.join(self.exp_dir, "log.json")
        if os.path.exists(self.log_path):
            print(f"{self.exp_name} already completed")
            return
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f: f.write(config.to_json())
        self.log = _init_log()
        self.checkpoint_path = os.path.join(self.exp_dir, f"checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.attn_maps, self.probes = {}, defaultdict(list)
        self.criterion = nn.CrossEntropyLoss()
        self.step = 0

    def info_process(self, info):
        return None
    
    def get_task_loss(self, outputs, targets, info):
        return self.criterion(outputs, targets).item()
    
    def log_eval(self, model, data, infos):
        step = self.step
        with torch.no_grad():
            model.eval()
            outputs = model(data["test"])
            target = data["test"][:, 1:].reshape(-1)
            outputs = outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
            
            eval_loss = self.criterion(outputs, target)
            self.log["eval/loss"].append(eval_loss.item())
            wandb.log({"eval/loss": eval_loss.item()}, step=step)
            eval_task_loss = self.get_task_loss(outputs, target, infos["test"]) 
            self.log["eval/IDLoss"].append(eval_task_loss)
            wandb.log({"eval/IDLoss": eval_task_loss}, step=step)
            self.log["eval/step"].append(step)
            if self.config.task.ood:
                ood_outputs = model(data["ood"])
                ood_outputs = ood_outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                ood_target = data["ood"][:, 1:].reshape(-1)
                ood_loss = self.get_task_loss(ood_outputs, ood_target, infos["ood"])
                self.log["eval/OODLoss"].append(ood_loss)
                wandb.log({"eval/OODLoss": ood_loss}, step=step)

            if "length_ood" in self.config.task and self.config.task.length_ood:
                ood_outputs = model(data["length_ood"])
                ood_outputs = ood_outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                length_target = data["length_ood"][:, 1:].reshape(-1)
                ood_loss = self.get_task_loss(ood_outputs, length_target, infos["length_ood"])
                self.log["eval/LengthLoss"].append(ood_loss)
                wandb.log({"eval/LengthLoss": ood_loss}, step=step)

    def save_checkpoint(self, model, optimizer, is_final=False):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        step = self.step
        if is_final:
            torch.save({
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "step": step,
                }, os.path.join(self.checkpoint_path, f"model_final_{step}.pt"))
        else:
            torch.save({
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "step": step,
                }, os.path.join(self.checkpoint_path, f"model_{step}.pt"))

    def train(self, model, verbose=False):
        sampler = get_sampler(self.config)
        if verbose: print(tabulate_model(model, self.config.seq_len, self.config.batch_size, self.config.device))

        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=self.config.training.learning_rate, 
                                      weight_decay=self.config.training.weight_decay)  # torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.training.T_max) if self.config.training.scheduler is True else None

        data = {"test": None, "ood": None, "length_ood": None}
        infos = {"test": None, "ood": None, "length_ood": None}
        data["test"], infos["test"] = sampler.generate(mode="test")
        infos["test"] = self.info_process(infos["test"])

        if self.config.task.ood:
            data["ood"], infos["ood"] = sampler.generate(mode="ood")
            infos["ood"] = self.info_process(infos["ood"])

        if "length_ood" in self.config.task and self.config.task.length_ood:
            data["length_ood"], infos["length_ood"] = sampler.generate(mode="length_ood")
            infos["length_ood"] = self.info_process(infos["length_ood"])

        # eval_batch, eval_info = sampler.generate(mode="eval")
        # eval_info = self.info_process(eval_info)

        epochs = min(self.config.training.num_epochs, self.MAX_SIZE)
        while self.config.training.num_epochs % epochs != 0: epochs -= 1

        tot_iters = self.config.training.num_epochs // epochs

        wandb.init(config=self.config, name=self.exp_name, **self.config["wandb"])

        if verbose: print("Starting training...")
        for iters in range(tot_iters): 
            train_data = sampler.generate(epochs=epochs)
            sample, sample_info = train_data
            for i in range(epochs): 
                self.step += 1
                model.train()
                batch, batch_info = sample[i], sample_info[i]
                optimizer.zero_grad()
                targets = batch[:, 1:].reshape(-1)

                if (self.config.training.get_attn) > 0 and (self.step % self.config.training.get_attn == 0): 
                    self.attn_maps[self.step] = get_attn_base(model, batch)

                outputs = model(batch)
                
                outputs = outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                loss = self.criterion(outputs, targets)

                self.log["train/loss"].append(loss.item())
                wandb.log({"train/loss": loss.item()}, step=self.step)
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()

                if self.config.training.get_checkpoints > 0 and ((self.step % self.config.training.get_checkpoints == 0) 
                                                            or (self.step < min(self.config.training.get_checkpoints, 200) and self.step % 5 == 0)):
                    self.save_checkpoint(model, optimizer, is_final=False)


                if (self.step % self.config.training.eval_iter == 0) or (self.step < min(self.config.training.eval_iter, 100) and self.step % 5 == 0):
                    if verbose: print(f"Step: {self.step}")
                    self.log["train/step"].append(self.step)
                    lr_val = scheduler.get_last_lr()[0] if scheduler else self.config.training.learning_rate
                    self.log["train/lr"].append(lr_val)
                    wandb.log({"train/lr": lr_val}, step=self.step)
                    self.log_eval(model, data, infos)

        self.save_checkpoint(model, optimizer, is_final=True)
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

        if verbose:
            print("Training complete.")

        return get_train_result(log=self.log, config=self.config, sampler=sampler, attn_maps=self.attn_maps, probes=self.probes)

class MarkovTrainer(BaseTrainer):
    def info_process(self, info):
        length = info.max(dim=1).values // 2
        mask = info > length[:, None]
        mask = mask[:, 1:].reshape(-1)
        return mask
    
    def get_task_loss(self, outputs, targets, info):
        return self.criterion(outputs[info], targets[info]).item()

    def log_eval(self, model, data, infos):
        step = self.step
        with torch.no_grad():
            model.eval()
            outputs = model(data["test"])
            target = data["test"][:, 1:].reshape(-1)
            outputs = outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
            
            eval_loss = self.criterion(outputs, target)
            self.log["eval/loss"].append(eval_loss.item())
            wandb.log({"eval/loss": eval_loss.item()}, step=step)
            eval_task_loss = self.get_task_loss(outputs, target, infos["test"]) 
            eval_task_acc = (outputs[infos["test"]].argmax(dim=-1) == target[infos["test"]]).float().mean().item()
            self.log["eval/IDLoss"].append(eval_task_loss)
            self.log["eval/IDAcc"].append(eval_task_acc)
            wandb.log({"eval/IDLoss": eval_task_loss}, step=step)
            wandb.log({"eval/IDAcc": eval_task_acc}, step=step)
            self.log["eval/step"].append(step)
            if self.config.task.ood:
                ood_outputs = model(data["ood"])
                ood_outputs = ood_outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                ood_target = data["ood"][:, 1:].reshape(-1)
                ood_loss = self.get_task_loss(ood_outputs, ood_target, infos["ood"])
                # ood_acc = (ood_outputs[infos["ood"] > ood_length].argmax(dim=-1) == ood_target[infos["ood"] > ood_length]).float().mean().item()
                ood_acc = (ood_outputs[infos["ood"]].argmax(dim=-1) == ood_target[infos["ood"]]).float().mean().item()
                self.log["eval/OODLoss"].append(ood_loss)
                self.log["eval/OODAcc"].append(ood_acc)
                wandb.log({"eval/OODLoss": ood_loss}, step=step)
                wandb.log({"eval/OODAcc": ood_acc}, step=step)

            if "length_ood" in self.config.task and self.config.task.length_ood:
                ood_outputs = model(data["length_ood"])
                ood_outputs = ood_outputs[:, :-1, :].reshape(-1, self.config.vocab_size)
                length_target = data["length_ood"][:, 1:].reshape(-1)
                ood_loss = self.get_task_loss(ood_outputs, length_target, infos["length_ood"])
                ood_acc = (ood_outputs[infos["length_ood"]].argmax(dim=-1) == length_target[infos["length_ood"]]).float().mean().item()
                self.log["eval/LengthLoss"].append(ood_loss)
                self.log["eval/LengthAcc"].append(ood_acc)
                wandb.log({"eval/LengthLoss": ood_loss}, step=step)
                wandb.log({"eval/LengthAcc": ood_acc}, step=step)






def get_sampler(config):
    task_samplers = {
        "markov": MarkovSampler,
        "icl-mc": ICLMarkovSampler,
        "frm": FRMarkovSampler,
        "latent": LatentMarkov,
        "repetition": RepetitionTask,
        "reversion": ReversedTask,
        "fuzzy": FuzzyCopyTask,
        "dyck": DyckPathTask,
        "coin": CoinTask,
    }
    if config.task.name in task_samplers: return task_samplers[config.task.name](config)
    raise NotImplementedError(f"Task '{config.task.name}' not implemented yet.")





# Train model based on task
def train_model(config):
    if config.task.name in ["frm", "bietti"]:
        raise NotImplementedError(f"Task '{config.task.name}' not implemented yet.")
        #return train_trigger(model, config)
    elif config.task.name in ["icl-mc"]:
        raise NotImplementedError(f"Task '{config.task.name}' not implemented yet.")
        # return train_latent(model, config)
    elif config.task.name in ["repetition", "reversion", "fuzzy", "dyck"]:
        return MarkovTrainer(config)
    else:
        return BaseTrainer(config)






def train_model_with_plot(model, config, show=False):
    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)

    print("Experiment directory: ", exp_dir) 

    if os.path.exists(os.path.join(exp_dir, "log.json")):
        print(f"{exp_name} already completed")
        return
    

    trainer = train_model(config)
    train_results = trainer.train(model, verbose=False)
    
    plot_path = os.path.join(exp_dir, "plots")
    os.makedirs(plot_path, exist_ok=True)

    get_loss_plots(config, train_results, folder=plot_path, show=show)
    #plot_attn_scores(train_results, config, folder=plot_path, show=True, log=False)
    #plot_attn_scores(train_results, config, folder=plot_path, show=True, log=True)

    '''
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
        get_head_view(model, config, train_results=train_results, trunc=trunc, action="view")

    html = get_head_view(model, config, train_results=train_results, trunc=0, action="return")

    html_file_name = os.path.join(attn_folder, "attn_view.html")
    with open(html_file_name, "w", encoding="utf-8") as file:
        file.write(html)
    '''

    last_key = sorted(list(train_results["attn_maps"].keys()))[-1]
    last_attn = train_results["attn_maps"][last_key]
    last_attn["steps"] = last_key
    train_results["attn_maps"] = last_attn

    result_file_name = os.path.join(exp_dir, "sampler.pkl")
    with open(result_file_name, "wb") as file:
        pickle.dump(train_results["sampler"], file)
    
    return train_results
    