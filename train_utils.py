import torch
from torch.utils.data import Dataset
from collections import defaultdict
from util import *
from causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm
from markov import *
import datetime
import os


def get_bayes_loss(bayes_prob, prob):
    return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()


def last_token_loss(logits, probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()

def bietti_bb_handler(model, batch, outputs, out_mask, criterion, bigram_losses, icl_losses, probes, sampler_config):
    bigram_loss, icl_loss = get_bigram_icl_loss(outputs, batch[:, 1:].reshape(-1), out_mask, criterion)
    bigram_losses.append(bigram_loss)
    icl_losses.append(icl_loss)

    if sampler_config.task_name == "bietti":
        probe_keys = ["wk0", "wk1", "wo1"]
        for pkey in probe_keys:
            probes[pkey].append(memory_recall_probe(sampler_config.vocab_size, model, pkey, sampler_config.seq_len, sampler_config.device))
        probes['ff'].append(feedforward_probe(sampler_config.vocab_size, model, sampler_config.trans_mat, sampler_config.device))

class SimulatedDataset(Dataset):
    def __init__(self, sampler, num_samples):
        self.num_samples = num_samples
        self.sampler = sampler

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sample on-the-fly
        return self.sampler.generate()


def get_sampler(sampler_config):
    task_samplers = {
        "markov": MarkovSampler,
        "bietti": BiettiTask,
        "bb": BBTask,
        "dag": InContextDAGTorch,
        "tree": InContextTreeTorch,
        "icl-mc": ICLMarkovSampler
    }
    if sampler_config.task_name in task_samplers:
        return task_samplers[sampler_config.task_name](sampler_config)
    raise NotImplementedError(f"Task '{sampler_config.task_name}' not implemented yet.")

# Compute bigram ICL loss
def get_bigram_icl_loss(outputs, targets, out_mask, criterion):
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    bigram_loss = criterion(outputs[~icl_mask_flat], targets[~icl_mask_flat])
    preds = torch.argmax(outputs, dim=-1)
    icl_error = (preds[icl_mask_flat] != targets[icl_mask_flat]).sum()
    total = icl_mask_flat.sum()
    icl_loss = icl_error.float() / total.float()
    return bigram_loss.item(), icl_loss.item()

def get_train_result(**kwargs):
    return kwargs


def save_model(model, config, train_results):
    os.makedirs("models", exist_ok=True)
    model_name = f"{config.task_name}_{config.num_heads}H_{config.num_layers}L"
    if any(config.mlp):
        model_name += "_MLP"
    if any(config.activation):
        model_name += "_ReLU"
    model_name += f"_{config.pos_enc}"
    model_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_results': train_results
    }, f"models/{model_name}.pt")
    print(f"Model saved as {model_name}.pt")