import torch
from torch.utils.data import Dataset
from collections import defaultdict
from util import *
from tasks.causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm
from tasks.markov import *
# from models.ngram_latent import *
from tasks.markov_latent import *
import datetime
import os


def get_bayes_loss(bayes_prob, prob):
    return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()


def last_token_loss(logits, probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()



def bietti_bb_handler(model, eval_batch, eval_mask, probes, probe_batch, config, sampler=None, random_tokens=None, layer=1, ood_batch=None, ood_mask=None):
    eval_outputs, _ = model(eval_batch)
    eval_outputs = eval_outputs[:, :-1, :].reshape(-1, config.vocab_size)
    ood_loss = None
    
    if config.task_name == "frm":
        bigram_loss, icl_loss = get_bigram_icl_loss(eval_outputs, eval_batch[:, 1:].reshape(-1), eval_mask)
    else:
        bigram_loss, icl_loss = get_bigram_icl_error(eval_outputs, eval_batch[:, 1:].reshape(-1), eval_mask)
    
    if (ood_batch is not None) and (ood_mask is not None):
        # Compute the ICL loss for the out-of-distribution samples
        ood_outputs, _ = model(ood_batch)
        ood_outputs = ood_outputs[:, :-1, :].reshape(-1, config.vocab_size)
        ood_loss = get_icl_error(ood_outputs, ood_batch[:,1:].reshape(-1), ood_mask)
        

    
    probe_keys = ["wk0", "wk1", "wo1"]
    for pkey in probe_keys:
        probes[pkey].append(memory_recall_probe(model, pkey, config, eval_batch[:1]))
    # if layer == 1:
    #    probes['ffr'].append(feedforward_residual_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=None))
    # probes['outr'].append(output_residual_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))
    if layer is not None:    
        kl_ffn_res, kl_ffn, kl_res = high_order_memory_probe(sampler, model, probe_batch)
        probes['ff+res'].append(kl_ffn_res)
        probes['ff'].append(kl_ffn)
        probes['res'].append(kl_res)
        # if config.task_name == "bietti":
        # probes['ff_emb'].append(ff_emb_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens, layer=layer))
        probes['ff_icl'].append(ff_icl_probe(config.vocab_size, model, config.device))
        probes['ff_mem_unif'].append(ff_memory_probe(config.vocab_size, model, sampler.trans_mat, config.device, weight="uniform"))
        probes['ff_mem_true'].append(ff_memory_probe(config.vocab_size, model, sampler.trans_mat, config.device, weight="true"))
        probes['combined_icl'].append(combined_icl_probe(config.vocab_size, model, config.device))
    # probes['emb'].append(emb_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))
    probes['out'].append(output_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))
    probes['attn'].append(attn_icl_probe(config.vocab_size, model, config.device))

    return bigram_loss, icl_loss, ood_loss
        
    '''
    elif config.task_name == "frm":
        if layer == 1:
            probes['ffr'].append(feedforward_residual_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=None))
        if layer is not None:
            probes['ff'].append(feedforward_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens, layer=layer))
            probes['ff_emb'].append(ff_emb_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens, layer=layer))
        probes['emb'].append(emb_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))
        probes['out'].append(output_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))
        probes['outr'].append(output_residual_probe(config.vocab_size, model, sampler.trans_mat, config.device, random_tokens=random_tokens))'
    '''
    
    
        

def get_sampler(sampler_config):
    task_samplers = {
        "markov": MarkovSampler,
        "bietti": BiettiTask,
        # "bb": BBTask,
        "dag": InContextDAGTorch,
        "tree": InContextTreeTorch,
        "icl-mc": ICLMarkovSampler,
        "frm": FRMarkovSampler,
        "latent": LatentMarkov,
    }
    if sampler_config.task_name in task_samplers:
        return task_samplers[sampler_config.task_name](sampler_config)
    raise NotImplementedError(f"Task '{sampler_config.task_name}' not implemented yet.")

# Compute bigram ICL loss
def get_bigram_icl_error(outputs, targets, out_mask, burn_in=3):
    criterion = nn.CrossEntropyLoss()
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    bigram_loss = criterion(outputs[~icl_mask_flat], targets[~icl_mask_flat])
    preds = torch.argmax(outputs, dim=-1)
    icl_error = (preds[icl_mask_flat][burn_in:] != targets[icl_mask_flat][burn_in:]).sum()
    total = icl_mask_flat.sum()
    icl_loss = icl_error.float() / total.float()
    return bigram_loss.item(), icl_loss.item()

def get_icl_error(outputs, targets, out_mask, burn_in=3):
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    preds = torch.argmax(outputs, dim=-1)
    icl_error = (preds[icl_mask_flat][burn_in:] != targets[icl_mask_flat][burn_in:]).sum()
    total = icl_mask_flat.sum()
    icl_loss = icl_error.float() / total.float()
    return icl_loss.item()


def get_bigram_icl_loss(outputs, targets, out_mask):
    # shifted_mask = torch.roll(out_mask, shifts=1, dims=1)
    # shifted_mask[:, 0] = 0
    # icl_mask_flat = (shifted_mask>0)[:,:-1].reshape(-1)
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    criterion = nn.CrossEntropyLoss()
    bigram_loss = criterion(outputs[~icl_mask_flat], targets[~icl_mask_flat])
    icl_loss = criterion(outputs[icl_mask_flat], targets[icl_mask_flat])
    return bigram_loss.item(), icl_loss.item()


def get_train_result(**kwargs):
    return kwargs










# Not in use      
class SimulatedDataset(Dataset):
    def __init__(self, sampler, num_samples):
        self.num_samples = num_samples
        self.sampler = sampler

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sample on-the-fly
        return self.sampler.generate()

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