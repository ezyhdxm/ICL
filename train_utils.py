import torch
from torch.utils.data import Dataset
from probe_util import *
from tasks.causal_graph import *
from tasks.markov import *
# from models.ngram_latent import *
from tasks.markov_latent import *
import datetime
import os
from torchinfo import summary


def get_bayes_loss(bayes_prob, prob):
    return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()


def last_token_loss(logits, probs):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()



def trigger_handler(model, eval_batch, eval_mask, probes, probe_batch, config, sampler=None, random_tokens=None, layer=1, 
                    ood_batch=None, ood_mask=None, copy_batch=None, copy_mask=None):
    eval_outputs, _ = model(eval_batch)
    eval_outputs = eval_outputs[:, :-1, :].reshape(-1, config.vocab_size)
    ood_loss = None
    
    if config.task.name == "frm":
        id_loss, icl_loss = get_bigram_icl_loss(eval_outputs, eval_batch[:, 1:].reshape(-1), eval_mask)
    else:
        id_loss, icl_loss = get_bigram_icl_error(eval_outputs, eval_batch[:, 1:].reshape(-1), eval_mask)
    
    if (ood_batch is not None) and (ood_mask is not None):
        # Compute the ICL loss for the out-of-distribution samples
        ood_outputs, _ = model(ood_batch)
        ood_outputs = ood_outputs[:, :-1, :].reshape(-1, config.vocab_size)
        _, ood_loss = get_bigram_icl_loss(ood_outputs, ood_batch[:,1:].reshape(-1), ood_mask)
    
    if (copy_batch is not None) and (copy_mask is not None):
        # Compute the ICL loss for the copy samples
        copy_outputs, _ = model(copy_batch)
        copy_outputs = copy_outputs[:, :-1, :].reshape(-1, config.vocab_size)
        _, copy_error = get_bigram_icl_error(copy_outputs, copy_batch[:,1:].reshape(-1), copy_mask)
        

    
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

    return id_loss, icl_loss, ood_loss, copy_error
        
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
    
    
        



# Compute bigram ICL loss
def get_bigram_icl_error(outputs, targets, out_mask):
    criterion = nn.CrossEntropyLoss()
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    id_loss = criterion(outputs[~icl_mask_flat], targets[~icl_mask_flat])
    preds = torch.argmax(outputs, dim=-1)
    icl_error = (preds[icl_mask_flat] != targets[icl_mask_flat]).sum()
    total = icl_mask_flat.sum()
    icl_loss = icl_error.float() / total.float()
    return id_loss.item(), icl_loss.item()

def get_icl_error(outputs, targets, out_mask):
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    preds = torch.argmax(outputs, dim=-1)
    icl_error = (preds[icl_mask_flat] != targets[icl_mask_flat]).sum()
    total = icl_mask_flat.sum()
    icl_loss = icl_error.float() / total.float()
    return icl_loss.item()


def get_bigram_icl_loss(outputs, targets, out_mask):
    # shifted_mask = torch.roll(out_mask, shifts=1, dims=1)
    # shifted_mask[:, 0] = 0
    # icl_mask_flat = (shifted_mask>0)[:,:-1].reshape(-1)
    icl_mask_flat = (out_mask==1)[:,:-1].reshape(-1)
    criterion = nn.CrossEntropyLoss()
    id_loss = criterion(outputs[~icl_mask_flat], targets[~icl_mask_flat])
    icl_loss = criterion(outputs[icl_mask_flat], targets[icl_mask_flat])
    return id_loss.item(), icl_loss.item()


def get_train_result(**kwargs):
    return kwargs



    

def tabulate_model(model: torch.nn.Module, seq_len: int, batch_size: int, device: str) -> str:
    dummy_data = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    try:
        info = summary(model, 
                       input_data=dummy_data, 
                       depth=3, 
                       col_names=["input_size", "output_size", "num_params"])
        return str(info)
    except Exception as e:
        return f"Could not tabulate model: {e}"
    


def pth_score(model, batch):
    embs = model.embed(batch)
    _, attn = model.layers[0].MHA(embs, True)
    attn = attn.squeeze(1)
    return attn.mean(dim=0).diagonal(offset=-1).mean().item()

def ih_score(model, batch, device):
    embs = model.embed(batch)
    hiddens, _ = model.layers[0](embs)
    _, attns = model.layers[1](hiddens, True)
    attns = attns.squeeze(1)
    B, T = batch.shape
    
    # Compare all pairs: (B, T, T), batch[b, i] == batch[b, t]
    matches = (batch.unsqueeze(2) == batch.unsqueeze(1))  # (B, T, T)
    
    # Mask out positions where i >= t (i.e., keep only i < t)
    tril_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)  # (T, T)
    valid_matches = matches & tril_mask  # broadcasted to (B, T, T)
    b_indices, t_indices, i_indices = valid_matches.nonzero(as_tuple=True)
    grouped_sums = torch.zeros((B, T), device=device)  # (B, T)

    # Accumulate values at corresponding (b, t) positions
    grouped_sums.index_put_((b_indices, t_indices), attns[b_indices, t_indices, i_indices+1], accumulate=True)
    return grouped_sums.mean(dim=0)[1:].mean().item()



