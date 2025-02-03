from tqdm.notebook import trange
import torch
import torch.nn as nn  
from markov import *
from collections import defaultdict
from causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *
import copy

from torch.utils.data import Dataset, DataLoader

### TODO: Use a data_loader to preload part of the simulated data

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


def train_generic(model, config, sampler_config, task_handler=None):
    sampler = get_sampler(sampler_config)
    train_losses, eval_losses, eval_steps = [], [], []
    attn_maps, ngramLosses, bigram_losses, icl_losses, probes = {}, defaultdict(list), [], [], defaultdict(list)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max) if config.scheduler is not None else None
    is_icl = "icl" in sampler_config.task_name
    is_masked = sampler_config.task_name in ["bietti", "bb"]
    test_data, test_mask = sampler.generate(mode="test") if is_masked else (sampler.generate(mode="test"), None)
    test_target = test_data[:, 1:].reshape(-1).to(config.device)
    
    if config.ngram:
        ngramLearnerDict = {i:ngramLearner(config, sampler_config, i, is_icl) for i in range(config.max_gram)}
    
    for epoch in trange(config.num_epochs):
        model.train()
        batch, out_mask = sampler.generate() if is_masked else (sampler.generate(), None)
        optimizer.zero_grad()
        targets = batch[:, 1:].reshape(-1).to(config.device)

        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(batch.to(config.device))
            attn_maps[epoch] = copy.deepcopy(attn)
        else:
            outputs, _ = model(batch.to(config.device))
        
        outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
        loss = criterion(outputs, targets)

        if config.ngram:
            for i, learner in ngramLearnerDict.items():
                learner.update(batch)
                ngram_loss = learner.loss(batch)
                ngramLosses[i].append(ngram_loss.item())
        
        with torch.no_grad():
            if task_handler:
                task_handler(epoch, model, batch, outputs, out_mask, criterion, bigram_losses, icl_losses, probes, sampler_config)
        
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data.to(config.device))
                outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_target)
                eval_losses.append(eval_loss.item())
                eval_steps.append(epoch)

    return get_train_result(train_losses=train_losses, eval_losses=eval_losses, eval_steps=eval_steps,
                            attn_maps=attn_maps, ngramLosses=ngramLosses, bigram_losses=bigram_losses,
                            icl_losses=icl_losses, probes=probes)

def bietti_bb_handler(epoch, model, batch, outputs, out_mask, criterion, bigram_losses, icl_losses, probes, sampler_config):
    bigram_loss, icl_loss = get_bigram_icl_loss(outputs, batch[:, 1:].reshape(-1), out_mask, criterion)
    bigram_losses.append(bigram_loss)
    icl_losses.append(icl_loss)

    if sampler_config.task_name == "bietti":
        probe_keys = ["wk0", "wk1", "wo1"]
        for pkey in probe_keys:
            probes[pkey].append(memory_recall_probe(sampler_config.vocab_size, model, pkey, sampler_config.seq_len, sampler_config.device))
        probes['ff'].append(feedforward_probe(sampler_config.vocab_size, model, sampler_config.trans_mat, sampler_config.device))

def train_causal(model, config, sampler_config):
    sampler = get_sampler(sampler_config)
    train_losses, eval_losses, eval_steps = [], [], []
    bayes_losses = []
    attn_maps = {}
    test_data, test_prob = sampler.generate(mode="test")

    if config.scheduler is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)  


    def criterion(logits, probs):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1).mean()
    
    def get_bayes_loss(bayes_prob, prob):
        return -torch.sum(prob * torch.log(bayes_prob), dim=-1).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in trange(config.num_epochs):
        model.train()
        x, prob = sampler.generate()
        with torch.no_grad():
            bayes_prob = sampler.bayes(x)
            bayes_loss = get_bayes_loss(bayes_prob, prob)
            bayes_losses.append(bayes_loss.item())

        optimizer.zero_grad()
        
        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(x.to(config.device))
            attn_maps[epoch] = copy.deepcopy(attn)
        else:
            outputs, _ = model(x.to(config.device))

        outputs = outputs[:,-1,:].reshape(-1, config.vocab_size)
        loss = criterion(outputs, prob)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if config.scheduler is not None: scheduler.step()
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data.to(config.device))
                outputs = outputs[:,-1,:].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_prob)
                eval_losses.append(eval_loss.item())
                eval_steps.append(epoch)
    
    return get_train_result(train_losses=train_losses, 
                            eval_losses=eval_losses, 
                            eval_steps=eval_steps, 
                            attn_maps=attn_maps, sampler=sampler, bayes_losses=bayes_losses)

# Train model based on task
def train_model(model, config, sampler_config):
    if sampler_config.task_name in ["dag", "tree"]:
        return train_causal(model, config, sampler_config)
    
    task_handlers = {
        "bietti": bietti_bb_handler,
        "bb": bietti_bb_handler
    }
    return train_generic(model, config, sampler_config, task_handlers.get(sampler_config.task_name, None))