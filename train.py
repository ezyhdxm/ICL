from tqdm import trange
import torch
import torch.nn as nn  
from markov import *
from collections import defaultdict
from causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *

def get_sampler(sampler_config, task_name):
    if task_name == "markov":
        sampler = MarkovSampler(sampler_config)
    elif task_name == "bietti":
        sampler = BiettiTask(sampler_config)
    elif task_name == "bb":
        sampler = BBTask(sampler_config)
    elif task_name == "dag":
        sampler = InContextDAGTorch(sampler_config)
    elif task_name == "tree":
        sampler = InContextTreeTorch(sampler_config)
    else:
        raise NotImplementedError("Task not implemented yet")
    return sampler

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

def train_causal(model, config, sampler_config, task_name):
    sampler = get_sampler(sampler_config, task_name)
    train_losses, eval_losses, eval_steps = [], [], []
    attn_maps = {}
    test_data, test_prob = sampler.generate(mode="test")

    if config.scheduler is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)  


    def criterion(logits, probs):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in trange(config.num_epochs):
        model.train()
        x, prob = sampler.generate()
        optimizer.zero_grad()
        
        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(x.to(config.device))
            attn_maps[epoch] = attn.clone()
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
                            attn_maps=attn_maps, sampler=sampler)

def train_markov(model, config, sampler_config):
    sampler = MarkovSampler(sampler_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)  

    train_losses, eval_losses, eval_steps = [], [], []
    attn_maps = {}
    ngramLearnerDict, ngramLosses = {}, {}
    test_data = sampler.generate(mode="test")
    test_y = test_data[:,1:].reshape(-1).to(config.device)

    if config.ngram:
        ngramLearnerDict = {i:ngramLearner(config.vocab_size, i) for i in range(config.max_gram)}
        ngramLosses = defaultdict(list)
    
    for epoch in trange(config.num_epochs):
        model.train()
        batch = sampler.generate() # bottleneck
        optimizer.zero_grad()
        targets = batch[:,1:].reshape(-1).to(config.device)

        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(batch.to(config.device))
            attn_maps[epoch] = attn.clone()
        else:
            outputs, _ = model(batch.to(config.device))

        outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)

        for i, learner in ngramLearnerDict.items():
            ngram_loss = learner.loss(batch)
            ngramLosses[i].append(ngram_loss)
            learner.update(batch)
        
        loss = criterion(outputs, targets)
        train_losses.append(loss.item())
            
        loss.backward()
        optimizer.step()
        
        if config.scheduler is not None: scheduler.step()
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data.to(config.device))
                outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_y)
                eval_losses.append(eval_loss.item())
                eval_steps.append(epoch)
    return get_train_result(train_losses=train_losses, 
                            eval_losses=eval_losses, 
                            eval_steps=eval_steps, 
                            attn_maps=attn_maps, 
                            ngramLosses=ngramLosses)

def train_bietti(model, config, sampler_config):
    sampler = get_sampler(sampler_config, "bietti")
    probes = defaultdict(list)
    probe_keys = ["wk0", "wk1", "wo1"]
    train_losses, eval_losses, eval_steps = [], [], []
    bigram_losses, icl_losses = [], []
    attn_maps = {}
    ngramLearnerDict, ngramLosses = {}, {}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    if config.scheduler is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)  
    
    test_data, test_mask = sampler.generate(mode="test")

    test_y = test_data[:,1:].reshape(-1).to(config.device)
    
    if config.ngram:
        ngramLearnerDict = {i:ngramLearner(config.vocab_size, i) for i in range(config.max_gram)}
        ngramLosses = defaultdict(list)
    
    for epoch in trange(config.num_epochs):
        model.train()
        batch, out_mask = sampler.generate()
        optimizer.zero_grad()
        
        targets = batch[:,1:].reshape(-1).to(config.device)
        
        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(batch.to(config.device))
            attn_maps[epoch] = attn.clone()
        else:
            outputs, _ = model(batch.to(config.device))

        outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)

        for i, learner in ngramLearnerDict.items():
            ngram_loss = learner.loss(batch)
            ngramLosses[i].append(ngram_loss)
            learner.update(batch)
        
        loss = criterion(outputs, targets)

        with torch.no_grad():
            bigram_loss, icl_loss = get_bigram_icl_loss(outputs, targets, out_mask, criterion)
            bigram_losses.append(bigram_loss)
            icl_losses.append(icl_loss)
                
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if config.scheduler is not None: scheduler.step()
        
        for pkey in probe_keys:
            probes[pkey].append(memory_recall_probe(config.vocab_size, 
                                                    model, 
                                                    pkey, 
                                                    config.seq_len, config.device))
        probes['ff'].append(feedforward_probe(config.vocab_size, 
                                              model, 
                                              sampler_config.trans_mat, config.device))
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data.to(config.device))
                outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_y)
                eval_losses.append(eval_loss.item())
                eval_steps.append(epoch)
    
    return get_train_result(train_losses=train_losses, 
                            eval_losses=eval_losses, 
                            eval_steps=eval_steps, 
                            attn_maps=attn_maps, 
                            ngramLosses=ngramLosses,
                            probes=probes,
                            bigram_losses=bigram_losses,
                            icl_losses=icl_losses,
                            ngramLosses=ngramLosses)


def train_bb(model, config, sampler_config):
    sampler = get_sampler(sampler_config, "bb")
    train_losses, eval_losses, eval_steps = [], [], []
    bigram_losses, icl_losses = [], []
    attn_maps = {}
    ngramLearnerDict, ngramLosses = {}, {}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    if config.scheduler is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)  
    
    test_data, test_mask = sampler.generate(mode="test")

    test_y = test_data[:,1:].reshape(-1).to(config.device)
    
    if config.ngram:
        ngramLearnerDict = {i:ngramLearner(config.vocab_size, i) for i in range(config.max_gram)}
        ngramLosses = defaultdict(list)
    
    for epoch in trange(config.num_epochs):
        model.train()
        batch, out_mask = sampler.generate()
        optimizer.zero_grad()
        
        targets = batch[:,1:].reshape(-1).to(config.device)
        
        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(batch.to(config.device))
            attn_maps[epoch] = attn.clone()
        else:
            outputs, _ = model(batch.to(config.device))

        outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)

        for i, learner in ngramLearnerDict.items():
            ngram_loss = learner.loss(batch)
            ngramLosses[i].append(ngram_loss)
            learner.update(batch)
        
        loss = criterion(outputs, targets)

        with torch.no_grad():
            bigram_loss, icl_loss = get_bigram_icl_loss(outputs, targets, out_mask, criterion)
            bigram_losses.append(bigram_loss)
            icl_losses.append(icl_loss)
                
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if config.scheduler is not None: scheduler.step()
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data.to(config.device))
                outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_y)
                eval_losses.append(eval_loss.item())
                eval_steps.append(epoch)
    
    return get_train_result(train_losses=train_losses, 
                            eval_losses=eval_losses, 
                            eval_steps=eval_steps, 
                            attn_maps=attn_maps, 
                            ngramLosses=ngramLosses,
                            bigram_losses=bigram_losses,
                            icl_losses=icl_losses,
                            ngramLosses=ngramLosses)

def train_model(model, config, sampler_config, task_name):
    if task_name == "markov":
        return train_markov(model, config, sampler_config)
    elif task_name == "bietti":
        return train_bietti(model, config, sampler_config)
    elif task_name == "bb":
        return train_bb(model, config, sampler_config)
    else:
        return train_causal(model, config, sampler_config, task_name)