from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn  
from markov import *
from collections import defaultdict
from causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *

from torch.utils.data import DataLoader
from train_utils import *




def train_generic(model, config, sampler_config, task_handler=None):
    sampler = get_sampler(sampler_config)
    # Create a DataLoader for efficient batch processing
    dataset = SimulatedDataset(sampler, num_samples=config.num_epochs)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    
    print(f"Dataset Size: {get_batch_size(dataset[0]) * len(dataset) / (1024 ** 2):.2f} MB")
    
    train_losses, eval_losses, eval_steps = [], [], []
    attn_maps, ngramLosses, bigram_losses, icl_losses, probes = {}, defaultdict(int), [], [], defaultdict(list)
    bayes_losses = []
    is_causal = sampler_config.task_name in ["dag", "tree"]
    criterion = nn.CrossEntropyLoss() if not is_causal else causal_criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max) if config.scheduler is True else None
    
    is_icl = "icl" in sampler_config.task_name
    is_info = sampler_config.task_name in ["bietti", "bb", "tree", "dag"]
    
    test_data, test_info = sampler.generate(mode="test") if is_info else (sampler.generate(mode="test"), None)
    test_target = test_data[:, 1:].reshape(-1).to(config.device)
    
    if config.ngram:
        ngramLearnerDict = {i:ngramLearner(config, sampler_config, i, is_icl) for i in range(config.max_gram)}

        for i, learner in ngramLearnerDict.items():
            learner.update(test_data)
            ngram_loss = learner.loss(test_data)
            ngramLosses[i] = ngram_loss.item()
    
    step = 0
    
    for sample in tqdm(data_loader):
        step += 1
        model.train()
        batch, batch_info = sample if is_info else (sample, None)
        batch = batch.squeeze(0)
        batch_info = batch_info.squeeze(0) if batch_info is not None else None
        optimizer.zero_grad()
        targets = batch[:, 1:].reshape(-1)

        if config.get_attn > 0 and step % config.get_attn == 0:
            outputs, attn = model(batch, get_attn=True)
            attn_maps[step] = {l: v.clone() for l, v in attn.items()}
        else:
            outputs, _ = model(batch.to(config.device))
        
        if is_causal:
            with torch.no_grad():
                bayes_prob = sampler.bayes(batch)
                bayes_loss = get_bayes_loss(bayes_prob, batch_info)
                bayes_losses.append(bayes_loss.item())
            outputs = outputs[:,-1,:].reshape(-1, config.vocab_size)
            loss = criterion(outputs, batch_info)
        else:
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            loss = criterion(outputs, targets)
        
        with torch.no_grad():
            if task_handler:
                task_handler(model, batch, outputs, batch_info, criterion, bigram_losses, icl_losses, probes, sampler_config)
        
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    
        if step % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                outputs, _ = model(test_data)
                outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size) if not is_causal else outputs[:,-1,:].reshape(-1, config.vocab_size)
                eval_loss = criterion(outputs, test_target) if not is_causal else criterion(outputs, test_info)
                eval_losses.append(eval_loss.item())
                eval_steps.append(step)

    return get_train_result(train_losses=train_losses, eval_losses=eval_losses, eval_steps=eval_steps,
                            attn_maps=attn_maps, ngramLosses=ngramLosses, bigram_losses=bigram_losses,
                            icl_losses=icl_losses, probes=probes, sampler=sampler, bayes_losses=bayes_losses)


# Train model based on task
def train_model(model, config, sampler_config):
    task_handlers = {
        "bietti": bietti_bb_handler,
        "bb": bietti_bb_handler
    }
    return train_generic(model, config, sampler_config, task_handlers.get(sampler_config.task_name, None))