from tqdm import trange
import torch
import torch.nn as nn  
from markov import *
from collections import defaultdict


def train(model, config, ngramLearnerDict=None, ngramLosses=None):
    sampler = MarkovSampler(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_losses, eval_losses, eval_steps = [], [], []
    test_data = sampler.generate(mode="test")
    test_y = test_data[:,1:].reshape(-1).to(config.device)
    attn_maps, test_attn_maps = {}, {}
    for epoch in trange(config.num_epochs):
        model.train()
        batch = sampler.generate() # bottleneck
        optimizer.zero_grad()
        targets = batch[:,1:]
        if config.get_attn > 0 and epoch % config.get_attn == 0:
            outputs, attn = model(batch.to(config.device))
            attn_maps[epoch] = attn.clone()
        else:
            if config.get_attn > 0:
                outputs, _ = model(batch.to(config.device))
            else:
                outputs = model(batch.to(config.device)) 

        outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)

        if config.ngram:
            for i, learner in ngramLearnerDict.items():
                ngram_loss = learner.loss(batch)
                ngramLosses[i].append(ngram_loss)
                learner.update(batch)
        
        targets = targets.reshape(-1)
        targets = targets.to(config.device)
        loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
        if epoch % config.eval_iter == 0:
            with torch.no_grad():
                model.eval()
                if config.get_attn > 0:
                    outputs, test_attn_maps = model(test_data.to(config.device))
                else:
                    outputs = model(test_data.to(config.device))
                outputs = outputs[:,:-1,:].reshape(-1, config.vocab_size)
                loss = criterion(outputs, test_y)
                eval_losses.append(loss.item())
                eval_steps.append(epoch)
                # print(f"Epoch: {epoch}, Evaluation loss: {loss.item()}")
    return train_losses, eval_losses, eval_steps, attn_maps, test_attn_maps