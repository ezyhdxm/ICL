from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn  
from tasks.markov import *
from ngram_learner import *
from collections import defaultdict
from tasks.causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *

# from torch.utils.data import DataLoader
from train_utils import *
from plot import *
from IPython.display import display, HTML
from head_view import *
import pickle

def train_generic(model, config, sampler_config, task_handler=None):
    sampler = get_sampler(sampler_config)
    
    train_losses, eval_losses, eval_steps = [], [], []
    last_token_losses = []
    attn_maps, ngramLosses, bigram_losses, icl_losses, probes = {}, defaultdict(int), [], [], defaultdict(list)
    bayes_losses = []
    is_causal = sampler_config.task_name in ["dag", "tree"]
    criterion = nn.CrossEntropyLoss() if not is_causal else last_token_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max) if config.scheduler is True else None
    
    is_icl = "icl" in sampler_config.task_name
    
    test_data, test_info = sampler.generate(mode="test")
    test_data = test_data.squeeze(0)
    test_info = test_info.squeeze(0)
    test_target = test_data[:, 1:].reshape(-1)
    
    if config.ngram > 0:
        ngramLearnerDict = {i:ngramLearner(config, sampler_config, i, is_icl) for i in range(config.ngram)}

        for i, learner in ngramLearnerDict.items():
            learner.update(test_data)
            ngram_loss = learner.loss(test_data)
            ngramLosses[i] = ngram_loss.item()
    
    step = 0
    epochs = min(config.num_epochs, 5000)
    tot_iters = config.num_epochs // epochs
    
    for iters in trange(tot_iters):
        data = sampler.generate(epochs=epochs)
        sample, sample_info = data
        miniters = epochs // 50
        for i in trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            batch_info = sample_info[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            if config.get_attn > 0 and step % config.get_attn == 0:
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            if is_causal:
                with torch.no_grad():
                    bayes_prob = sampler.bayes(batch)
                    bayes_loss = get_bayes_loss(bayes_prob, batch_info)
                    bayes_losses.append(bayes_loss.item())
                outputs = outputs[:,-1,:].reshape(-1, config.vocab_size)
                loss = criterion(outputs, batch_info)
            else:
                last_token = outputs[:, -2, :].reshape(-1, config.vocab_size) # (B, V)
                outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                loss = criterion(outputs, targets)
                if is_icl:
                    last_token_losses.append(last_token_loss(last_token, batch_info).item())
            
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
                            icl_losses=icl_losses, probes=probes, sampler=sampler, 
                            bayes_losses=bayes_losses, last_token_losses=last_token_losses)


# Train model based on task
def train_model(model, config, sampler_config):
    task_handlers = {
        "bietti": bietti_bb_handler,
        "bb": bietti_bb_handler
    }
    return train_generic(model, config, sampler_config, task_handlers.get(sampler_config.task_name, None))


def train_model_with_plot(model, config, sampler_config, show=False, log=True):
    train_results = train_model(model, config, sampler_config)
    get_loss_plots(config, train_results, show=show, log=log)
    gif_paths = defaultdict(list)
    counts = 0
    for layer in range(config.num_layers):
        # for head in range(config.num_heads[layer]):
        gif_paths[layer].append(get_attn_gif(layer, "all", train_results, config))
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

    html = get_head_view(model, train_results, config, trunc=0, action="return")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("attns_plot", exist_ok=True)
    html_file_name = f"attns_plot/attn_view_s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}{config.task_name}_{curr_time}.html"
    with open(html_file_name, "w", encoding="utf-8") as file:
        file.write(html)
    
    # os.makedirs("train_results", exist_ok=True)
    # result_file_name = f"train_results/train_results_s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}{config.task_name}_{curr_time}.pkl"
    # with open(result_file_name, "wb") as file:
    #    pickle.dump(train_results, file)
    