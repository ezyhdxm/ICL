from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn  
from tasks.markov import *
from models.ngram_learner import *
from collections import defaultdict
from tasks.causal_graph import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import *

# from torch.utils.data import DataLoader
from train_utils import *
from figures.plot import *
from IPython.display import display, HTML
from figures.head_view import *
import pickle



def train_generic(model, config, sampler_config, task_handler=None, run_time=None):

    # Specify the maximum number of epochs to generate in one pass to speedup data generation
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)

    # print("Max size: ", MAX_SIZE)

    # Use for saving results
    if run_time is None:
        run_time = datetime.now().strftime("%Y%m%d_%H%M")

    sampler = get_sampler(sampler_config)
    random_tokens = None
    if hasattr(sampler_config, 'fixed') and sampler_config.fixed is True:
        if sampler_config.task_name == "frm":
            random_tokens = sampler.random_rows
        if sampler_config.task_name == "bietti":
            random_tokens = sampler.q_toks
    
    if sampler_config.task_name in ["frm", "bietti", "bb"]:
        layer = config.mlp.index(True)
        print(f"Layer: {layer}")



    train_losses, eval_losses, eval_steps = [], [], []
    last_token_losses = []
    attn_maps, ngramLosses, bigram_losses, icl_losses, probes = {}, defaultdict(int), [], [], defaultdict(list)
    many_ngram_losses = {}
    bayes_losses = []
    is_causal = sampler_config.task_name in ["dag", "tree"]
    criterion = nn.CrossEntropyLoss() if not is_causal else last_token_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max) if config.scheduler is True else None
    
    is_icl = "icl" in sampler_config.task_name
    is_mixed = sampler_config.task_name in ["bb", "frm", "bietti"]
    
    test_data, test_info = sampler.generate(mode="test")
    test_data = test_data.squeeze(0)
    test_info = test_info.squeeze(0)
    test_target = test_data[:, 1:].reshape(-1)
    
    
    # Collect ngram losses for baseline comparison
    if config.ngram > 0:
        if is_mixed:
            ngramLearnerDict = {i:mixed_ngramLearner(sampler_config, i, is_icl) for i in range(config.ngram)}

        else:
            ngramLearnerDict = {i:ngramLearner(sampler_config, i, is_icl) for i in range(config.ngram)}

        for i, learner in ngramLearnerDict.items():
            if not is_mixed:
                learner.update(test_data)
                ngram_loss = learner.loss(test_data)
            else:
                learner.update(test_data, test_info)
                ngram_loss = learner.loss(test_data, test_info)
            ngramLosses[i] = ngram_loss.item()
        
        if sampler_config.task_name == "latent":
            many_ngramLearnersDict = {i:many_ngramLearners(sampler_config, i, sampler) for i in range(config.ngram)}
            for i, learner in many_ngramLearnersDict.items():
                many_ngram_loss = learner.loss()
                many_ngram_losses[i] = many_ngram_loss
    
    step = 0
    early_steps = 1000
    epochs = min(config.num_epochs, MAX_SIZE)
    while config.num_epochs % epochs != 0:
        epochs -= 1

    tot_iters = config.num_epochs // epochs

    
    ##################
    # Start training #
    ##################

    for iters in trange(tot_iters, leave=False):
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

            get_attn_flag = (step < early_steps) or (step % early_steps == 0)

            if (config.get_attn) > 0 and (step % config.get_attn == 0) and get_attn_flag:
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
                    # collect probes etc.
                    task_handler(model, batch, outputs, batch_info, criterion, bigram_losses, icl_losses, probes, config, sampler, random_tokens, layer)
            
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            if config.get_checkpoints > 0 and step % config.get_checkpoints == 0:
                os.makedirs(f"checkpoints/{config.task_name}/{run_time}", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{config.task_name}/{run_time}/model_{step}.pt")
        
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
                            bayes_losses=bayes_losses, last_token_losses=last_token_losses, 
                            config=config, sampler_config=sampler_config, many_ngram_losses=many_ngram_losses)


# Train model based on task
def train_model(model, config, sampler_config, run_time=None):
    task_handlers = {
        "bietti": bietti_bb_handler,
        "bb": bietti_bb_handler,
        "frm": bietti_bb_handler,
    }
    return train_generic(model, config, sampler_config, task_handlers.get(sampler_config.task_name, None), run_time)



def train_model_with_plot(model, config, sampler_config, show=False):
    run_time = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(f"loss_plots/{config.task_name}/{run_time}", exist_ok=True)

    train_results = train_model(model, config, sampler_config, run_time=run_time)
    
    folder = f"loss_plots/{config.task_name}/{run_time}"

    get_loss_plots(config, train_results, folder=folder, show=show)

    plot_probes(train_results, config, folder=folder, show=True, log=False)
    plot_probes(train_results, config, folder=folder, show=True, log=True)

    plot_bigram_icl_risk(config, train_results, folder=folder, show=True)

    gif_paths = defaultdict(list)
    counts = 0
    attn_folder = f"attns_plot/{config.task_name}/{run_time}"
    os.makedirs(attn_folder, exist_ok=True)

    for layer in range(config.num_layers):
        # for head in range(config.num_heads[layer]):
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
        get_head_view(model, train_results, config, trunc=0, action="view")
    
    
    html = get_head_view(model, train_results, config, trunc=0, action="return")
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    html_file_name = f"{attn_folder}/attn_view_s{config.seq_len}p_{config.pos_enc}_l{config.num_layers}h{'_'.join(map(str, config.num_heads))}v{config.vocab_size}{config.task_name}_{curr_time}.html"
    with open(html_file_name, "w", encoding="utf-8") as file:
        file.write(html)
    
    os.makedirs(f"checkpoints/{config.task_name}/{run_time}", exist_ok=True)

    last_key = sorted(list(train_results["attn_maps"].keys()))[-1]
    last_attn = train_results["attn_maps"][last_key]
    last_attn["steps"] = last_key
    train_results["attn_maps"] = last_attn

    train_results.pop("eval_losses", None)
    train_results.pop("eval_steps", None)
    train_results.pop("many_ngram_losses", None)
    train_results.pop("last_token_losses", None)
    train_results.pop("bayes_losses", None)
    train_results.pop("ngramLosses", None)
    train_results.pop("bigram_losses", None)
    train_results.pop("icl_losses", None)
    train_results.pop("probes", None)

    result_file_name = f"checkpoints/{config.task_name}/{run_time}/train_results.pkl"
    with open(result_file_name, "wb") as file:
        pickle.dump(train_results, file)
    
    return train_results
    