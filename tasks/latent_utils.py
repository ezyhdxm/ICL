import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import re
from tqdm import tqdm
from models.ngram_learner import ngramLearner
from tasks.markov import *
from models.base_models import *
from config import *


def ngram_checker(train_results, alpha, verbose=True):
    sampler_config = train_results["sampler_config"]
    seq_len = sampler_config.seq_len
    vocab_size = sampler_config.vocab_size
    test_size = sampler_config.test_size
    device = sampler_config.device
    
    config = train_results["config"]

    sampler_config = MarkovSamplerConfig(order=1, alpha=alpha, seq_len=seq_len, vocab_size=vocab_size, 
                                  test_size=test_size, task_name="icl-mc", device=device)
    sampler = ICLMarkovSampler(sampler_config)
    
    test_data, test_info = sampler.generate(mode="test")
    test_data = test_data.squeeze(0)
    test_info = test_info.squeeze(0)
    test_target = test_data[:, 1:].reshape(-1)
    
    unigram_learner = ngramLearner(sampler_config, 0, True)
    unigram_learner.update(test_data)
    bigram_learner = ngramLearner(sampler_config, 1, True)
    bigram_learner.update(test_data)
    
    unigram_loss = unigram_learner.loss(test_data).item()
    bigram_loss = bigram_learner.loss(test_data).item()

    if verbose: 
        print(f"Unigram Losses Mean: {unigram_loss:.3f}")
        print(f"Bigram Losses Mean: {bigram_loss:.3f}")

    return {"unigram_loss": unigram_loss, "bigram_loss": bigram_loss}


def ID_OOD_check(model, train_results, alpha, verbose=False, ngram=False):
    sampler_config = train_results["sampler_config"]
    
    if alpha != sampler_config.alpha:
        if verbose:
            print(f"OOD check with alpha being {alpha}.")
        check_type = "OOD"
    else:
        check_type = "ID"
        if verbose:
            print(f"ID check.")
    
    seq_len = sampler_config.seq_len
    vocab_size = sampler_config.vocab_size
    test_size = sampler_config.test_size
    device = sampler_config.device
    
    criterion = nn.CrossEntropyLoss()
    
    config = train_results["config"]

    sampler_config = MarkovSamplerConfig(order=1, alpha=alpha, seq_len=seq_len, vocab_size=vocab_size, 
                                  test_size=test_size, task_name="icl-mc", device=device)
    sampler = ICLMarkovSampler(sampler_config)
    
    test_data, test_info = sampler.generate(mode="test")
    test_data = test_data.squeeze(0)
    test_info = test_info.squeeze(0)
    test_target = test_data[:, 1:].reshape(-1)
    test_target_burn = test_data[:, 20:].reshape(-1)
        
    with torch.no_grad():
        model.eval()
        outputs, _ = model(test_data)
        outputs_burn = outputs[:, 19:-1, :].reshape(-1, config.vocab_size)
        outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
        
        eval_loss = criterion(outputs, test_target).item()
        eval_loss_burn = criterion(outputs_burn, test_target_burn).item()

    if verbose:
        print("*******************")
        print(f"Evaluation Losses: {eval_loss:.3f}")
        print(f"Evaluation Losses with 20 burn-in: {eval_loss_burn:.3f}")
 
    return {"eval_loss": eval_loss, "eval_loss_burn": eval_loss_burn, "type": check_type}


def get_ood_id(path, alphas):

    # List all directories in the given path
    folders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    check_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for path in folders:
        with open(f'{path}/train_results.pkl', 'rb') as file:
            train_results_ckpt = pickle.load(file)
            
        config = train_results_ckpt["config"]
        model_ckpt = Transformer(config)
        files = os.listdir(path)
        model_files = [f for f in files if f.startswith('model_') and f.endswith('.pt')]
    
        # Extract the numbers from the filenames
        model_numbers = [(int(re.search(r'(\d+)', f).group()), f) for f in model_files]
        tot_trans = train_results_ckpt['sampler_config'].total_trans
        check_results[tot_trans] = {}
        
        for steps, model in tqdm(sorted(model_numbers), leave=False):
            checkpoint = torch.load(f"{path}/{model}", weights_only=True)
            model_ckpt.load_state_dict(checkpoint)
            model_ckpt = model_ckpt.to(device)
            check_results[tot_trans][steps] = {}
        
            for alpha in alphas:
                check_results[tot_trans][steps][alpha] = ID_OOD_check(model_ckpt, train_results_ckpt, alpha=alpha)
    
    total_tr = len(folders)
    tot_steps = len(model_numbers)
    id_results = np.zeros((tot_steps, total_tr))
    ood_results = {}
    for alpha in alphas:
        ood_results[alpha] = np.zeros((tot_steps, total_tr))
    
    for j, key in enumerate(check_results.keys()):
        check_tot = check_results[key]
        for i, k in enumerate(check_tot.keys()):
            chek_step = check_tot[k]
            id_results[i][j] = chek_step[1]["eval_loss_burn"]
            for alpha in alphas:
                ood_results[alpha][i][j] = chek_step[alpha]["eval_loss_burn"]

    return id_results, ood_results, check_results

def get_ood_id_icl(path, alphas, verbose=True):

    # List all directories in the given path
    check_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(f'{path}/train_results.pkl', 'rb') as file:
        train_results_ckpt = pickle.load(file)
        
    config = train_results_ckpt["config"]
    model_ckpt = Transformer(config)
    files = os.listdir(path)
    model_files = [f for f in files if f.startswith('model_') and f.endswith('.pt')]

    # Extract the numbers from the filenames
    model_numbers = [(int(re.search(r'(\d+)', f).group()), f) for f in model_files]
    
    for steps, model in tqdm(sorted(model_numbers), leave=False):
        checkpoint = torch.load(f"{path}/{model}", weights_only=True)
        model_ckpt.load_state_dict(checkpoint)
        model_ckpt = model_ckpt.to(device)
        check_results[steps] = {}
    
        for alpha in alphas:
            check_results[steps][alpha] = ID_OOD_check(model_ckpt, train_results_ckpt, alpha=alpha)
    
    tot_steps = len(model_numbers)
    id_results = np.zeros((tot_steps, 1))
    ood_results = {}
    for alpha in alphas:
        ood_results[alpha] = np.zeros((tot_steps, 1))

    for i, k in enumerate(check_results.keys()):
        check_step = check_results[k]
        id_results[i][0] = check_step[1]["eval_loss_burn"]
        for alpha in alphas:
            if alpha != 1:
                ood_results[alpha][i][0] = check_step[alpha]["eval_loss_burn"]

    if verbose: 
        for alpha in alphas:
            print("alpha: ", alpha)
            ngram_checker(train_results_ckpt, alpha, verbose=True)
    
    return id_results, ood_results, check_results