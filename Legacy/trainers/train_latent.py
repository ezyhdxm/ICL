import os
import json

def train_latent(model, config, verbose=False):

    exp_name = f"train_{get_hash(config)}"
    exp_dir = os.path.join(config.work_dir, exp_name)  
    logging.info(f"Train Experiment\nNAME: {exp_name}\nCONFIG:\n{config}")
    
    # Specify the maximum number of epochs to generate in one pass to speedup data generation
    if config.device == "cpu":
        MAX_SIZE = 500 * (32 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)
    else:
        MAX_SIZE = 500 * (64 * 1024 * 1024 // (config.batch_size * config.seq_len) // 500)

    # Skip if already completed
    log_path = os.path.join(exp_dir, "log.json")
    if os.path.exists(log_path):
        print(f"{exp_name} already completed")
        return
    
    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        f.write(config.to_json())
    
    log = _init_log()
    
    checkpoint_path = os.path.join(exp_dir, f"checkpoints")

    print(tabulate_model(model, 
                         config.seq_len, config.batch_size, config.device)) 
    
    sampler = get_sampler(config)


    # last_token_losses = []
    attn_maps, probes = {}, defaultdict(list)
    # many_ngram_losses = {}
    # bayes_losses = []
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay) #torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.training.T_max) if config.training.scheduler is True else None
    
    is_icl = "icl" in config.task.name
    
    test_data, test_info = sampler.generate(mode="test")
    test_target = test_data[:, 1:].reshape(-1)
    
    
    if config.task.ood:
        ood_batch, _ = sampler.generate(mode="ood")
        ood_target = ood_batch[:, 1:].reshape(-1)
    
    eval_batch, _ = sampler.generate(mode="eval")

    
    # Collect ngram losses for baseline comparison
    if config.ngram > 0:
        print("Evaluating baselines...")

        if config.task.name == "latent":
            many_ngramLearnersDict = {i:many_ngramLearners(config, i, sampler) for i in range(config.ngram)}
            for i, learner in many_ngramLearnersDict.items():
                many_ngram_loss = learner.loss()
                log['baseline'][i] = many_ngram_loss
        
        else:
            ngramLearnerDict = {i:ngramLearner(config, i, is_icl) for i in range(config.ngram)}

            for i, learner in ngramLearnerDict.items():
                learner.update(test_data)
                ngram_loss = learner.loss(test_data)
                log['baseline'][i] = ngram_loss.item()
    
    step = 0
    epochs = min(config.training.num_epochs, MAX_SIZE)
    while config.training.num_epochs % epochs != 0:
        epochs -= 1

    tot_iters = config.training.num_epochs // epochs

    wandb.init(config=config, name=exp_name, **config["wandb"])

    
    ##################
    # Start training #
    ##################

    print("Starting training...")
    for iters in range(tot_iters): #trange(tot_iters, leave=False):
        data = sampler.generate(epochs=epochs)
        sample, sample_info = data
        # miniters = epochs // 50
        for i in range(epochs): # trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            batch_info = sample_info[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0): # and get_attn_flag:
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            loss = criterion(outputs, targets)
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss.item()}, step=step)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            
            if config.training.get_checkpoints > 0 and ((step % config.training.get_checkpoints == 0) 
                                                        or (step < min(config.training.get_checkpoints, 200) and step % 5 == 0)):
                
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save({
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    }, os.path.join(checkpoint_path, f"model_{step}.pt"))


            if (step % config.training.eval_iter == 0) or (step < min(config.training.eval_iter, 100) and step % 5 == 0):
                if verbose:
                    print(f"Step: {step}")
                log["train/step"].append(step)
                lr_val = scheduler.get_last_lr()[0] if scheduler else config.training.learning_rate
                log["train/lr"].append(lr_val)
                wandb.log({"train/lr": lr_val}, step=step)
                with torch.no_grad():
                    model.eval()
                    outputs, _ = model(test_data)
                    outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
                    eval_loss = criterion(outputs, test_target)
                    log["eval/loss"].append(eval_loss.item())
                    wandb.log({"eval/IDLoss": eval_loss.item()}, step=step)
                    log["eval/step"].append(step)
                    if config.task.ood:
                        ood_outputs, _ = model(ood_batch)
                        ood_outputs = ood_outputs[:, :-1, :].reshape(-1, config.vocab_size)
                        ood_loss = criterion(ood_outputs, ood_target)
                        log["eval/OODLoss"].append(ood_loss.item())
                        wandb.log({"eval/OODLoss": ood_loss.item()}, step=step)
                    
                    pth = pth_score(model, eval_batch)
                    log["eval/pth_score"].append(pth)
                    wandb.log({"eval/pth_score": pth}, step=step)
                    ih = ih_score(model, eval_batch, config.device)
                    log["eval/ih_score"].append(ih)
                    wandb.log({"eval/ih_score": ih}, step=step)
                    
    
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "step": step,
        }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")
            

    return get_train_result(log=log, config=config, sampler=sampler, attn_maps=attn_maps, probes=probes)




