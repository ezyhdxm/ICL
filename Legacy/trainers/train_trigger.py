def train_trigger(model, config, verbose=False):
    
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
    random_tokens = None
    if hasattr(config.task, 'fixed') and config.task.fixed is True:
        random_tokens = sampler.q_toks

    layer = None
    if True in config.model.mlp:
        layer = config.model.mlp.index(True)
    # print(f"Layer: {layer}")

    # train_losses, eval_losses, eval_steps = [], [], []
    # last_token_losses = []
    attn_maps, probes = {}, defaultdict(list)
    # ood_losses = []
    # many_ngram_losses = {}
    # bayes_losses = []
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.training.learning_rate, 
                                  weight_decay=config.training.weight_decay) #torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config.training.T_max) if config.training.scheduler is True else None
    
    test_data, test_mask = sampler.generate(mode="test")
    test_target = test_data[:, 1:].reshape(-1)
    
    if config.task.ood:
        ood_batch, ood_mask = sampler.generate(mode="ood")
        copy_batch, copy_mask = sampler.generate(mode="copy")
    else:
        ood_batch, ood_mask = None, None
        copy_batch, copy_mask = None, None
    eval_batch, eval_mask = sampler.generate(mode="eval")
    probe_batch = sampler.generate(mode="probe")
    
    # Collect ngram losses for baseline comparison
    if config.ngram > 0:
        print("Evaluating baselines...")
        ngramLearnerDict = {i:mixed_ngramLearner(config, i) for i in range(config.ngram)}

        for i, learner in ngramLearnerDict.items():
            learner.update(test_data, test_mask)
            ngram_loss = learner.loss(test_data, test_mask)
            log["baseline"][i] = ngram_loss.item()
    
    step = 0
    epochs = min(config.training.num_epochs, MAX_SIZE)
    while (config.training.num_epochs % epochs != 0) and (epochs > 0):
        epochs -= 1

    tot_iters = config.training.num_epochs // epochs
    
    wandb.init(config=config, name=exp_name, **config["wandb"])

    
    ##################
    # Start training #
    ##################
    print("Starting training...")
    for iters in range(tot_iters): #trange(tot_iters, leave=False):
        data = sampler.generate(epochs=epochs)
        sample, sample_mask = data
        #miniters = epochs // 50
        for i in range(epochs): #trange(epochs, leave=False, miniters=miniters):
            step += 1
            model.train()
            batch = sample[i]
            # batch_mask = sample_mask[i]
            
            optimizer.zero_grad()
            targets = batch[:, 1:].reshape(-1)

            # get_attn_flag = (step < early_steps) or (step % early_steps == 0)

            if (config.training.get_attn) > 0 and (step % config.training.get_attn == 0):
                outputs, attn = model(batch, get_attn=True)
                attn_maps[step] = {l: v.clone() for l, v in attn.items()}
            else:
                outputs, _ = model(batch)
            
            # last_token = outputs[:, -2, :].reshape(-1, config.vocab_size) # (B, V)
            outputs = outputs[:, :-1, :].reshape(-1, config.vocab_size)
            loss = criterion(outputs, targets)
            
            log["train/loss"].append(loss.item())
            wandb.log({"train/loss": loss.item()}, step=step)
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            
            if config.training.get_checkpoints > 0 and step % config.training.get_checkpoints == 0:
                
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
                    wandb.log({"eval/loss": eval_loss.item()}, step=step)
                    log["eval/step"].append(step)

                    id_loss, icl_loss, ood_loss, copy_error = trigger_handler(model, eval_batch, eval_mask, probes, probe_batch, 
                                                                  config, sampler, random_tokens, layer, 
                                                                  ood_batch, ood_mask, copy_batch, copy_mask)
                    log["eval/IDLoss"].append(id_loss)
                    wandb.log({"eval/IDLoss": id_loss}, step=step)
                    log["eval/ICLLoss"].append(icl_loss)
                    wandb.log({"eval/ICLLoss": icl_loss}, step=step)
                    if ood_loss is not None:
                        log["eval/OODLoss"].append(ood_loss)
                        wandb.log({"eval/OODLoss": ood_loss}, step=step)
                    if copy_error is not None:
                        log["eval/CopyError"].append(copy_error)
                        wandb.log({"eval/CopyError": copy_error}, step=step)
    

    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "step": step,
        }, os.path.join(checkpoint_path, f"model_final_{step}.pt"))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("Training complete.")

    return get_train_result(log=log, attn_maps=attn_maps, probes=probes, sampler=sampler, config=config)
                            # bayes_losses=bayes_losses, last_token_losses=last_token_losses, 