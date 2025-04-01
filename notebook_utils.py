def memory_check(token, model, sampler):
    batch, mask, triggers, _ = sampler.test()
    while (batch == token).sum() == 0 or token in triggers:
        batch, mask, triggers, _  = sampler.test()
    probs = nn.Softmax(-1)(model(batch)[0])[batch==token]
    batch = batch.squeeze(0)
    indices = (batch == token).nonzero(as_tuple=True)[0]

    valid_pairs = []
    for idx in indices:
        if idx + 1 < len(batch):  # Ensure there is a next token
            valid_pairs.append((batch[idx].item(), batch[idx + 1].item()))
    
    print(valid_pairs)
    
    TV_dist = (probs - sampler.trans_mat[token].unsqueeze(0)).abs().sum(dim=-1).detach().cpu()
    KL_div = F.kl_div(sampler.trans_mat[token].log().unsqueeze(0), probs, reduction='none').sum(dim=-1).detach().cpu()
    print("TV distance: ", TV_dist)
    print("KL divergence: ", KL_div)

def generalization_check(sampler, model):
    batch, mask, triggers, out_probs = sampler.test()
    out_probs += 1e-4
    out_probs /= out_probs.sum(-1, keepdim=True)

    tokens = torch.arange(sampler.num_states)
    for trigger in triggers:
        tokens[trigger.item()] = -1

    counter = 1
    pairs = defaultdict(set)
    batch_squeezed = batch.squeeze(0).cpu().numpy()
    
    for i in range(len(batch_squeezed)-1):
        pair = (batch_squeezed[i], batch_squeezed[i+1])
        if (tokens[batch_squeezed[i]] != -1) and (pair not in pairs[batch_squeezed[i]]):
            if len(pairs[batch_squeezed[i]]) == 1:
                tokens[batch_squeezed[i]] = -1
            else:
                pairs[batch_squeezed[i]].add(pair)
        counter += 1
        if tokens.sum() == -sampler.num_states:
            break

    print("All non-triggers can be determined at step: ", counter)
    
    for i, trigger in enumerate(triggers):
        print("Trigger: ", trigger.item())
        probs = nn.Softmax(-1)(model(batch)[0].squeeze())[(mask[0]==1) & (batch[0]==trigger)]
        indices = (batch[0]==trigger).nonzero(as_tuple=True)[0]
        print(probs)
        print("The trigger occurs at: ", indices)
        TV_dist = (probs - out_probs[i].unsqueeze(0)).abs().sum(dim=-1).detach().cpu()
        KL_div = F.kl_div(out_probs[i].log().unsqueeze(0), probs, reduction='none').sum(dim=-1).detach().cpu()
        print("TV distance: ", TV_dist)
        print("KL divergence: ", KL_div)
    
    
    def get_memory(model, tok, mlp=False):
    batch, _, trigger, _ = sampler.test()
    SEQ_LEN = sampler.seq_len
    VOC_SIZE = sampler.num_states
    assert tok < VOC_SIZE, "Token index exceeds the vocabulary size."
    ind = (batch[0] == tok).nonzero(as_tuple=True)[0]
    while (tok in trigger[0]) or (ind[-1]<2) or (ind[-1] > SEQ_LEN-2):
        batch, _, trigger, _ = sampler.test()
        ind = (batch[0] == tok).nonzero(as_tuple=True)[0]
    pos = ind[-1]
    print("Position: ", pos.item())
    base_prob = nn.Softmax(dim=-1)(model(batch)[0])[0][pos].detach().cpu()
    embs = model.embed(batch)
    emb_prob = nn.Softmax(dim=-1)(model.output_layer(embs))[0][pos].detach().cpu()
    if mlp:
        ffn_emb_prob = nn.Softmax(dim=-1)(model.output_layer(model.layers[1].mlp(embs)))[0][pos].detach().cpu()
        ffn_emb_res_prob = nn.Softmax(dim=-1)(model.output_layer(model.layers[1].mlp(embs)+embs))[0][pos].detach().cpu()
    sa = model.layers[0].MHA(embs, False)[0]
    sa_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos].detach().cpu()
    if mlp:
        sa_ffn = model.layers[1].mlp(sa)
        sa_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos].detach().cpu()
    
    if mlp:
        df = pd.DataFrame({'base': base_prob, 'emb': emb_prob, 'ffn(emb)': ffn_emb_prob, "ffn(emb)+emb": ffn_emb_res_prob,
                           'sa1': sa_prob, 'ffn(sa1)': sa_ffn_prob})
    else:
        df = pd.DataFrame({'base': base_prob, 'emb': emb_prob, 'sa1': sa_prob})

    layer = 0    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ground_truth = sampler.trans_mat[batch[0][pos]].cpu()
    print("-"*50)
    print("Layer 1 KL divergence: ")
    print("-"*20)
    for i, key in enumerate(df.keys()):
        kl = F.kl_div(ground_truth.log(), torch.tensor(df[key].values), reduction="sum")
        sig = ""
        if kl < 0.05:
            sig = " ***"
        elif kl < 0.15:
            sig = " **"
        elif kl < 0.3:
            sig = " *"
        ending = "   " if ((i+1) % 4) != 0 else "\n"
        print(f"{key} : {kl.item():.3f}", sig, end=ending)
    # Plot the heatmap
    sns.heatmap(df, cmap="viridis", ax=ax[0], annot=True, fmt=".2f", cbar=False)
    
    # Customize labels
    ax[0].set_title(f"Probability Heatmap: Layer {layer}")
    
    layer = 1
    toks = model.layers[0](embs)[0]
    sa = model.layers[1].MHA(toks, False)[0]
    res_prob = nn.Softmax(dim=-1)(model.output_layer(toks))[0][pos].detach().cpu()
    sa_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos].detach().cpu()
    if mlp:
        sa_ffn = model.layers[1].mlp(sa)
        sa_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos].detach().cpu()
    sa_res = toks+sa
    sa_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res))[0][pos].detach().cpu()
    if mlp:
        sa_res_ffn = model.layers[1].mlp(sa_res)
        sa_res_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn))[0][pos].detach().cpu()
        sa_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+sa))[0][pos].detach().cpu()
        sa_ffn_toks_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+toks))[0][pos].detach().cpu()
    if mlp:
        df = pd.DataFrame({'base': base_prob, 'sa2': sa_prob, 'out1': res_prob, 'sa2+out1': sa_res_prob, 
                           'ffn(sa2)': sa_ffn_prob, "ffn(sa2+out1)": sa_res_ffn_prob, 
                           'ffn(sa2+out1)+sa2': sa_ffn_res_prob, "ffn(sa2+out1)+out1": sa_ffn_toks_prob})
    else:
        df = pd.DataFrame({'base': base_prob, 'sa2': sa_prob, 'out1': res_prob, 'sa2+out1': sa_res_prob})
    print("\n")
    print("-"*20)
    print("Layer 2 KL divergence: SA is second layer self attention applied to the output of the first layer")
    print("-"*20)
    for i, key in enumerate(df.keys()[1:]):
        kl = F.kl_div(ground_truth.log(), torch.tensor(df[key].values), reduction="sum")
        sig = ""
        if kl < 0.05:
            sig = " ***"
        elif kl < 0.15:
            sig = " **"
        elif kl < 0.3:
            sig = " *"
        ending = "   " if ((i+1) % 4) != 0 else "\n"
        print(f"{key} : {kl.item():.3f}", sig, end=ending)
    print("\n")
    print("-"*50)
    # Plot the heatmap
    sns.heatmap(df, cmap="viridis", ax=ax[1], annot=True, fmt=".2f", cbar=False)
    
    # Customize labels
    ax[1].set_title(f"Probability Heatmap: Layer {layer}")
    
    # Show plot
    plt.tight_layout()
    plt.show()

def get_high_order_memory(model, toks, mlp=False):
    toks = torch.tensor([int(ch) for ch in toks], device=sampler.device)
    batch, _, triggers, _ = sampler.generate(mode="testing")
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    batch_stride = torch.as_strided(batch, size=(1, SEQ_LEN-order, order), stride=(batch.stride(0), batch.stride(1), batch.stride(1))).squeeze(0)
    matches = (batch_stride == toks).all(dim=-1)
    indices = torch.nonzero(matches, as_tuple=False).squeeze()
    powers = VOC_SIZE ** torch.arange(order - 1, -1, -1, device=sampler.device)
    tok_ind = torch.sum(toks * powers)

    print(indices.ndim)
    
    while ((torch.isin(toks, triggers).any()) 
           or indices.ndim < 1
           or indices.size(0) < 2
           or (indices[-1]<2) 
           or (indices[-1] > SEQ_LEN-order)):
        batch, _, triggers, _ = sampler.generate(mode="testing")
        batch_stride = torch.as_strided(batch, size=(1, SEQ_LEN-order, order), stride=(batch.stride(0), batch.stride(1), batch.stride(1))).squeeze(0)
        matches = (batch_stride == toks).all(dim=-1)
        indices = torch.nonzero(matches, as_tuple=False).squeeze()
    
    pos = indices[-1]
    print("Position: ", pos.item())
    base_prob = nn.Softmax(dim=-1)(model(batch)[0])[0][pos+order-1].detach().cpu()

    embs = model.embed(batch)
    sa = model.layers[0].MHA(embs, False)[0]
    sa1_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos+order-1].detach().cpu()
    if mlp:
        sa_ffn = model.layers[1].mlp(sa)
        sa1_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos+order-1].detach().cpu()
    
    hidden = model.layers[0](embs)[0]
    sa = model.layers[1].MHA(hidden, False)[0]
    res_prob = nn.Softmax(dim=-1)(model.output_layer(hidden))[0][pos+order-1].detach().cpu()
    sa_prob = nn.Softmax(dim=-1)(model.output_layer(sa))[0][pos+order-1].detach().cpu()
    if mlp:
        out_ffn = model.layers[1].mlp(hidden)
        out_ffn_res = model.layers[1].mlp(hidden) + hidden
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        out_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn_res))[0][pos+order-1].detach().cpu()
        sa_ffn = model.layers[1].mlp(sa)
        sa_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_ffn))[0][pos+order-1].detach().cpu()
    sa_res = hidden+sa
    sa_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res))[0][pos+order-1].detach().cpu()
    if mlp:
        sa_res_ffn = model.layers[1].mlp(sa_res)
        sa_res_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn))[0][pos+order-1].detach().cpu()
        sa_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+sa))[0][pos+order-1].detach().cpu()
        sa_ffn_toks_prob = nn.Softmax(dim=-1)(model.output_layer(sa_res_ffn+hidden))[0][pos+order-1].detach().cpu()
    if mlp:
        df = pd.DataFrame({'base': base_prob, 'sa1': sa1_prob, 'ffn(sa1)': sa1_ffn_prob,
                           'sa2': sa_prob, 'out1': res_prob, 'ffn(out1)': out_ffn_prob, 
                           'sa2+out1': sa_res_prob, "ffn(out1)+out1": out_ffn_res_prob,
                           'ffn(sa2)': sa_ffn_prob, "ffn(sa2+out1)": sa_res_ffn_prob, 
                           'ffn(sa2+out1)+sa2': sa_ffn_res_prob, "ffn(sa2+out1)+out1": sa_ffn_toks_prob})
    else:
        df = pd.DataFrame({'base': base_prob, 'sa1': sa_prob, 
                           'sa2': sa_prob, 'out1': res_prob, 'sa2+out1': sa_res_prob})

    ground_truth = sampler.trans_mat[tok_ind].cpu()

    display(df)
    
    print("-"*50)
    for i, key in enumerate(df.keys()):
        kl = F.kl_div(ground_truth.log(), torch.tensor(df[key].values), reduction="sum")
        sig = ""
        if kl < 0.05:
            sig = " ***"
        elif kl < 0.15:
            sig = " **"
        elif kl < 0.3:
            sig = " *"
        ending = "   " if ((i+1) % 4) != 0 else "\n"
        print(f"{key} : {kl.item():.3f}", sig, end=ending)
    print("\n")
    print("-"*50)
    # Plot the heatmap
    sns.heatmap(df, cmap="viridis", annot=True, fmt=".2f", cbar=False)
    
    # Customize labels
    plt.title(f"Probability Heatmap")
    
    # Show plot
    plt.tight_layout()
    plt.show()


def high_order_memory_probe_df(model, to_probe="ff+res"):
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    perms = list(product(range(VOC_SIZE), repeat=order))
    perms = [''.join(map(str, p)) for p in perms]
    pos = SEQ_LEN - 10
    batch = sampler.generate(mode="probe")
    df = pd.DataFrame(0., index=perms, columns=[f"{i}" for i in range(VOC_SIZE)])
    
    for p in perms:
        toks = torch.tensor([int(ch) for ch in p], device=sampler.device)
        batch_copy = batch.clone()
        batch_copy[0][pos:pos+order] = toks[:]
    
        embs = model.embed(batch_copy)
        hidden = model.layers[0](embs)[0]
        out_ffn = model.layers[1].mlp(hidden)
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        out_prob = nn.Softmax(dim=-1)(model.output_layer(hidden))[0][pos+order-1].detach().cpu()
        out_ffn_res = model.layers[1].mlp(hidden) + hidden
        out_ffn_res_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn_res))[0][pos+order-1].detach().cpu()
        if to_probe == "ff":
            df.loc[p] = out_ffn_prob.numpy()
        elif to_probe == "res":
            df.loc[p] = out_prob.numpy()
        else:
            df.loc[p] = out_ffn_res_prob.numpy()

    print(f"KL divergence: {F.kl_div(sampler.trans_mat.log().cpu(), torch.tensor(df.values), reduction="none").sum(axis=-1).mean():.4f}")
    return df

def high_order_memory_probe_ff_df(model):
    SEQ_LEN, VOC_SIZE, order = sampler.seq_len, sampler.num_states, sampler.order
    perms = list(product(range(VOC_SIZE), repeat=order))
    perms = [''.join(map(str, p)) for p in perms]
    pos = SEQ_LEN - 10
    batch = sampler.generate(mode="probe")
    df = pd.DataFrame(0., index=perms, columns=[f"{i}" for i in range(VOC_SIZE)])
    
    for p in perms:
        toks = torch.tensor([int(ch) for ch in p], device=sampler.device)
        batch_copy = batch.clone()
        batch_copy[0][pos:pos+order] = toks[:]
    
        embs = model.embed(batch_copy)
        hidden = model.layers[0](embs)[0]
        out_ffn = model.layers[1](hidden)[0]
        out_ffn_prob = nn.Softmax(dim=-1)(model.output_layer(out_ffn))[0][pos+order-1].detach().cpu()
        df.loc[p] = out_ffn_prob.numpy()

    print(f"KL divergence: {F.kl_div(sampler.trans_mat.log().cpu(), torch.tensor(df.values), reduction="none").sum(axis=-1).mean():.4f}")
    return df