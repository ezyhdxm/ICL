import matplotlib.pyplot as plt

def get_loss_plots(config, train_losses, eval_losses, eval_steps, ngramLosses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs + 1), train_losses, 
             linestyle='-', color='lightblue', label='Training Loss')
    plt.plot(range(1, config.num_epochs + 1), ngramLosses[0], 
             linestyle='-', color='bisque', label='Unigram Loss')
    plt.plot(range(1, config.num_epochs + 1), ngramLosses[1], 
             linestyle='-', color='khaki', label='Bigram Loss')
    plt.plot(range(1, config.num_epochs + 1), ngramLosses[2], 
             linestyle='-', color='burlywood', label='Trigram Loss')
    plt.plot(range(1, config.num_epochs + 1), ngramLosses[3], 
             linestyle='-', color='orange', label='Four-gram Loss')
    plt.plot(eval_steps, eval_losses, 
             linestyle='--', color='palevioletred', label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    mlp = "no" if config.mlp == False else "with"
    plt.title(f'{config.num_heads} Heads {config.num_layers} Layers {mlp} MLP Loss Over Epochs ({config.pos_enc})')
    plt.legend()
    plt.grid()
    plt.show()