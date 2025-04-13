import torch
import torch.nn.functional as F
from typing import Tuple

import pandas as pd
from itertools import product
from IPython.display import display

from collections import defaultdict

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use

class LatentMarkov:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.device = config.device

        self.total_trans = config.task.total_trans # Total number of transition matrices
        self.order = config.task.order # Order of the Markov chain
        self.num_states_order = self.num_states ** self.order # Number of states in the (high order) Markov chain

        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * config.task.alpha)

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        
        if self.total_trans > 0:
            self.trans_mat = self.dirichlet_dist.sample((self.total_trans, self.num_states_order,))  # Shape: (topics, num_states_order, num_states)
            self.trans_mat /= self.trans_mat.sum(dim=-1, keepdim=True)

    def print_trans_mat(self, task_id):
        '''
        Print the transition matrix for a given task_id.
        task_id: int, the index of the task
        '''
        perms = list(product(range(self.num_states), repeat=self.order))
        perms = [''.join(map(str, p)) for p in perms]
        df = pd.DataFrame(self.trans_mat[task_id].cpu(), index=perms, columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)
    
    # generate samples from the model
    def generate(self, epochs=1, mode:str="train",
                 task=None, num_samples=None)-> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size 
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode == "testing":
            num_samples = num_samples if num_samples is not None else 1
        elif mode in ["eval", "ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        
        num_samples *= epochs
        if task is None:
            self.latent = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device) # Shape: (num_samples,), randomly choose a transition matrix for each sample
        else:
            assert task < self.total_trans, "task id out of range"
            self.latent = torch.full((num_samples,), task, dtype=torch.long, device=self.device)

        if mode in ["train", "test", "testing", "eval"]:
            trans_mat = self.trans_mat[self.latent] # Shape: (num_samples, num_states_order, num_states)
        elif (mode == "ood") or (self.total_trans == 0):
            trans_mat = self.dirichlet_dist.sample((num_samples, self.num_states_order,))  # Shape: (num_samples, num_states_order, num_states)
            trans_mat /= trans_mat.sum(dim=-1, keepdim=True)

        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state

        range_vec = torch.arange(num_samples, device=self.device) # Shape: (num_samples,)
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)

            probs = trans_mat[range_vec, state_indices]  # Shape: (num_samples, num_states)

            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:].clone()  # Shift left
            state[:, -1] = next_states    # Append new state
        
        if mode == "train":
            return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)

        if mode == "testing" and task is None:
            return samples, probs, self.latent

        return samples, probs

    
    # generate summary statistics of the sampler
    def summary(self)-> defaultdict:
        unigram_stats = defaultdict(torch.Tensor)
        num_samples = 1000
        for i in range(self.total_trans):
            samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
            samples[:, :self.order] = state
            
            for t in range(self.order, self.seq_len):
                state_indices = torch.sum(state*self.powers, dim=1)
                probs = self.trans_mat[i][state_indices]  # Shape: (num_samples, num_states)
                
                # Sample the next states for the entire batch
                next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Update the sequence with the sampled next states
                samples[:, t] = next_states
                
                # Update the state window (shift left and append the new state)
                # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
                state[:, :-1] = state[:, 1:]  # Shift left
                state[:, -1] = next_states    # Append new state
            
            unigram_stats[i] = torch.bincount(samples.flatten(), minlength=self.num_states).float() / num_samples / self.seq_len
        
        return unigram_stats
    
    def modified(self, latent_old, latent_new, token):
        old_trans_mat = self.trans_mat.clone()
        self.trans_mat[latent_old, token] = self.trans_mat[latent_new, token]
        num_samples = 1
        
        latent = latent_old
        
        print("Latent variable: ", latent)
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)
            probs = self.trans_mat[latent][state_indices]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:].clone()  # Shift left
            state[:, -1] = next_states    # Append new state
            
        self.trans_mat = old_trans_mat

        return samples, probs