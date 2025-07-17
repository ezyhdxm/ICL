import torch
# import torch.nn.functional as F
from typing import Tuple, Optional

import pandas as pd
from itertools import product
from IPython.display import display

from collections import defaultdict

from icl.tasks.utils.latent_utils import generate_markov_chains

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use



class LatentMarkov:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.pad = config.task.pad
        if self.pad:
            self.num_states = config.vocab_size - 1
        else:
            self.num_states = config.vocab_size
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.device = config.device
        if 'stationary' in config.task: # To be compatible with the old config
            self.random_stationary = config.task.stationary # Whether to use sampled stationary distribution
        else:
            self.random_stationary = False 
        if self.random_stationary: assert config.task.order == 1, "Order must be 1 for random stationary distribution in current implementation"
        
        self.alpha = config.task.alpha # Dirichlet prior for the transition matrix
        self.seed = config.seed # Seed for random number generation
        

        self.total_trans = config.task.total_trans # Total number of transition matrices
        self.order = config.task.order # Order of the Markov chain
        self.num_states_order = self.num_states ** self.order # Number of states in the (high order) Markov chain

        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * config.task.alpha)

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        
        if self.total_trans > 0:
            if self.random_stationary is False:
                self.trans_mat = self.dirichlet_dist.sample((self.total_trans, self.num_states_order,))  # Shape: (topics, num_states_order, num_states)
                self.trans_mat /= self.trans_mat.sum(dim=-1, keepdim=True)
                self.stationary = None
            else:
                self.trans_mat, self.stationary = generate_markov_chains(self.total_trans, 
                                                                         self.num_states, 
                                                                         self.alpha, 
                                                                         device=self.device,
                                                                         seed=self.seed)  # Shape: (topics, num_states_order, num_states)

    def to(self, device):
        self.device = device
        self.trans_mat = self.trans_mat.to(device)
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
        if self.random_stationary:
            self.stationary = self.stationary.to(device)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        return 

    def print_trans_mat(self, task_id):
        '''
        Print the transition matrix for a given task_id.
        task_id: int, the index of the task
        '''
        perms = list(product(range(self.num_states), repeat=self.order))
        perms = [''.join(map(str, p)) for p in perms]
        df = pd.DataFrame(self.trans_mat[task_id].cpu(), 
                          index=perms, 
                          columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)
    
    # generate samples from the model
    def generate(self, 
                 epochs=1, mode:str="train",
                 task=None, num_samples: Optional[int] = None, 
                 return_trans_mat=False)-> Tuple[torch.Tensor, torch.Tensor]:
        
        assert mode in ["train", "test", "testing", "eval", "ood"], f"Invalid mode: {mode}"

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
            if self.random_stationary is False:
                trans_mat = self.dirichlet_dist.sample((num_samples, self.num_states_order,))  # Shape: (num_samples, num_states_order, num_states)
                trans_mat /= trans_mat.sum(dim=-1, keepdim=True)
                stationary = None
            else:
                trans_mat, stationary = generate_markov_chains(num_samples,
                                                               self.num_states, 
                                                               self.alpha, 
                                                               device=self.device,
                                                               seed=self.seed)

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
        
        if self.pad:
            padded_samples = torch.zeros((num_samples, 2*self.seq_len-1), dtype=torch.long, device=self.device)
            padded_samples[:, 1::2] = self.num_states
            padded_samples[:, ::2] = samples
            samples = padded_samples

        if mode == "train":
            return samples.reshape(epochs, num_samples//epochs, -1), probs.reshape(epochs, num_samples//epochs, -1)

        if mode == "testing" and task is None:
            return samples, probs, self.latent
        
        if mode == "ood" and return_trans_mat:
            return samples, probs, trans_mat, stationary


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
    

class LatentIDBayes:
    def __init__(self, trans_mat, device="cpu"):
        self.log_trans_mat = trans_mat.log().to(device) # (K, N, N)
        self.total_trans = trans_mat.size(0)
        self.num_states = trans_mat.size(1)
        self.device = device
    
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        K, N, _ = self.log_trans_mat.size()
        B, T = samples.size()
        preds = torch.zeros((B, T, self.num_states), dtype=torch.float, device=self.device)
        samples = samples.to(self.device)

        s0 = samples[:, 0]  # (B,)

        # Build index tensors:
        k_idx = torch.arange(K, device=self.device).view(K, 1, 1)      # (K, 1, 1)
        n_idx = torch.arange(N, device=self.device).view(1, 1, N)      # (1, 1, N)

        # Expand s0 to (1, B, 1) for broadcasting
        s0_expand = s0.view(1, B, 1)                                   # (1, B, 1)

        # Advanced indexing:
        log_trans_rows = self.log_trans_mat[k_idx, s0_expand, n_idx]   # (K, B, N)

        preds[:, 0] = log_trans_rows.exp().mean(dim=0) # (B, N)
        cumulative_log_probs = torch.zeros((self.total_trans, B), dtype=torch.float, device=self.device)

        log_trans_flat = self.log_trans_mat.view(K, N * N)  # (K, N*N)

        for t in range(samples.size(1)-1):
            s_t = samples[:, t] # (B,)
            s_tp1 = samples[:, t+1] # (B,)
            flat_indices = s_t * N + s_tp1  # (B,)
            
            cumulative_log_probs += log_trans_flat[k_idx.view(K,1), flat_indices.view(1,B)] # (K, B)
            s_tp1_expand = s_tp1.view(1, B, 1)  # (1, B, 1)
            curr = self.log_trans_mat[k_idx, s_tp1_expand, n_idx] # (K, B, N)
            log_numerator = torch.logsumexp(curr + cumulative_log_probs.unsqueeze(-1), dim=0)  # (B, N)
            log_denominator = torch.logsumexp(cumulative_log_probs.unsqueeze(-1), dim=0) # (B, 1)
            preds[:, t+1] = (log_numerator - log_denominator).exp()  # (B, N)

        return preds


class LatentOODBayes:
    def __init__(self, num_states, alpha, device="cpu"):
        self.num_states = num_states
        self.device = device
        self.alpha = alpha
    
    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(self.device)
        B, T = samples.size()
        cumsums = torch.zeros((B, self.num_states, self.num_states), 
                              dtype=torch.float, device=self.device)
        
        preds = torch.zeros((B, T, self.num_states), dtype=torch.float, device=self.device)
        preds[:, 0] = 1.0 / self.num_states  # Uniform distribution for the first token
        b_idx = torch.arange(B, device=self.device)

        for t in range(T-1):
            s_t = samples[:, t]  # (B,)
            s_tp1 = samples[:, t+1]
            
            cumsums[b_idx, s_t, s_tp1] += 1
            preds[:, t+1] = (cumsums[b_idx, s_tp1] + self.alpha) / (cumsums[b_idx, s_tp1].sum(dim=-1, keepdim=True) + self.num_states * self.alpha)

        return preds

