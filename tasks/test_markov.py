import torch
import torch.nn.functional as F
from typing import Tuple

from collections import defaultdict

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use

class LatentMarkov:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.total_trans = config.total_trans
        self.order = config.order

        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * config.alpha)

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        
        self.trans_matrix = dirichlet_dist.sample((self.total_trans, self.num_states,))  # Shape: (topics, num_states, num_states)
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
    
    # generate samples from the model
    def generate(self, epochs=1, mode:str="train")-> Tuple[torch.Tensor, torch.Tensor]:
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        self.latent = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device)
        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)

            probs = self.trans_matrix[self.latent, state_indices]  # Shape: (num_samples, num_states)

            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
            
        return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)
    
    # generate one sample for manual inspection
    def test(self)-> Tuple[torch.Tensor, torch.Tensor]:
        num_samples = 1
        latent = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device).item()
        print("Latent variable: ", latent)
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)
            probs = self.trans_matrix[latent][state_indices]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
            
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
                probs = self.trans_matrix[i][state_indices]  # Shape: (num_samples, num_states)
                
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