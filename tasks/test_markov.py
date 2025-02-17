import torch
import torch.nn.functional as F
from typing import Tuple


# config specifies the number of different topics
# each time, we randomly sample a topic to use its bigram transition matrix

class TopicModel:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.topics = config.topics
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.alpha)
        
        self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
    
    def generate(self, epochs=1, mode:str="train")-> torch.Tensor:
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state*self.powers, dim=1)
            probs = self.trans_matrix[state_indices]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
            
        return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)