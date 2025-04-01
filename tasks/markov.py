import torch
import torch.nn.functional as F
from typing import Tuple
import pandas as pd
from itertools import product
from IPython.display import display
from tasks.random_distributions import *

# TODO: maybe switch to JAX in the future?

# Simple Markov chain sampler
class MarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.trans = {}
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.task.alpha)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        # Sample all transition probabilities in one go
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

def markov_generate_unjitted(trans_matrix:torch.Tensor, num_samples:int, seq_len:int, num_states:int, order:int, device:str, epochs:int=1)->Tuple[torch.Tensor, torch.Tensor]:
    # num_samples = self.batch_size if mode == "train" else self.test_size
        
    # Initialize the samples tensor
    num_samples *= epochs
    powers = (num_states ** torch.arange(order - 1, -1, -1, device=device)).long()
    samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    
    # Initialize the state (randomly choose starting states for each sequence)
    state = torch.randint(high=num_states, size=(num_samples, order), device=device)
    samples[:, :order] = state
    probs = torch.zeros((num_samples, num_states), device=device)
        
    for t in range(order, seq_len):
        state_indices = torch.sum(state*powers, dim=1)
        probs = trans_matrix[state_indices]  # Shape: (num_samples, num_states)
        
        # Sample the next states for the entire batch
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Update the sequence with the sampled next states
        samples[:, t] = next_states
        
        # Update the state window (shift left and append the new state)
        # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
        state[:, :-1] = state[:, 1:]  # Shift left
        state[:, -1] = next_states    # Append new state
        
    return samples.reshape(epochs, -1, seq_len), probs.reshape(epochs, -1, num_states)


@torch.jit.script
def markov_generate_jitted(trans_matrix:torch.Tensor, num_samples:int, seq_len:int, num_states:int, order:int, device:str, epochs:int=1)->Tuple[torch.Tensor, torch.Tensor]:
    # num_samples = self.batch_size if mode == "train" else self.test_size
        
    # Initialize the samples tensor
    num_samples *= epochs
    powers = (num_states ** torch.arange(order - 1, -1, -1, device=device)).long()
    samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    
    # Initialize the state (randomly choose starting states for each sequence)
    state = torch.randint(high=num_states, size=(num_samples, order), device=device)
    samples[:, :order] = state
    probs = torch.zeros((num_samples, num_states), device=device)
        
    for t in range(order, seq_len):
        state_indices = torch.sum(state*powers, dim=1)
        probs = trans_matrix[state_indices]  # Shape: (num_samples, num_states)
        
        # Sample the next states for the entire batch
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Update the sequence with the sampled next states
        samples[:, t] = next_states
        
        # Update the state window (shift left and append the new state)
        # state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
        state[:, :-1] = state[:, 1:]  # Shift left
        state[:, -1] = next_states    # Append new state
        
    return samples.reshape(epochs, -1, seq_len), probs.reshape(epochs, -1, num_states)


# ICL Markov chain sampler
class ICLMarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.alpha = config.task.alpha
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*self.alpha)
    
    def get_stationary(self, pi: torch.Tensor)->torch.Tensor:
        pi_t = pi.transpose(1, 2)  # Transpose each matrix, Shape: (num_samples, num_states, num_states_order)
        svd_input = pi_t - torch.eye(self.num_states, device=self.device).unsqueeze(0)
        _, _, v = torch.linalg.svd(svd_input)
        mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix
        return mu / mu.sum(dim=1, keepdim=True)

    def generate(self, mode="train", epochs=1):
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        range_vecs = torch.arange(num_samples, device=self.device)

        # Sample all transition probabilities in one go
        trans_matrix = self.dirichlet_dist.sample((num_samples, self.num_states_order,))  # Shape: (num_samples, num_states_order, num_states)
        trans_matrix /= trans_matrix.sum(dim=-1, keepdim=True)

        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

        if self.order == 1:
            mu = self.get_stationary(trans_matrix) # Shape: (num_samples, num_states)
            state = torch.multinomial(mu, num_samples=1) # Shape: (num_samples,1)
            samples[:, :self.order] = state
        else:
            # Initialize the state (randomly choose starting states for each sequence)
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device) # Shape: (num_samples, order)
            samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1) #shape: (num_samples,)
            probs = trans_matrix[range_vecs, state_indices, :]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
        
        return samples.reshape(epochs, -1, self.seq_len), probs.reshape(epochs, -1, self.num_states)







# Fixed Random Markov chain sampler
class FRMarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.card = config.task.cardinality
        self.test_size = config.test_size
        self.device = config.device
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.task.alpha)
        self.random_dist = get_dist(config) #RandomHotDistribution(num_states=self.num_states, card=self.card) # torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.random_alpha)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        # Sample all transition probabilities in one go
        self.trans_mat = self.dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.trans_mat /= self.trans_mat.sum(dim=1, keepdim=True)
        self.k = int(config.task.rho * self.num_states) # proportion of rows that have a random transition
        self.fixed = config.task.fixed
        if self.fixed:
            self.q_toks = torch.randperm(self.num_states)[:self.k] # pick random rows
            self.q_toks = self.q_toks.to(self.device)
            print(f"Random triggers: {self.q_toks}")
    
    def print_trans_mat(self):
        perms = list(product(range(self.num_states), repeat=self.order))
        perms = [''.join(map(str, p)) for p in perms]
        df = pd.DataFrame(self.trans_mat.cpu(), index=perms, columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)

    def generate(self, epochs=1, mode:str="train", verbose=False)-> torch.Tensor:
        if mode == "train":
            num_samples = self.batch_size
        elif mode == "test":
            num_samples = self.test_size
        elif mode == "testing":
            num_samples = 1
        elif mode == "probe":
            old_k = self.k
            self.k = 0
            num_samples = self.batch_size
        elif mode == "eval":
            num_samples = self.eval_size
        elif mode == "ood":
            num_samples = self.eval_size

        num_samples *= epochs
        trans_random = self.random_dist.sample((num_samples, self.k,)).to(self.device)  # Shape: (num_samples, k, num_states)
        if verbose:
            print("Random transition probabilities: ", trans_random)
        
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device) # if random transition is used, set to 1
        
        if not self.fixed:
            self.q_toks = torch.argsort(torch.rand(num_samples, self.num_states), dim=1)[:, :self.k] # shape: (num_samples, random_rows_size)
            self.q_toks = self.q_toks.to(self.device)

            if verbose:
                print(f"Random triggers: {self.q_toks}")

        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device) # Shape: (num_samples, order)
        samples[:, :self.order] = state

        current_tokens = state[:, -1]
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1) # shape: (num_samples,)
            probs = self.trans_mat[state_indices]  # Shape: (num_samples, num_states)
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1) # shape: (num_samples,)

            if self.k > 0:
                matches = (self.q_toks == current_tokens.unsqueeze(1))
                matched_indices = matches.nonzero(as_tuple=False)


                if matched_indices.size(0) > 0:
                    rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                    next_states[rows] = torch.multinomial(trans_random[rows, cols], num_samples=1).squeeze(1)  # Using corresponding random transition
                    output_mask[rows, t-1] = 1  # Update output mask
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            current_tokens = next_states
            
            # Update the state window (shift left and append the new state)
            state[:, :-1] = state[:, 1:].clone()  # Shift left
            state[:, -1] = next_states    # Append new state
            
        if mode == "testing":
            return samples, output_mask, self.q_toks, trans_random
        
        if mode == "probe":
            self.k = old_k
            return samples
        
        if mode in ["ood", "eval", "test"]:
            return samples, output_mask
        
        return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len)





class BiettiTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.device = config.device
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.task.alpha)
        self.trans_mat = self.dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.trans_mat /= self.trans_mat.sum(dim=1, keepdim=True)
        self.batch_size = config.batch_size
        self.eval_size = config.eval_size
        self.test_size = config.test_size
        self.k = int(config.task.rho * self.num_states) 
        self.o_max = config.task.o_max
        self.seed = config.seed
        
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        self.fixed = config.task.fixed
        self.alpha = config.task.alpha
        if self.fixed:
            self.q_toks = torch.argsort(self.marginal, descending=True)[:self.k]
            print("Fixed triggers: ", self.q_toks)
    
    def print_trans_mat(self):
        perms = list(product(range(self.num_states), repeat=self.order))
        perms = [''.join(map(str, p)) for p in perms]
        df = pd.DataFrame(self.trans_mat.cpu(), index=perms, columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)
    
    def generate(self, mode="train", epochs=1, return_triggers=False, verbose=False):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        testing_flag = False
        if mode == "train":
            num_samples = self.batch_size
        elif mode == "test":
            num_samples = self.test_size
        elif mode == "eval":
            num_samples = self.eval_size
        elif mode == "probe":
            old_k = self.k
            self.k = 0
            num_samples = self.batch_size
        elif mode == "ood":
            num_samples = self.eval_size
        elif mode == "testing":
            num_samples = 1
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'test', 'eval', 'probe', 'ood', 'testing'")
        
        num_samples *= epochs
        prob_matrix = torch.ones((num_samples, self.num_states)).to(self.device) # self.marginal.unsqueeze(0).repeat(num_samples, 1)
        prob_matrix[:, self.o_max:] = 0 # Avoid sampling from the last o_max tokens
        prob_matrix /= prob_matrix.sum(dim=-1, keepdim=True) # Uniform trigger tokens. 
        # Sample without replacement
        if self.k > 0:
            if (not self.fixed):
                q_toks = torch.multinomial(prob_matrix, self.k, replacement=False)  # Shape: (num_samples, k)
            else:
                q_toks = self.q_toks.unsqueeze(0).repeat(num_samples, 1)
            if verbose:
                print("triggers: ", q_toks)

            trans_probs = torch.ones((num_samples*self.k, self.num_states)).to(self.device)  # Shape: (num_samples * k, num_states)
            trans_probs[torch.arange(num_samples*self.k), q_toks.reshape(-1)] = 0 # Avoid repeating the same token
             
            if mode != "ood":
                trans_probs[:, self.o_max:] = 0
            else:
                trans_probs[:, :self.o_max] = 0

            trans_probs /= trans_probs.sum(dim=-1, keepdim=True) # Uniform output tokens.
            o_toks = torch.multinomial(trans_probs, num_samples=1).reshape(num_samples, self.k)
            if verbose:
                print("outputs: ", o_toks)
        
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)


        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device) # Shape: (num_samples, order)
        samples[:, :self.order] = state
        current_tokens = state[:, -1]
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1) # shape: (num_samples,)
            probs = self.trans_mat[state_indices]  # Shape: (num_samples, num_states)
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1) # shape: (num_samples,)

            if self.k > 0:
                matches = (q_toks == current_tokens.unsqueeze(1))
                matched_indices = matches.nonzero(as_tuple=False)


                if matched_indices.size(0) > 0:
                    rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                    next_states[rows] = o_toks[rows, cols]  # Assign corresponding o_toks
                    output_mask[rows, t-1] = 1  # Update output mask
            
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            current_tokens = next_states
            
            # Update the state window (shift left and append the new state)
            state[:, :-1] = state[:, 1:].clone()  # Shift left
            state[:, -1] = next_states    # Append new state
        
        
        
        if mode == "testing":
            return samples, output_mask, q_toks, F.one_hot(o_toks, self.num_states).squeeze(0).float().to(self.device)

        if mode == "probe":
            self.k = old_k
            return samples
        
        if return_triggers:
            return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len), q_toks.reshape(epochs, -1, self.k)

        if mode in ["ood", "eval"]:
            return samples, output_mask
        
        return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len)
        