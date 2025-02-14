import torch
import torch.nn.functional as F
from typing import Tuple

# TODO: maybe switch to JAX in the future?

# Simple Markov chain sampler
class MarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.trans = {}
        self.order = config.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.alpha)
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
        self.order = config.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.alpha = config.alpha
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




# Bigram data task: https://arxiv.org/pdf/2306.00802
# By vectorization, our implemetation is much faster compared to the original implementation.
class BiettiTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.marginal = config.marginal.to(config.device)
        self.trans_mat = config.trans_mat.to(config.device)
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.k = config.k
        self.show_latents = config.show_latents
        self.seed = config.seed
        self.show_mask = config.show_mask
        self.device = config.device
    
    def generate(self, mode="train", epochs=1):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        prob_matrix = self.marginal.unsqueeze(0).repeat(num_samples, 1)
        # Sample without replacement
        q_toks = torch.multinomial(prob_matrix, self.k, replacement=False)  # Shape: (num_samples, k)
        trans_probs = self.trans_mat[q_toks.reshape(-1)]  # Shape: (num_samples * k, num_states)
        o_toks = torch.multinomial(trans_probs, num_samples=1).reshape(num_samples, self.k)
        
        # Initialize the samples tensor
        if self.show_latents:
            samples = torch.zeros((num_samples, self.seq_len+self.k), dtype=torch.long, device=self.device)
            samples[:, :self.k] = q_toks
            output_mask = torch.zeros((num_samples, self.seq_len+self.k), dtype=torch.long, device=self.device)
            output_mask[:, :self.k] = -1
            off_set = self.k
        else:
            samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
            output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
            off_set = 0

         # Initialize the state (randomly choose starting states for each sequence)
        current_tokens = torch.multinomial(prob_matrix, num_samples=1).squeeze(1)
        samples[:, off_set] = current_tokens
        
        for t in range(1, self.seq_len):
            # Check if current_tokens are in q_toks
            matches = (q_toks == current_tokens.unsqueeze(1))  # Shape: (num_samples, k)
            matched_indices = matches.nonzero(as_tuple=False)  # Indices where matches occur

            # Prepare next tokens
            nxt_tokens = torch.full((num_samples,), -1, dtype=torch.long, device=self.device)  # Placeholder for next tokens

            # Case 1: Replace with o_toks when current_tokens match q_toks
            if matched_indices.size(0) > 0:
                rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                nxt_tokens[rows] = o_toks[rows, cols]  # Assign corresponding o_toks
                output_mask[rows, off_set+t-1] = 1  # Update output mask

            # Case 2: Sample from the transition matrix for unmatched tokens
            unmatched_mask = nxt_tokens == -1  # Mask for tokens not matched in q_toks
            unmatched_tokens = current_tokens[unmatched_mask]  # Unmatched tokens
            if unmatched_tokens.size(0) > 0:
                transition_probs = self.trans_mat[unmatched_tokens]  # Get transition probabilities
                sampled_tokens = torch.multinomial(transition_probs, num_samples=1).squeeze(1)  # Sample next tokens
                nxt_tokens[unmatched_mask] = sampled_tokens

            # Update sequences and current tokens
            samples[:, off_set+t] = nxt_tokens
            current_tokens = nxt_tokens
        
        if self.show_mask:
            return samples.reshape(epochs, -1, self.seq_len+off_set), output_mask.reshape(epochs, -1, self.seq_len+off_set)
        
        return samples.reshape(epochs, -1, self.seq_len+off_set)


# Bigram Backcopy task: https://arxiv.org/pdf/2410.13835
# TODO: add attention visualization
class BBTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.device = config.device
        self.num_states = config.vocab_size-1
        self.bos = self.num_states
        self.marginal = config.marginal
        self.trans_mat = config.trans_mat
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.k = config.k
        self.seed = config.seed
        self.show_mask = config.show_mask
        # fixed triggers
        self.q_toks = torch.multinomial(self.marginal, self.k, replacement=False)  # Shape: (k,)
        # initial probability without triggers
        self.init_prob = self.marginal.clone()
        self.init_prob[self.q_toks] = 0.
        self.init_prob /= self.init_prob.sum() 
        
    
    def generate(self, mode="train", epochs=1):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs

        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

         # Initialize the state to be BOS
        prev_tokens = torch.ones((num_samples,), dtype=torch.long, device=self.device) * self.bos # shape: (num_samples,)
        samples[:, 0] = prev_tokens
        current_tokens = torch.multinomial(self.init_prob.repeat(num_samples, 1).to(self.device), num_samples=1).squeeze()
        samples[:, 1] = current_tokens
        
        for t in range(2, self.seq_len):
            # Check if current_tokens are in q_toks
            is_trigger = torch.isin(current_tokens, self.q_toks)

            # Prepare next tokens
            nxt_tokens = torch.full((num_samples,), -1, dtype=torch.long, device=self.device) # Placeholder for next tokens
            nxt_tokens[is_trigger] = prev_tokens[is_trigger]
            output_mask[is_trigger, t-1] = 1  # Update output mask

            not_trigger_indices = torch.nonzero(~is_trigger).squeeze(1)
            if not_trigger_indices.dim() > 0 and len(not_trigger_indices) > 0:  # Avoid empty sampling
                # Get the current token indices for rows of the transition matrix
                transition_rows = current_tokens[not_trigger_indices]
                # Sample new tokens for non-trigger indices
                sampled_tokens = torch.multinomial(self.trans_mat[transition_rows], 1).squeeze()
                nxt_tokens[not_trigger_indices] = sampled_tokens
                

            # Update result for time step t
            samples[:, t] = nxt_tokens

            # Prepare for the next iteration
            prev_tokens, current_tokens = current_tokens.clone(), nxt_tokens.clone()
        
        if self.show_mask:
            return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len)
        
        return samples.reshape(epochs, -1, self.seq_len)





# Fixed Random Markov chain sampler
class FRMarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.order = config.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device)*config.alpha)
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        # Sample all transition probabilities in one go
        self.base_trans_matrix = self.dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.base_trans_matrix /= self.base_trans_matrix.sum(dim=1, keepdim=True)
        self.random_rows_size = int(config.rho * self.num_states_order) # proportion of rows that have a random transition
        self.fixed = config.fixed
        if self.fixed:
            self.random_rows = torch.randperm(self.num_states_order)[:self.random_rows_size] # pick random rows
            self.random_rows = self.random_rows.to(self.device)
        
            print(f"Random rows: {self.random_rows}")
        
    
    def generate(self, epochs=1, mode:str="train")-> torch.Tensor:
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        trans_random = self.dirichlet_dist.sample((num_samples, self.random_rows_size,))  # Shape: (num_samples, random_rows_size, num_states)
        # print(trans_random)
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device) # if random transition is used, set to 1
        
        if not self.fixed:
            self.random_rows = torch.argsort(torch.rand(num_samples, self.num_states_order), dim=1)[:, :self.random_rows_size] # shape: (num_samples, random_rows_size)
            self.random_rows = self.random_rows.to(self.device)

            # print(f"Random rows: {self.random_rows}")

        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device) # Shape: (num_samples, order)
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1) # shape: (num_samples,)
            probs = self.base_trans_matrix[state_indices]  # Shape: (num_samples, num_states)
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)

            if self.fixed:
                matches = self.random_rows==state_indices.unsqueeze(1) # check if the current state is in random rows, shape: (num_samples, random_rows_size)
                matched_indices = matches.nonzero(as_tuple=False)  # Indices where matches occur, shape: (num_samples, 2)
            else:
                # print(state_indices)
                matches = self.random_rows==state_indices.unsqueeze(1)
                # print(matches)
                matched_indices = matches.nonzero(as_tuple=False)
                # print(matched_indices)

            if matched_indices.size(0) > 0:
                rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                next_states[rows] = torch.multinomial(trans_random[rows, cols], num_samples=1).squeeze(1)  # Using corresponding random transition
                output_mask[rows, t-1] = 1  # Update output mask
            
            
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            state[:, :-1] = state[:, 1:]  # Shift left
            state[:, -1] = next_states    # Append new state
        
        return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len)