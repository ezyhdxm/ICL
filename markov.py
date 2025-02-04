import torch
import torch.nn.functional as F
from typing import Tuple

# TODO: maybe switching to JAX in the future?

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
    
    def generate(self, mode="train", epochs=1):
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        range_vecs = torch.arange(num_samples, device=self.device)

        # Sample all transition probabilities in one go
        trans_matrix = self.dirichlet_dist.sample((num_samples, self.num_states_order,))  # Shape: (num_samples, num_states_order, num_states)
        trans_matrix /= trans_matrix.sum(dim=1, keepdim=True)
        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        
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



# Empirical n-gram learner
class ngramLearner:
    def __init__(self, config, sampler_config, order, is_icl=False):
        self.order = order
        self.vocab_size = config.vocab_size
        self.alpha = sampler_config.alpha
        self.num_states_order = config.vocab_size**self.order
        self.device = config.device
        self.is_icl = is_icl
        
        if self.order > 0:
            if not is_icl:
                self.trans_mat_est = self.alpha * torch.ones((self.num_states_order, self.vocab_size), device=self.device) # (num_states_order, num_states)
            self.state_powers = self.vocab_size ** torch.arange(self.order - 1, -1, -1, device=self.device)
            
        else:
            self.trans_mat_est = self.alpha*torch.ones((self.vocab_size,), device=self.device)
    
    def update(self, batch): # batch: (B,T)
        batch_size, seq_len = batch.shape
        if self.order > 0:
            if self.is_icl:
                self.trans_mat_est = self.alpha * torch.ones((batch_size, self.num_states_order, self.vocab_size), device=self.device)
            states = torch.stack([batch[:, t:t + self.order] for t in range(seq_len - self.order)], dim=1)  # (B, T-O, O)
            next_states = batch[:, self.order:]  # (B, T-O)

            # Compute state indices as base-vocab_size numbers
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            values = torch.ones_like(state_indices[:,0], dtype=torch.float, device=self.device)  # Same size as positions
            # Update transition matrix
            for t in range(state_indices.size(1)):  # Loop over sequence length (T-O)
                # Add values to the specified positions
                if not self.is_icl:
                    self.trans_mat_est.index_put_((state_indices[:,t], next_states[:,t]), values, accumulate=True)
                else:
                    self.trans_mat_est.index_put_((torch.arange(batch_size), state_indices[:,t], next_states[:,t]), values, accumulate=True)
            if self.is_icl:   
                self.trans_mat_est /= self.trans_mat_est.sum(dim=-1, keepdim=True)
        else:
            if not self.is_icl:
                self.trans_mat_est += torch.bincount(batch.flatten(), minlength=self.vocab_size)
            else:
                bin_counts = torch.stack([torch.bincount(batch[i], minlength=self.vocab_size) for i in range(batch_size)])
                self.trans_mat_est = bin_counts / (bin_counts.sum(dim=-1, keepdim=True)+1e-6)
                
    def predict(self, batch):
        batch_size, seq_len = batch.size()
        if self.order > 0:
            probs = torch.zeros((batch_size, seq_len, self.vocab_size), device=self.device) # (B, T, N)
            uniform = torch.ones((self.vocab_size,), device=self.device) / self.vocab_size # N
            probs[:,:self.order,:] = uniform.repeat(batch_size, self.order, 1)
            states = torch.stack([batch[:, t:t+self.order] for t in range(seq_len-self.order)], dim=1) # (B, T-O, O)
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            if not self.is_icl:
                probs[:, self.order:] = self.trans_mat_est[state_indices] / self.trans_mat_est[state_indices].sum(dim=-1, keepdim=True)
            else:
                batch_indices = torch.arange(batch_size).unsqueeze(1)
                probs[:, self.order:] = self.trans_mat_est[batch_indices, state_indices] 
            return probs

        else:
            if not self.is_icl:
                targets = batch.reshape(-1)
                probs = self.trans_mat_est / self.trans_mat_est.sum()
                probs = probs.unsqueeze(0).repeat(targets.size(0), 1)
                return probs.reshape(batch_size, seq_len, self.vocab_size)
            else:
                probs = self.trans_mat_est.unsqueeze(1).repeat(1, seq_len, 1)
                return probs
            
    def loss(self, batch):
        probs = self.predict(batch)
        one_hot_labels = F.one_hot(batch, num_classes=self.vocab_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(probs+1e-6)) / (batch.size(0) * batch.size(1))
        return loss


