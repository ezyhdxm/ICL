import torch
import torch.nn.functional as F
from tasks.markov import *
import pandas as pd
from itertools import product
from IPython.display import display


# Empirical n-gram learner for handling random triggers with a global n-gram model
class mixed_ngramLearner:
    def __init__(self, sampler_config, order):
        self.order = order
        self.random_order = sampler_config.random_order
        self.vocab_size = sampler_config.vocab_size
        self.alpha = sampler_config.alpha
        self.random_alpha = sampler_config.random_alpha
        self.num_states_order = sampler_config.vocab_size**self.order
        self.num_states_random_order = sampler_config.vocab_size**self.random_order
        self.device = sampler_config.device
        if sampler_config.task_name in ["bb", "bietti"]:
            self.random_row_size = sampler_config.k
        else: 
            self.random_row_size = int(sampler_config.rho * self.num_states_order) # proportion of rows that have a random transition
        
        self.random_powers = self.vocab_size ** torch.arange(self.random_order - 1, -1, -1, device=self.device)

        if self.order > 0:
            self.trans_mat_est = self.alpha * torch.ones((self.num_states_order, self.vocab_size), device=self.device) # (num_states_order, num_states)
            self.state_powers = self.vocab_size ** torch.arange(self.order - 1, -1, -1, device=self.device)
            
        else:
            self.trans_mat_est = self.alpha*torch.ones((self.vocab_size,), device=self.device)
    
    def print_trans_mat(self):
        trans_prob_est = self.trans_mat_est / self.trans_mat_est.sum(dim=-1, keepdim=True)
        if self.order > 0:
            perms = list(product(range(self.vocab_size), repeat=self.order))
            perms = [''.join(map(str, p)) for p in perms]
            df = pd.DataFrame(trans_prob_est.cpu(), index=perms, columns=[f"{i}" for i in range(self.vocab_size)])
            pd.set_option('display.float_format', '{:.3f}'.format)
            display(df)
            
        else:
            print(trans_prob_est.cpu())
        
        

    def update(self, batch, mask): 
        # batch: (B,T); 
        # mask: (B,T-O), whether to use random transition, mask[:, t] is a bool vector for each sample at step t, true for random transition and false otherwise

        assert batch.ndim == 2, "batch should be 2D"
        assert mask.ndim == 2, "mask should be 2D"

        batch_size, seq_len = batch.shape
        mask = mask > 0
        
        if self.order > 0:
            backrolled_mask = mask.clone()
            o = self.order - 1
            for t in range(o, mask.size(1)):
                backrolled_mask[:, t-o] = torch.logical_or(backrolled_mask[:, t-o], mask[:, t])
            backrolled_mask = backrolled_mask > 0


            if self.random_alpha > 0:
                self.random_transition = self.random_alpha * torch.ones((batch_size, self.num_states_random_order, self.vocab_size), device=self.device)
            else:
                self.random_transition = torch.zeros((batch_size, self.num_states_random_order, self.vocab_size), device=self.device)
            states = torch.as_strided(batch, 
                                      size=(batch_size, seq_len - self.order, self.order), 
                                      stride=(batch.stride(0), batch.stride(1), batch.stride(1)))  # (B, T-O, O)
            
            next_states = batch[:, self.order:]  # (B, T-O)

            # Compute state indices as base-vocab_size numbers
            state_indices = torch.sum(states * self.state_powers, dim=-1)  # (B, T-O)
            values = torch.ones_like(state_indices[:,0], dtype=torch.float, device=self.device)  # Same size as positions
            # Update transition matrix
            for t in range(state_indices.size(1)):  # Loop over sequence length (T-O)
                # Add values to the specified positions
                # the following is equivalent to: self.trans_mat_est[state_indices[b, t], next_states[b, t]] += 1, if  mask[b,t] is False for all b
                if state_indices[~mask[:,t],t].size(0) > 0:
                    self.trans_mat_est.index_put_((state_indices[~backrolled_mask[:,t],t], next_states[~backrolled_mask[:,t],t]), 
                                                  values[~backrolled_mask[:,t]], accumulate=True) # TODO: take a look at scatter_add_
                # the following is equivalent to: random_transition[b, state_indices[b, t], next_states[b, t]] += 1, if  mask[b,t] is True for all b
            
            random_states = torch.as_strided(batch, 
                                             size=(batch_size, seq_len - self.random_order, self.random_order), 
                                             stride=(batch.stride(0), batch.stride(1), batch.stride(1)))  # (B, T-R, R)
            
            random_state_indices = torch.sum(random_states * self.random_powers, dim=-1)  # (B, T-R)

            next_states = batch[:, self.random_order:]  # (B, T-R)
            
            for t in range(random_state_indices.size(1)):  # Loop over sequence length (T-O)
                random_indices = random_state_indices[mask[:,t],t]
                range_vec = torch.arange(batch_size, device=self.device)[mask[:,t]]
                if random_indices.size(0) > 0:
                    self.random_transition[range_vec, random_indices, next_states[mask[:,t],t]] += 1. # there will not be any overlap in this case
        
        else:
            #if not self.is_icl:
            self.trans_mat_est += torch.bincount(batch.flatten(), minlength=self.vocab_size)
            #else:
            #    bin_counts = torch.stack([torch.bincount(batch[i], minlength=self.vocab_size) for i in range(batch_size)])
            #    self.trans_mat_est = bin_counts / (bin_counts.sum(dim=-1, keepdim=True)+1e-6)
        
        # print(self.random_transition / self.random_transition.sum(dim=-1, keepdim=True))
                
    
    def predict(self, batch, mask):
        batch_size, seq_len = batch.size()
        mask = mask > 0 # (B, T-O)
        if self.order > 0:
            probs = torch.zeros((batch_size, seq_len, self.vocab_size), device=self.device) # (B, T, N)
            uniform = torch.ones((self.vocab_size,), device=self.device) / self.vocab_size # N
            probs[:,:self.order,:] = uniform.repeat(batch_size, self.order, 1)
            states = torch.as_strided(batch, 
                                      size=(batch_size, seq_len - self.order, self.order), 
                                      stride=(batch.stride(0), batch.stride(1), batch.stride(1))) # (B, T-O, O)
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O) 
            trans_prob_est = self.trans_mat_est / self.trans_mat_est.sum(dim=-1, keepdim=True)
            random_prob_est = self.random_transition / self.random_transition.sum(dim=-1, keepdim=True)
            
            for t in range(self.order, state_indices.size(1)+self.order): # Loop over sequence length (T-O)
                if state_indices[~mask[:,t-1],t-self.order].size(0) > 0:
                    probs[~mask[:, t-1], t] = trans_prob_est[state_indices[~mask[:,t-1],t-self.order]]
            
            
            random_states = torch.as_strided(batch, 
                                             size=(batch_size, seq_len - self.random_order, self.random_order), 
                                             stride=(batch.stride(0), batch.stride(1), batch.stride(1))) # (B, T-O, O)
            
            random_state_indices = torch.sum(random_states * self.random_powers, dim=-1)  # (B, T-R)

            for t in range(self.order, state_indices.size(1)+self.random_order):  # Loop over sequence length (T-O)
                if random_state_indices[mask[:,t-1], t-self.random_order].size(0) > 0:
                    random_indices = random_state_indices[mask[:,t-1], t-self.random_order]
                    probs[mask[:, t-1], t] = random_prob_est[mask[:,t-1], random_indices]
            
            return probs

        else:
            #if not self.is_icl:
            targets = batch.reshape(-1)
            probs = self.trans_mat_est / self.trans_mat_est.sum()
            probs = probs.unsqueeze(0).repeat(targets.size(0), 1)
            return probs.reshape(batch_size, seq_len, self.vocab_size)
            #else:
            #    probs = self.trans_mat_est.unsqueeze(1).repeat(1, seq_len, 1)
            #    return probs
            
    def loss(self, batch, mask):
        probs = self.predict(batch, mask)
        one_hot_labels = F.one_hot(batch, num_classes=self.vocab_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(probs+1e-13)) / (batch.size(0) * batch.size(1))
        return loss