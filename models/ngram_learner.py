import torch
import torch.nn.functional as F
from tasks.markov import *


# Empirical n-gram learner
class ngramLearner:
    def __init__(self, sampler_config, order, is_icl=False):
        self.order = order
        self.vocab_size = sampler_config.vocab_size
        self.alpha = sampler_config.alpha
        self.num_states_order = sampler_config.vocab_size**self.order
        self.device = sampler_config.device
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


class many_ngramLearners:
    def __init__(self, sampler_config, order, sampler):
        self.order = order
        self.sampler = sampler
        self.markov_sampler = MarkovSampler(sampler_config)
        self.sampler_config = sampler_config
    
    def loss(self):
        total_trans = self.sampler.trans_matrix.size(0)
        loss = 0
        for i in range(total_trans):
            ngram_learner = ngramLearner(self.sampler_config, self.order, is_icl=False)
            self.markov_sampler.trans_matrix = self.sampler.trans_matrix[i]
            batch = self.markov_sampler.generate(1, mode="test")[0].squeeze()
            ngram_learner.update(batch)
            loss += ngram_learner.loss(batch).item()
        return loss / total_trans



# Empirical n-gram learner with masked random transitions
class mixed_ngramLearner:
    def __init__(self, sampler_config, order, is_icl=True):
        self.order = order
        self.vocab_size = sampler_config.vocab_size
        self.alpha = sampler_config.alpha
        self.num_states_order = sampler_config.vocab_size**self.order
        self.device = sampler_config.device
        if sampler_config.task_name in ["bb", "bietti"]:
            self.random_row_size = sampler_config.k
        else: 
            self.random_row_size = int(sampler_config.rho * self.num_states_order) # proportion of rows that have a random transition

        self.is_icl = is_icl
        
        if self.order > 0:
            self.trans_mat_est = self.alpha * torch.ones((self.num_states_order, self.vocab_size), device=self.device) # (num_states_order, num_states)
            self.state_powers = self.vocab_size ** torch.arange(self.order - 1, -1, -1, device=self.device)
            
        else:
            self.trans_mat_est = self.alpha*torch.ones((self.vocab_size,), device=self.device)
    
    def update(self, batch, mask): 
        # batch: (B,T); 
        # mask: (B,T-O), whether to use random transition, mask[:, t] is a bool vector for each sample at step t, true for random transition and false otherwise
        batch_size, seq_len = batch.shape
        mask = mask > 0
        
        if self.order > 0:
            self.random_transition = self.alpha * torch.ones((batch_size, self.num_states_order, self.vocab_size), device=self.device)
            states = torch.as_strided(batch, 
                                      size=(batch_size, seq_len - self.order, self.order), 
                                      stride=(batch.stride(0), batch.stride(1), batch.stride(1)))  # (B, T-O, O)
            next_states = batch[:, self.order:]  # (B, T-O)

            # Compute state indices as base-vocab_size numbers
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            values = torch.ones_like(state_indices[:,0], dtype=torch.float, device=self.device)  # Same size as positions
            # Update transition matrix
            for t in range(state_indices.size(1)):  # Loop over sequence length (T-O)
                # Add values to the specified positions
                # the following is equivalent to: self.trans_mat_est[state_indices[b, t], next_states[b, t]] += 1, if  mask[b,t] is False for all b
                if state_indices[~mask[:,t],t].size(0) > 0:
                    self.trans_mat_est.index_put_((state_indices[~mask[:,t],t], next_states[~mask[:,t],t]), values[~mask[:,t]], accumulate=True) # TODO: take a look at scatter_add_
                # the following is equivalent to: random_transition[b, state_indices[b, t], next_states[b, t]] += 1, if  mask[b,t] is True for all b
                random_indices = state_indices[mask[:,t],t]
                range_vec = torch.arange(batch_size, device=self.device)[mask[:,t]]
                if random_indices.size(0) > 0:
                    self.random_transition[range_vec, random_indices, next_states[mask[:,t],t]] += 1. # there will not be any overlap in this case
        else:
            if not self.is_icl:
                self.trans_mat_est += torch.bincount(batch.flatten(), minlength=self.vocab_size)
            else:
                bin_counts = torch.stack([torch.bincount(batch[i], minlength=self.vocab_size) for i in range(batch_size)])
                self.trans_mat_est = bin_counts / (bin_counts.sum(dim=-1, keepdim=True)+1e-6)
        
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
            for t in range(state_indices.size(1)): # Loop over sequence length (T-O)
                if state_indices[~mask[:,t],t].size(0) > 0:
                    probs[~mask[:, t], self.order+t] = trans_prob_est[state_indices[~mask[:,t],t]]
                if state_indices[mask[:,t],t].size(0) > 0:
                    probs[mask[:, t], self.order+t] = random_prob_est[mask[:, t], state_indices[mask[:,t],t]]
            
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
            
    def loss(self, batch, mask):
        probs = self.predict(batch, mask)
        one_hot_labels = F.one_hot(batch, num_classes=self.vocab_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(probs)) / (batch.size(0) * batch.size(1))
        return loss