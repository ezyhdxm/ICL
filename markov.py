import torch
import torch.nn.functional as F

class MarkovSampler:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.trans = {}
        self.order = config.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states)*config.alpha)
        # Sample all transition probabilities in one go
        self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))  # Shape: (num_states_order, num_states)
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
        
    def generate(self, mode="train"):
        num_samples = self.batch_size if mode == "train" else self.test_size
        
        # Initialize the samples tensor
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long)
        
        # Initialize the state (randomly choose starting states for each sequence)
        state = torch.randint(high=self.num_states, size=(num_samples, self.order))
        samples[:, :self.order] = state
            
        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * (self.num_states ** torch.arange(self.order - 1, -1, -1, device=state.device)), dim=1)
            probs = self.trans_matrix[state_indices]  # Shape: (num_samples, num_states)
            
            # Sample the next states for the entire batch
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update the sequence with the sampled next states
            samples[:, t] = next_states
            
            # Update the state window (shift left and append the new state)
            state = torch.cat([state[:, 1:], next_states.unsqueeze(1)], dim=1)
        
        return samples

class ngramLearner:
    def __init__(self, vocab_size, order):
        self.order = order
        self.vocab_size = vocab_size
        self.num_states_order = vocab_size**order
        if order > 0:
            self.trans_mat_est = torch.ones((self.num_states_order, self.vocab_size)) # (num_states_order, num_states)
            self.state_powers = self.vocab_size ** torch.arange(self.order - 1, -1, -1)
        else:
            self.trans_mat_est = torch.ones((self.vocab_size,))
    
    def update(self, batch): # batch: (B,T)
        seq_len = batch.size(1)
        if self.order > 0:
            self.state_powers = self.state_powers.to(batch.device)
            states = torch.stack([batch[:, t:t + self.order] for t in range(seq_len - self.order)], dim=1)  # (B, T-O, O)
            next_states = batch[:, self.order:]  # (B, T-O)

            # Compute state indices as base-vocab_size numbers
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            values = torch.ones_like(state_indices[:,0], dtype=torch.float)  # Same size as positions
            # Update transition matrix
            for t in range(state_indices.size(1)):  # Loop over sequence length (T-O)
                # Add values to the specified positions
                self.trans_mat_est.index_put_((state_indices[:,t], next_states[:,t]), values, accumulate=True)
        else:
            self.trans_mat_est += torch.bincount(batch.flatten(), minlength=self.vocab_size)
                
    def predict(self, batch):
        batch_size, seq_len = batch.size()
        if self.order > 0:
            self.state_powers = self.state_powers.to(batch.device)
            probs = torch.zeros((batch_size, seq_len, self.vocab_size)) # (B, T, N)
            uniform = torch.ones((self.vocab_size,)) / self.vocab_size # N
            probs[:,:self.order,:] = uniform.repeat(batch_size, self.order, 1)
            states = torch.stack([batch[:, t:t+self.order] for t in range(seq_len-self.order)], dim=1) # (B, T-O, O)
            state_indices = torch.sum(states * self.state_powers, dim=2)  # (B, T-O)
            probs[:, self.order:] = self.trans_mat_est[state_indices] / self.trans_mat_est[state_indices].sum(dim=-1, keepdim=True)
            return probs
        else:
            targets = batch.reshape(-1)
            probs = self.trans_mat_est / self.trans_mat_est.sum()
            probs = probs.unsqueeze(0).repeat(targets.size(0), 1)
            return probs.reshape(batch_size, seq_len, self.vocab_size)

    def loss(self, batch):
        probs = self.predict(batch)
        one_hot_labels = F.one_hot(batch, num_classes=self.vocab_size).float()
        loss = -torch.sum(one_hot_labels * torch.log(probs)) / batch.size(0) / (batch.size(1))
        return loss