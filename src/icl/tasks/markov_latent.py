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
        self.bandwidth = getattr(config.task, "bandwidth", None) 
        self.circular_band = getattr(config.task, "circular_band", False) 
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
        

        self.n_major_tasks = config.task.n_tasks # Total number of transition matrices
        self.n_minor_tasks = config.task.n_minor_tasks if 'n_minor_tasks' in config.task else 0
        self.p_minor = config.task.p_minor if 'p_minor' in config.task else 0.0 # Probability of tasks from the minor task pool
        self.order = config.task.order # Order of the Markov chain
        self.num_states_order = self.num_states ** self.order # Number of states in the (high order) Markov chain

        self.dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * config.task.alpha)

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        
        if self.n_major_tasks > 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    self.major_trans_mat = self.dirichlet_dist.sample((self.n_major_tasks, self.num_states_order,))
                    self.major_trans_mat /= self.major_trans_mat.sum(dim=-1, keepdim=True)
                else:
                    self.major_trans_mat = self._sample_banded_trans_mats(self.n_major_tasks)
                self.stationary = None
            else:
                raise NotImplementedError("Random stationary distribution with multiple transition matrices is not implemented yet.")
                self.trans_mat, self.stationary = generate_markov_chains(self.total_trans, 
                                                                         self.num_states, 
                                                                         self.alpha, 
                                                                         device=self.device,
                                                                         seed=self.seed)  # Shape: (topics, num_states_order, num_states)
        if self.n_minor_tasks > 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    self.minor_trans_mat = self.dirichlet_dist.sample((self.n_minor_tasks, self.num_states_order,))
                    self.minor_trans_mat /= self.minor_trans_mat.sum(dim=-1, keepdim=True)
                else:
                    self.minor_trans_mat = self._sample_banded_trans_mats(self.n_minor_tasks)

    def _make_banded_mask(self) -> torch.Tensor:
        if self.bandwidth is None:
            return torch.ones(self.num_states_order, self.num_states, device=self.device, dtype=torch.bool)
        
        k = int(self.bandwidth)
        rows = torch.arange(self.num_states_order, device=self.device)
        last_sym = rows % self.num_states
        cols = torch.arange(self.num_states, device=self.device)

        last_sym = last_sym.view(-1, 1)  # Shape: (num_states_order, 1)
        cols = cols.view(1, -1)          # Shape: (1, num_states)
        if self.circular_band:
            d = torch.abs(last_sym - cols)
            d = torch.minimum(d, self.num_states - d)
            mask = (d <= k)
        else:
            mask = (torch.abs(last_sym - cols) <= k)
        return mask

    def _sample_banded_trans_mats(self, n_mats: int) -> torch.Tensor:
        mask = self._make_banded_mask()  # (num_states_order, num_states) bool
        mask_3d = mask.unsqueeze(0).expand(n_mats, -1, -1)  # (n_mats, num_states_order, num_states)
        gamma = torch.distributions.Gamma(self.alpha * torch.ones((), device=self.device), torch.ones((), device=self.device))
        W = gamma.sample((n_mats, self.num_states_order, self.num_states))
        W = W * mask_3d

        row_sums = W.sum(dim=-1, keepdim=True)  # (n_mats, num_states_order, 1)
        need_fix = (row_sums.squeeze(-1) == 0)  # (n_mats, num_states_order)
        if need_fix.any():
            n = self.num_states
            rows = torch.arange(self.num_states_order, device=self.device)
            diag_col = (rows % n) 
            eye_cols = torch.nn.functional.one_hot(diag_col, num_classes=n).to(W.dtype).to(W.device)  # (num_states_order, n)
            eye_cols = eye_cols.unsqueeze(0).expand(n_mats, -1, -1)  # (n_mats, num_states_order, n)
            W = torch.where(need_fix.unsqueeze(-1), eye_cols, W)
            row_sums = W.sum(dim=-1, keepdim=True)
        P = W / row_sums
        return P

    def to(self, device):
        self.device = device
        if self.n_major_tasks > 0:
            self.major_trans_mat = self.major_trans_mat.to(device)
        if self.n_minor_tasks > 0:
            self.minor_trans_mat = self.minor_trans_mat.to(device)
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
        df = pd.DataFrame(self.major_trans_mat[task_id].cpu(), 
                          index=perms, 
                          columns=[f"{i}" for i in range(self.num_states)])
        pd.set_option('display.float_format', '{:.3f}'.format)
        display(df)
    
    @property
    def total_trans(self) -> int:
        return int(self.n_major_tasks + self.n_minor_tasks)
    
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

        if mode in ["train", "test", "testing", "eval"]:
            if task is None:
                latent_major = torch.randint(high=self.n_major_tasks, size=(num_samples,), device=self.device) # Shape: (num_samples,), randomly choose a transition matrix for each sample
                latent_minor = torch.randint(high=self.n_minor_tasks, size=(num_samples,), device=self.device) if self.n_minor_tasks > 0 else None
            else:
                assert 0 <= task < self.n_major_tasks + self.n_minor_tasks, "task id out of range"
                if task < self.n_major_tasks:
                    latent_major = torch.full((num_samples,), task, dtype=torch.long, device=self.device)
                    latent_minor = None
                else:
                    latent_major = None
                    latent_minor = torch.full((num_samples,), task - self.n_major_tasks, dtype=torch.long, device=self.device)
            
            trans_major = self.major_trans_mat[latent_major] if latent_major is not None else None # Shape: (num_samples, num_states_order, num_states)
            trans_minor = self.minor_trans_mat[latent_minor] if latent_minor is not None else None # Shape: (num_samples, num_states_order, num_states)
            # Build final trans_mat
            if trans_major is not None and trans_minor is not None:
                # flip a Bernoulli coin for each sample
                use_minor = (torch.rand(num_samples, device=self.device) < self.p_minor)
                trans_mat = torch.where(use_minor.view(-1, 1, 1), trans_minor, trans_major)
                latent = torch.where(
                    use_minor,
                    self.n_major_tasks + latent_minor,   # minor bank shifted by n_major_tasks
                    latent_major                          # major bank as-is
                )
            elif trans_major is not None:
                trans_mat = trans_major
                latent = latent_major
            elif trans_minor is not None:
                trans_mat = trans_minor
                latent = self.n_major_tasks + latent_minor
            else:
                raise ValueError("No transition matrices available.")
            
        
        elif mode == "ood" or self.n_major_tasks + self.n_minor_tasks == 0:
            if self.random_stationary is False:
                if self.bandwidth is None:
                    trans_mat = self.dirichlet_dist.sample((num_samples, self.num_states_order,))
                    trans_mat /= trans_mat.sum(dim=-1, keepdim=True)
                else:
                    trans_mat = self._sample_banded_trans_mats(num_samples)
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
            return samples, probs, latent
        
        if mode == "ood" and return_trans_mat:
            return samples, probs, trans_mat, stationary


        return samples, probs

    def get_task_matrix(self, global_task_id: int) -> torch.Tensor:
        """
        Map a global task id [0, total_trans) to a (num_states_order, num_states) matrix.
        """
        assert 0 <= global_task_id < self.total_trans
        if global_task_id < self.n_major_tasks:
            return self.major_trans_mat[global_task_id]
        else:
            return self.minor_trans_mat[global_task_id - self.n_major_tasks]
    
    # generate summary statistics of the sampler
    def summary(self) -> defaultdict:
        unigram_stats = defaultdict(torch.Tensor)
        num_samples = 1000
        device = self.device

        for i in range(self.total_trans):
            trans_i = self.get_task_matrix(i).to(device)  # (S^order, S)

            samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=device)
            state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=device)
            samples[:, :self.order] = state

            # rolling index for speed (see tip below)
            base = self.num_states ** (self.order - 1)
            idx = torch.sum(state * self.powers, dim=1)

            for t in range(self.order, self.seq_len):
                probs = trans_i[idx]  # (num_samples, num_states)
                next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
                samples[:, t] = next_states

                # update rolling index
                if self.order > 0:
                    idx = (idx % base) * self.num_states + next_states

            unigram_stats[i] = torch.bincount(samples.flatten(), minlength=self.num_states).float() / (num_samples * self.seq_len)

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

