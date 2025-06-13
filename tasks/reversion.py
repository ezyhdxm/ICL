import torch
from tasks.base_latent_abc import BaseLatentSequenceTask

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use




  # Combine the two tensors


# For any sequence a in [n]**L, we give it a index in [0, n**L - 1] based on a[0] + a[1] * n + a[2] * n**2 + ... + a[L-1] * n**(L-1). 
# Thus, given an index i, we can find the sequence a by computing a[j] = (i // n**j) % n for j in range(L).

class ReversedTask(BaseLatentSequenceTask):
    def _init_task_pool(self):
        task_pool = torch.randint(high=self.num_states, size=(self.total_trans, self.repeat_length), device=self.device)
        # powers = (self.num_states ** torch.arange(self.repeat_length - 1, -1, -1, device=self.device)).long()
        # task_pool = torch.sum(hidden_values_seq * powers, dim=1)
        '''
        task_pool = torch.randint(
            low=0,
            high=self.num_states ** self.repeat_length,
            size=(self.total_trans,),
            device=self.device
        )
        '''
        
        return task_pool
    
    def _get_index(self, hidden_state):
        x = hidden_state.clone()
        total = 2*self.repeat_length + 1
        x[x > self.repeat_length] = total - x[x > self.repeat_length]
        return x - 1

    def _hidden_state_update(self, x: torch.Tensor) -> torch.Tensor:
        L = self.repeat_length
        p = self.repeat_prob
        x = x.clone()
        # Step 1: increment non-zero values by 1, wrapping around at L
        nonzero_mask = (x > 0)
        x[nonzero_mask] += 1

        x[x == (L+1)] = -2

        # Step 2: replace 0s with 1s with probability p
        zero_mask = (x == 0)
        flip_mask = torch.rand_like(x, dtype=torch.float) < p
        replace_mask = zero_mask & flip_mask
        x[replace_mask] = 1

        # Step 3: replace -1s with (L+1)s with probability p
        neg_mask = (x == -1)
        flip_mask = torch.rand_like(x, dtype=torch.float) < p
        replace_mask = neg_mask & flip_mask
        x[replace_mask] = L+1

        # Step 3: wrap around values greater than 2L to -3 and values equal to -2 to -1
        x[x > (2*L)] = -3
        x[x == -2] = -1

        return x

    def _get_total_length(self):
        return self.repeat_length

    def _get_powers(self, x):
        return torch.where(x <= self.repeat_length, self.repeat_length - x, x - self.repeat_length - 1)