import torch
from tasks.base_latent_abc import BaseLatentSequenceTask

# config specifies the number of different transitions
# each time, we randomly sample a transition matrix to use

    


# For any sequence a in [n]**L, we give it a index in [0, n**L - 1] based on a[0] + a[1] * n + a[2] * n**2 + ... + a[L-1] * n**(L-1). 
# Thus, given an index i, we can find the sequence a by computing a[j] = (i // n**j) % n for j in range(L).

class FuzzyCopyTask(BaseLatentSequenceTask):
    def _init_task_pool(self):
        task_pool = torch.randint(
            low=0,
            high=self.num_states ** (2*self.repeat_length),
            size=(self.total_trans,),
            device=self.device
        )
        
        return task_pool

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

        # Step 3: wrap around values greater than 2L to 0 and values equal to -2 to -1
        x[x > (2*L)] = 0
        x[x == -2] = -1

        return x  # Combine the two tensors

    #def _markov_mask(self, hidden_state):
    #    return hidden_state <= 0
    
    def _get_total_length(self):
        return 2*self.repeat_length
    
    def _get_powers(self, x):
        return self.total_length - x
