import torch
from .utils.trie import Trie
import torch
# from tqdm.notebook import trange


class DyckPathTask:
    def __init__(self, config):
        self.pad = config.task.pad if hasattr(config.task, 'pad') else False
        self.bos_pad = config.task.bos_pad if hasattr(config.task, 'bos_pad') else False
        assert not (self.pad and self.bos_pad), "Cannot use both padding and BOS padding at the same time."
        if self.pad or self.bos_pad: self.num_states = config.vocab_size - 3
        else: self.num_states = config.vocab_size - 2  
        self.seq_len = config.seq_len
        if self.pad: assert self.seq_len % 2 == 1, "Sequence length must be odd when padding is enabled."
        self.dyck_length = config.task.dyck_length
        self.total_trans = config.task.total_trans
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.eval_size = config.eval_size
        self.device = config.device
        self.alpha = config.task.alpha
        self.repeat_prob = config.task.repeat_prob
        self.one = self.num_states + 1
        self.neg = self.num_states
        if self.total_trans > 0: self.task_pool = self._random_dyck_path(self.total_trans)
        
        if self.order > 0:
            self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()

            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
            self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))
            self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)
        

    @staticmethod
    def _dyck_path_probability(r,k):
        """
        Vectorized computation of Arnold & Sleep's probability of placing -1.
        r: Tensor of shape [batch_size], number of unmatched +1 steps
        k: Tensor of shape [batch_size], number of steps remaining
        Returns: Tensor of probabilities, shape [batch_size]
        """
        prob = torch.zeros_like(r, dtype=torch.float32)
        mask = r > 0  # valid positions where -1 is possible
        r_, k_ = r[mask], k[mask]
        prob[mask] = (r_ * (k_ + r_ + 2)) / (2 * k_ * (r_ + 1))
        return prob
    
    def _random_dyck_path(self, num_samples) -> torch.Tensor:
        """
        Generate a batch of Dyck paths of length 2n using PyTorch.
        Returns a tensor of shape [batch_size, 2n] with values +1 or -1.
        """
        L = 2 * self.dyck_length
        path = torch.empty(num_samples, L, device=self.device, dtype=torch.int8)
        r = torch.zeros(num_samples, device=self.device, dtype=torch.int32)  # unmatched +1
        k = torch.full((num_samples,), L, device=self.device, dtype=torch.int32)  # remaining steps

        for t in range(L):
            prob_down = self._dyck_path_probability(r, k)
            rand = torch.rand(num_samples, device=self.device)
            step = torch.where(rand < prob_down, torch.full_like(rand, -1, dtype=torch.int8),
                                                torch.full_like(rand,  1, dtype=torch.int8))
            path[:, t] = step
            r += step
            k -= 1
        
        path[path == 1] = self.one
        path[path == -1] = self.neg
        
        return path

    def _plant_dyck(self, dyck_str: torch.Tensor, dyck_mask=None) -> torch.Tensor:
        batch_size, dyck_len = dyck_str.shape # dyck_str shape: [B, L]
        padded = hasattr(self, 'pad')
        if padded:
            padded = self.pad
        seq_len = self.seq_len if not padded else (self.seq_len + 1) // 2

        assert dyck_len <= seq_len, "Dyck path too long for the target sequence."

        if dyck_mask is not None:
            mask = dyck_mask.unsqueeze(0).expand(batch_size, seq_len).to(self.device).to(torch.uint8)
        else:
            mask = (torch.rand((batch_size, seq_len), device=self.device) < self.repeat_prob).to(torch.uint8)
        cumsum_rows = torch.cumsum(mask, dim=1)
        # Create a boolean mask for values beyond limit
        cutoff_mask = cumsum_rows > (self.dyck_length * 2)

        # Zero out only those positions
        mask = mask.masked_fill(cutoff_mask, 0)

        # Initialize with zeros (or any pad value if needed)
        planted = torch.zeros((batch_size, seq_len), dtype=dyck_str.dtype, device=self.device)
        running_index = torch.cumsum(mask.to(torch.int64), dim=1) - 1
        running_index[mask == 0] = -1
        valid_pos = running_index != -1 
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, seq_len)

        planted[valid_pos] = dyck_str[batch_indices[valid_pos], running_index[valid_pos]]

        return planted

    def generate(self, epochs=1, mode="train", num_samples=None, task=None, dyck_mask=None):
        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size 
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode in ["eval", "ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'train', 'test', 'eval', and 'ood'.")
        num_samples *= epochs
        
        padded = hasattr(self, 'pad')
        if padded:
            padded = self.pad

        seq_len = (self.seq_len + 1) // 2 if padded else self.seq_len

        samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=self.device)

        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)

        if task is not None:
            hidden_value_ids = task * torch.ones(num_samples, dtype=torch.long, device=self.device)
            hidden_values = self.task_pool[hidden_value_ids]
        else:
            if self.total_trans > 0 and mode != "ood":
                hidden_value_ids = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device)
                hidden_values = self.task_pool[hidden_value_ids]
            else:
                hidden_values = self._random_dyck_path(num_samples)

        planted_dyck = self._plant_dyck(hidden_values, dyck_mask)
        samples[:, :self.order] = state

        for t in range(self.order, seq_len):
            state_indices = torch.sum(state * self.powers, dim=1)
            probs = self.trans_matrix[state_indices]
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)
            samples[:, t] = next_states

            state[:, :-1] = state[:, 1:]
            state[:, -1] = next_states
        
        masks = planted_dyck != 0
        samples[planted_dyck != 0] = planted_dyck[planted_dyck != 0].long()
        
        padded = hasattr(self, 'pad')
        if padded: padded = self.pad

        if padded:
            padded_samples = torch.full((num_samples, self.seq_len), fill_value=self.num_states+2, dtype=torch.long, device=self.device)
            padded_samples[:, ::2] = samples
            padded_masks = torch.full((num_samples, self.seq_len), fill_value=0, dtype=torch.long, device=self.device)
            padded_masks[:, ::2] = masks
            samples, masks = padded_samples, padded_masks

        if hasattr(self, 'bos_pad') and self.bos_pad: 
            padded_samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
            padded_samples[:, 1:] = samples[:, :-1]
            padded_samples[:, 0] = self.num_states + 2  # BOS token
            padded_masks = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
            padded_masks[:, 1:] = masks[:, :-1]
            samples, masks = padded_samples, padded_masks

        if mode == "train":
            if padded: return samples.reshape(epochs, -1, self.seq_len), masks.reshape(epochs, -1, self.seq_len)
            else: return samples.reshape(epochs, -1, seq_len), masks.reshape(epochs, -1, seq_len)

        return samples, masks


class DyckBayes:
    def __init__(self, config, sampler, flag=False):
        self.pad = config.task.pad if "pad" in config.task else False
        self.bos_pad = config.task.bos_pad if "bos_pad" in config.task else False
        self.num_states = config.vocab_size-1 if (self.pad or self.bos_pad) else config.vocab_size
        self.dyck_length = config.task.dyck_length
        self.trans_matrix = sampler.trans_matrix
        self.repeat_prob = config.task.repeat_prob
        self.trie = None
        self.one = sampler.one
        self.neg = sampler.neg
        self.flag = flag
        
        if config.task.total_trans > 0:
            self.trie = Trie()
            for seq in sampler.task_pool:
                self.trie.insert([1 if s==sampler.one else -1 for s in seq.tolist()])

    def dyck_pos(self, dyckseq):
        """
        seq: Tensor of shape [2*dyck_length], where each element is -1 or 1.
        return: probs[i] = Pr(seq[i] = 1 | seq[:i])
        """
        eps = 1e-6  # small value to avoid division by zero
        probs = torch.zeros(dyckseq.shape[0], device=dyckseq.device, dtype=torch.float32) + eps # probability of being one at each position
        seq_list = [1 if s==self.one else -1 for s in dyckseq.tolist()]
        
        if (self.trie is None) or self.flag:
            dR, dU = 0, 0
            for i, s in enumerate(seq_list):
                probs[i] = (dR - dU + 2) / (dR - dU + 1) * (self.dyck_length - dR) / (2*self.dyck_length - dR - dU)
                if s == -1:
                    dU += 1
                else:
                    dR += 1
        
        else:
            node = self.trie.root
            probs[0] = 1
            for i, s in enumerate(seq_list[:-1]):
                if s not in node.children:
                    break
                node = node.children[s]
                probs[i+1] = node.count_pos / node.count 
        
        return probs

    def extend_dyck_prob(self, seq):
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)  # [1, T]
        B, T = seq.shape

        mask = (seq == self.one) | (seq == self.neg)

        prob = torch.zeros((B,T), dtype=torch.float32, device=seq.device)

        for b in range(B):
            mask_b = mask[b]
            seq_b = seq[b]
            dyck_probs = self.dyck_pos(seq_b[mask_b])
            
            # Step 1: compute cumulative sum of mask (in int form)
            cumsum = mask_b.int().cumsum(dim=0)
            tot = dyck_probs.shape[0]

            prob_b = torch.zeros(T, dtype=dyck_probs.dtype, device=seq.device)

            # Step 3: assign dyckprob[j] where j = cumsum[i] if j < 2L
            valid = (cumsum < tot) & (cumsum >= 0)
            indices = cumsum[valid]
            prob_b[valid] = dyck_probs[indices]
            prob_b[cumsum == tot] = -1
            prob[b] = prob_b
            
        return prob
    
    def fast_markov_probs(self, seq):
        """
        seq: [B, T] integer tokens
        return: [B, T, num_states - 2] markov part
        """
        B, T = seq.shape
        K = self.num_states - 2

        # Preallocate output
        markov_out = torch.zeros((B, T, K), device=seq.device, dtype=torch.float32)

        # Initial uniform probs
        prev = torch.ones((B, K), device=seq.device) / K
        chosen_rows = torch.zeros((B, K), device=seq.device, dtype=torch.float32)

        for t in range(T):
            s_t = seq[:, t]  # [B]

            # Mask where s in {self.one, self.neg}
            update_mask = (s_t == self.one) | (s_t == self.neg) # [B]

            # For mask == True: multiply prev @ trans_matrix
            updated = torch.matmul(prev, self.trans_matrix)  # [B, K]

            # For mask == False: use trans_matrix[s_t]
            chosen_rows[~update_mask] = self.trans_matrix[s_t[~update_mask]]  # [B, K]

            # Combine based on mask
            prev = torch.where(update_mask.unsqueeze(1), updated, chosen_rows)

            # Save
            markov_out[:, t] = prev

        return markov_out

    def pos_prob(self, seq):
        # probs[i] : Pr(seq[i+1] | seq[:i+1])
        if seq.dim() == 1: seq = seq.unsqueeze(0)
        B, T = seq.shape
        K = self.num_states - 2

        probs = torch.zeros((B, T, self.num_states), device=seq.device, dtype=torch.float32)
        dyck_probs = self.extend_dyck_prob(seq) # [B, T]
        dyck_mask = dyck_probs >= 0
        batch_idx, time_idx = torch.where(dyck_mask)
        dyck_vals = dyck_probs[batch_idx, time_idx]
        probs[batch_idx, time_idx, self.one] = self.repeat_prob * dyck_vals
        probs[batch_idx, time_idx, self.neg] = self.repeat_prob * (1 - dyck_vals)
        
        markov_part = self.fast_markov_probs(seq)  # shape [B, T, K]
        probs[batch_idx, time_idx, :K] = markov_part[batch_idx, time_idx, :K] * (1 - self.repeat_prob)
        batch_idx, time_idx = torch.where(~dyck_mask)
        probs[batch_idx, time_idx, :K] = markov_part[batch_idx, time_idx, :K]

        if self.pad or self.bos_pad:
            eps = 1e-8  # or any small constant you need

            # Create a column filled with eps
            eps_column = torch.full((B, T, 1), fill_value=eps, device=probs.device, dtype=probs.dtype)

            # Concatenate along the last dimension
            probs = torch.cat([probs, eps_column], dim=-1)  # shape (B, T, D+1)


        return probs[:,:-1,:]
    
    def predict(self, seq):
        probs = self.pos_prob(seq)
        preds = torch.argmax(probs, dim=-1)
        return preds