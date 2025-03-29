# Bigram data task: https://arxiv.org/pdf/2306.00802
# By vectorization, our implemetation is much faster compared to the original implementation.
class BiettiTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.marginal = config.marginal.to(config.device)
        self.trans_mat = config.trans_mat.to(config.device)
        self.order = config.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.k = config.k
        self.seed = config.seed
        self.device = config.device
        self.fixed = config.fixed
        self.alpha = config.alpha
        if self.fixed:
            self.q_toks = torch.argsort(self.marginal, descending=True)[:self.k]
            print("Fixed triggers: ", self.q_toks)
    
    def generate(self, mode="train", epochs=1, return_triggers=False):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        num_samples = self.batch_size if mode == "train" else self.test_size
        num_samples *= epochs
        prob_matrix = self.marginal.unsqueeze(0).repeat(num_samples, 1)
        # Sample without replacement
        if self.k > 0:
            if not self.fixed:
                q_toks = torch.multinomial(prob_matrix, self.k, replacement=False)  # Shape: (num_samples, k)
            else:
                q_toks = self.q_toks.unsqueeze(0).repeat(num_samples, 1)
            trans_probs = torch.ones((num_samples*self.k, self.num_states)).to(self.device)  # Shape: (num_samples * k, num_states)
            trans_probs[torch.arange(num_samples*self.k), q_toks.reshape(-1)] = 0 # Avoid repeating the same token
            trans_probs /= trans_probs.sum(dim=-1, keepdim=True) # Uniform output tokens. 
            o_toks = torch.multinomial(trans_probs, num_samples=1).reshape(num_samples, self.k)
        
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

         # Initialize the state (randomly choose starting states for each sequence)
        current_tokens = torch.multinomial(prob_matrix, num_samples=1).squeeze(1)
        samples[:, 0] = current_tokens
        
        for t in range(1, self.seq_len):
            # Check if current_tokens are in q_toks
            matches = (q_toks == current_tokens.unsqueeze(1))  # Shape: (num_samples, k)
            matched_indices = matches.nonzero(as_tuple=False)  # Indices where matches occur

            # Prepare next tokens
            nxt_tokens = torch.full((num_samples,), -1, dtype=torch.long, device=self.device)  # Placeholder for next tokens

            # Case 1: Replace with o_toks when current_tokens match q_toks
            if matched_indices.size(0) > 0:
                rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                output_mask[rows, t-1] = samples[rows, t-1]+1  # Update output mask
                nxt_tokens[rows] = o_toks[rows, cols]  # Assign corresponding o_toks
                
            # Case 2: Sample from the transition matrix for unmatched tokens
            unmatched_mask = nxt_tokens == -1  # Mask for tokens not matched in q_toks
            unmatched_tokens = current_tokens[unmatched_mask]  # Unmatched tokens
            if unmatched_tokens.size(0) > 0:
                transition_probs = self.trans_mat[unmatched_tokens]  # Get transition probabilities
                sampled_tokens = torch.multinomial(transition_probs, num_samples=1).squeeze(1)  # Sample next tokens
                nxt_tokens[unmatched_mask] = sampled_tokens

            # Update sequences and current tokens
            samples[:, t] = nxt_tokens
            current_tokens = nxt_tokens
        
        if return_triggers:
            return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len), q_toks.reshape(epochs, -1, self.k)
        
        return samples.reshape(epochs, -1, self.seq_len), output_mask.reshape(epochs, -1, self.seq_len)
        
    
    def test(self):
        num_samples = 1
        prob_matrix = self.marginal.unsqueeze(0).repeat(num_samples, 1)
        # Sample without replacement
        if not self.fixed:
            self.q_toks = torch.multinomial(prob_matrix, self.k, replacement=False)
        print("triggers: ", self.q_toks)
        trans_probs = torch.ones((num_samples*self.k, self.num_states)).to(self.device)  # Shape: (num_samples * k, num_states)
        trans_probs[torch.arange(num_samples*self.k), self.q_toks.reshape(-1)] = 0 # Avoid repeating the trigger tokens
        trans_probs /= trans_probs.sum(dim=-1, keepdim=True)  # Shape: (num_samples * k, num_states)
        o_toks = torch.multinomial(trans_probs, num_samples=1).reshape(num_samples, self.k) # Shape: (num_samples, k)
        print("outputs: ", o_toks)
        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        output_mask = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

         # Initialize the state (randomly choose starting states for each sequence)
        current_tokens = torch.multinomial(prob_matrix, num_samples=1).squeeze(1)
        samples[:, 0] = current_tokens

        for t in range(1, self.seq_len):
            # Check if current_tokens are in q_toks
            matches = (self.q_toks == current_tokens.unsqueeze(1))  # Shape: (num_samples, k)
            matched_indices = matches.nonzero(as_tuple=False)  # Indices where matches occur

            # Prepare next tokens
            nxt_tokens = torch.full((num_samples,), -1, dtype=torch.long, device=self.device)  # Placeholder for next tokens

            # Case 1: Replace with o_toks when current_tokens match q_toks
            if matched_indices.size(0) > 0:
                rows, cols = matched_indices[:, 0], matched_indices[:, 1]  # Batch indices and column indices
                output_mask[rows, t-1] = samples[rows, t-1]+1  # Update output mask
                nxt_tokens[rows] = o_toks[rows, cols]  # Assign corresponding o_toks
                
            # Case 2: Sample from the transition matrix for unmatched tokens
            unmatched_mask = nxt_tokens == -1  # Mask for tokens not matched in q_toks
            unmatched_tokens = current_tokens[unmatched_mask]  # Unmatched tokens
            if unmatched_tokens.size(0) > 0:
                transition_probs = self.trans_mat[unmatched_tokens]  # Get transition probabilities
                sampled_tokens = torch.multinomial(transition_probs, num_samples=1).squeeze(1)  # Sample next tokens
                nxt_tokens[unmatched_mask] = sampled_tokens

            # Update sequences and current tokens
            samples[:, t] = nxt_tokens
            current_tokens = nxt_tokens
        
        return samples, output_mask, self.q_toks, F.one_hot(o_toks, self.num_states).squeeze(0).float().to(self.device)
    
    
    def summary(self):
        batch, output, q_toks = self.generate(epochs=1, mode="test", return_triggers=True)
        batch = batch.squeeze(0)
        output = output.squeeze(0)
        output_mask = (output > 0).float()
        trigger_stats = [torch.unique(output[i][output[i]!=0], return_counts=True)[1] for i in range(self.test_size)]
        max_len = max(tensor.size(0) for tensor in trigger_stats)
        # Pad all tensors to the maximum length
        trigger_stats = torch.stack([F.pad(tensor, (0, max_len - tensor.size(0)), value=0) for tensor in trigger_stats]).float()
        max_trigger_count = trigger_stats.max(axis=-1).values.float().mean().item()
        std_max_trigger_count = trigger_stats.max(axis=-1).values.float().std().item()
        avg_trigger_count = trigger_stats.mean(axis=-1).mean().item()
        std_avg_trigger_count = trigger_stats.mean(axis=-1).std().item()
        n_triggers = output_mask.sum(dim=-1).mean().item()
        std_n_triggers = output_mask.sum(dim=-1).std().item()
        
        pairs = batch.unfold(dimension=1, size=2, step=1).contiguous()
        encoded_pairs = pairs[:, :, 0] * 100 + pairs[:, :, 1]
        repeat_stats = [torch.unique(encoded_pairs[i], return_counts=True)[1] for i in range(self.test_size)]
        # Find the maximum length
        max_len = max(tensor.size(0) for tensor in repeat_stats)
        # Pad all tensors to the maximum length
        repeat_stats = torch.stack([F.pad(tensor, (0, max_len - tensor.size(0)), value=0) for tensor in repeat_stats]).float() # shape: (test_size, seq_len-1)
        # max_count, total_count, n
        max_count = repeat_stats.max(axis=-1).values.float().mean().item()
        std_max_count = repeat_stats.max(axis=-1).values.float().std().item()
        total_count = repeat_stats.sum(axis=-1).mean().item()
        std_total_count = repeat_stats.sum(axis=-1).std().item()
        avg_count = repeat_stats.mean(axis=-1).mean().item()
        med_count = repeat_stats.median(axis=-1).values.float().mean().item()
        std_med_count = repeat_stats.median(axis=-1).values.float().std().item()
        std_avg_count = repeat_stats.mean(axis=-1).std().item()
        count = repeat_stats > 0
        n_avg = count.sum(axis=-1).float().mean().item()
        std_n_avg = count.sum(axis=-1).float().std().item()
        
        
        print("#############################################################################")
        print(f"     Summary statistics of the sampler: averaged over {self.test_size} samples  ")
        print("#############################################################################\n")
        print(f"        Count of the most repeated pair: {max_count:.2f} ({std_max_count:.2f})")
        print(f"        Total repetitions of repeated pairs: {total_count:.2f} ({std_total_count:.2f})")
        print(f"        Avg repetitions for each repeated pair: {avg_count:.2f} ({std_avg_count:.2f})")
        print(f"        Median repetitions for each repeated pair: {med_count:.2f} ({std_med_count:.2f})")
        print(f"        Number of pairs repeated: {n_avg:.2f} ({std_n_avg:.2f})\n\n")
        print(f"        Fraction of trigger pairs: {n_triggers/(self.seq_len-1):.2f} ({std_n_triggers/(self.seq_len-1):.2f})")
        print(f"        Total number of triggers: {n_triggers:.2f} ({std_n_triggers:.2f})")
        print(f"        Avg repetitions among triggers: {avg_trigger_count:.2f} ({std_avg_trigger_count:.2f})")
        print(f"        Max repetitions among triggers: {max_trigger_count:.2f} ({std_avg_trigger_count:.2f})")
        print("#############################################################################")
        
    def modified(self, token):
        old_trans_mat = self.trans_mat.clone()
        prior = torch.ones(self.num_states, device=self.device) * self.alpha
        dirichlet_dist = torch.distributions.Dirichlet(prior)
        self.trans_mat[token] = dirichlet_dist.sample()  # Shape: (vocab_size,)
        while (self.trans_mat[token] - old_trans_mat[token]).abs().sum() < 0.5:
            self.trans_mat[token] = dirichlet_dist.sample()
        print("Modified transition matrix for token: ", self.trans_mat[token])
        print("Old transition matrix for token: ", old_trans_mat[token])
        
        batch, _, q_toks = self.generate(epochs=1, mode="test", return_triggers=True)
        mask = (token == q_toks).sum(dim=-1) # shape: (test_size,)
        batch = batch[mask==0]
        self.trans_mat = old_trans_mat
        return batch