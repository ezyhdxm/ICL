import torch



def plant_dyck(dyck_str: torch.Tensor, total_len: int, device=None) -> torch.Tensor:
    dyck_len = dyck_str.size(0)
    assert dyck_len <= total_len, "Dyck path too long for the target sequence."

    # Sample dyck_len distinct indices
    indices = torch.randperm(total_len, device=device)[:dyck_len]

    # Initialize with zeros (or any pad value if needed)
    planted = torch.zeros(total_len, dtype=dyck_str.dtype, device=device)

    # Plant dyck path at sampled positions (unordered but aligned)
    planted[indices] = dyck_str

    return planted


class DyckPathTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.dyck_length = config.task.dyck_length
        self.total_trans = config.task.total_trans
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.eval_size = config.eval_size
        self.device = config.device
        self.alpha = config.task.alpha

        if self.total_trans > 0:
            self.task_pool = self._init_task_pool()
            # print(f"Task pool initialized with {self.task_pool}.")
        
        self.total_length = self._get_total_length()

        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()

        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
        self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)

    def _random_dyck_path(self) -> torch.Tensor:
        # Step 1: create random balanced walk
        steps = torch.cat([
            torch.ones((self.dyck_length, self.total_trans), dtype=torch.int8, device=self.device),
            -torch.ones((self.dyck_length, self.total_trans), dtype=torch.int8, device=self.device)
        ])
        for i in range(self.total_trans):
            perm = torch.randperm(2 * self.dyck_length, device=self.device)
            steps = steps[perm]

        # Step 2: compute prefix sum and find first min
        prefix_sum = torch.cumsum(steps, dim=0)
        min_val = prefix_sum.min()
        min_indices = (prefix_sum == min_val).nonzero(as_tuple=False)
        min_index = min_indices[0].item() + 1  # +1 to rotate after minimum

        # Step 3: rotate
        steps = torch.cat([steps[min_index:], steps[:min_index]])

        return steps  # tensor of +1/-1

    def _hidden_state_update(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must override `_hidden_state_update`.")
    
    #def _markov_mask(self, hidden_state):
    #    raise NotImplementedError("Subclasses must override `_markov_mask`.")


    def _get_total_length(self):
        raise NotImplementedError("Subclasses must override `_get_total_length`.")

    def _get_powers(self, x):
        raise NotImplementedError("Subclasses must override `_get_powers`.")


    def generate(self, epochs=1, mode="train", num_samples=None, task=None):
        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size 
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode in ["eval", "ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        num_samples *= epochs

        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        masks = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)

        if task is not None:
            hidden_values = task * torch.ones(num_samples, dtype=torch.long, device=self.device)
        else:
            if self.total_trans > 0 and mode != "ood":
                hidden_values = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device)
            else:
                hidden_values = torch.randint(high=self.num_states ** (self.total_length), size=(num_samples,), device=self.device)

        hidden_state = (torch.rand(num_samples) < self.repeat_prob).long().to(device=self.device)

        samples[:, :self.order] = state

        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1)
            probs = self.trans_matrix[state_indices]
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)

            use_markov = hidden_state <= 0

            samples[use_markov, t] = next_states[use_markov]

            power = self._get_powers(hidden_state[~use_markov])

            if self.total_trans == 0 or mode == "ood" or task is not None:
                curr_values = hidden_values[~use_markov] // (self.num_states ** (power))
                
            elif self.total_trans > 0:
                curr_values = self.task_pool[hidden_values[~use_markov]] // (self.num_states ** (power))

            samples[~use_markov, t] = curr_values % self.num_states
            masks[~use_markov, t] = hidden_state[~use_markov]

            hidden_state = self._hidden_state_update(hidden_state)

            state[:, :-1] = state[:, 1:]
            state[:, -1] = next_states

        if mode == "train":
            return samples.reshape(epochs, -1, self.seq_len), masks.reshape(epochs, -1, self.seq_len)

        return samples, masks

    