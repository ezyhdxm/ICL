import torch

class BaseLatentSequenceTask:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.num_states = config.vocab_size
        self.repeat_length_high = config.task.repeat_length_high
        self.repeat_length_low = config.task.repeat_length_low
        self.total_trans = config.task.total_trans
        self.order = config.task.order
        self.num_states_order = self.num_states ** self.order
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.eval_size = config.eval_size
        self.device = config.device
        self.repeat_prob = config.task.repeat_prob
        self.alpha = config.task.alpha

        self.repeat_length = torch.randint(low=self.repeat_length_low, 
                                            high=self.repeat_length_high + 1, 
                                            size=(self.total_trans,), 
                                            device=self.device)

        self.total_length = self._get_total_length()

        if self.total_trans > 0:
            self.task_pool = self._init_task_pool()
            

        # Obtain global transition matrix
        self.powers = (self.num_states ** torch.arange(self.order - 1, -1, -1, device=self.device)).long()
        dirichlet_dist = torch.distributions.Dirichlet(torch.ones(self.num_states, device=self.device) * self.alpha)
        self.trans_matrix = dirichlet_dist.sample((self.num_states_order,))
        self.trans_matrix /= self.trans_matrix.sum(dim=1, keepdim=True)

    def _init_task_pool(self):
        raise NotImplementedError("Subclasses must override `_init_task_pool`.")

    def _hidden_state_update(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must override `_hidden_state_update`.")

    def _get_total_length(self):
        raise NotImplementedError("Subclasses must override `_get_total_length`.")

    #def _get_powers(self, x):
    #    raise NotImplementedError("Subclasses must override `_get_powers`.")

    def _get_index(self, hidden_state, length):
        raise NotImplementedError("Subclasses must override `_get_index`.")

    def generate(self, epochs=1, mode="train", num_samples=None, task=None, length=None):
        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size 
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode in ["eval", "ood", "length_ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        num_samples *= epochs

        samples = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)
        masks = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

        # Generate initial states
        state = torch.randint(high=self.num_states, size=(num_samples, self.order), device=self.device)

        if task is not None:
            assert isinstance(task, int), "Task must be an integer representing the task ID."
            assert 0 <= task < self.total_trans, "Task ID must be within the range of total transitions." 
            hidden_values_ids = task * torch.ones(num_samples, 
                                                  device=self.device)
            hidden_values = self.task_pool[hidden_values_ids]
        else:
            if (self.total_trans > 0) and (mode not in ["ood", "length_ood"]):
                hidden_values_ids = torch.randint(high=self.total_trans, 
                                                  size=(num_samples,), 
                                                  device=self.device)
                hidden_values = self.task_pool[hidden_values_ids]
                length = self.repeat_length[hidden_values_ids]
            elif mode == "ood":
                hidden_values = torch.randint(high=self.num_states, 
                                              size=(num_samples, self.total_length), 
                                              device=self.device)
                length = self.total_length * torch.ones((num_samples,), device=self.device)
            elif mode == "length_ood":
                if length is None:
                    length = torch.randint(low=self.repeat_length_high+1, 
                                           high=2*self.repeat_length_high, 
                                           size=(num_samples,), 
                                           device=self.device)
                    max_length = length.max().item()
                else:
                    max_length = length
                    length = max_length * torch.ones((num_samples,), 
                                                     device=self.device)
                hidden_values = torch.randint(high=self.num_states, 
                                              size=(num_samples, max_length), 
                                              device=self.device)

        hidden_state = (torch.rand(num_samples) < self.repeat_prob).long().to(device=self.device)

        samples[:, :self.order] = state

        for t in range(self.order, self.seq_len):
            state_indices = torch.sum(state * self.powers, dim=1)
            probs = self.trans_matrix[state_indices]
            next_states = torch.multinomial(probs, num_samples=1).squeeze(1)

            use_markov = (hidden_state <= 0) 
            samples[use_markov, t] = next_states[use_markov]

            curr_values = hidden_values[~use_markov, self._get_index(hidden_state[~use_markov], length[~use_markov])] 

            if len(curr_values) > 0:
                samples[~use_markov, t] = curr_values 
                masks[~use_markov, t] = hidden_state[~use_markov]

            hidden_state = self._hidden_state_update(hidden_state, length)

            state[:, :-1] = state[:, 1:]
            state[:, -1] = next_states

        if mode == "train":
            return samples.reshape(epochs, -1, self.seq_len), masks.reshape(epochs, -1, self.seq_len)

        return samples, masks

    