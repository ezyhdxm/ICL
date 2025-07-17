import torch

class CoinTask:
    def __init__(self, config):
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.num_states = config.vocab_size
        self.seq_len = config.seq_len
        self.total_trans = config.task.total_trans
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.eval_size = config.eval_size
        self.device = config.device
        self.init_task_pool = config.task.init_task_pool if "init_task_pool" in config.task else None
        
        if self.total_trans > 0:
            if self.init_task_pool is not None:
                init_size = self.init_task_pool.shape[0]
                self.task_pool = torch.cat([self.init_task_pool.to(self.device), torch.rand(self.total_trans - init_size, device=self.device)], dim=0)

            else:
                self.task_pool = torch.rand(self.total_trans, device=self.device)

    def to(self, device):
        self.device = device
        self.task_pool = self.task_pool.to(device)

    def generate(self, epochs=1, mode="train", num_samples=None, task=None):
        if mode == "train":
            num_samples = num_samples if num_samples is not None else self.batch_size 
        elif mode == "test":
            num_samples = num_samples if num_samples is not None else self.test_size
        elif mode in ["eval", "ood"]:
            num_samples = num_samples if num_samples is not None else self.eval_size
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported modes are 'train', 'test', 'eval', and 'ood'.")
        num_samples *= epochs

        if task is None:
            if mode != "ood" and self.total_trans > 0:
                indices = torch.randint(high=self.total_trans, size=(num_samples,), device=self.device)
                probs = self.task_pool[indices]
            else:
                probs = torch.rand(num_samples, device=self.device)
        if task is not None:
            indices = task * torch.ones(num_samples, dtype=torch.long, device=self.device)
            probs = self.task_pool[indices]

        samples = (torch.rand(num_samples, self.seq_len, device=self.device) < probs[:, None]).long()
        info = torch.zeros((num_samples, self.seq_len), dtype=torch.long, device=self.device)

        if mode == "train":
            return samples.reshape(epochs, -1, self.seq_len), info.reshape(epochs, -1, self.seq_len)

        return samples, info



class CoinBayes:
    def __init__(self, config, sampler, ood=False):
        self.ood_flag = ood
        if not self.ood_flag:
            self.task_pool = sampler.task_pool
    
    def pos_prob(self, coinseq):
        cumsum = torch.cumsum(coinseq, dim=-1)
        B, T = cumsum.shape
        probs = torch.zeros((B, T, 2), dtype=torch.float32)
        t_vals = torch.arange(1, T+1, device=cumsum.device)  # shape (T,)

        if not self.ood_flag:
            p = self.task_pool
            p_expand = p.view(1, 1, -1)            # (1, 1, K)
            cumsum_expand = cumsum.unsqueeze(-1)  # (B, T, 1)
            t_expand = t_vals.view(1, T, 1)       # (1, T, 1)

            log_p = torch.log(p_expand)           # (1, 1, K)
            log_1mp = torch.log(1 - p_expand)     # (1, 1, K)

            num_log = (cumsum_expand + 1) * log_p + (t_expand - cumsum_expand) * log_1mp  # (B, T, K)
            den_log = cumsum_expand * log_p + (t_expand - cumsum_expand) * log_1mp # (B, T, K)
            num_logsumexp = torch.logsumexp(num_log, dim=-1)  # (B, T)
            den_logsumexp = torch.logsumexp(den_log, dim=-1)  # (B, T)

            probs[:, :, 1] = torch.exp(num_logsumexp - den_logsumexp)  # (B, T)
            probs[:, :, 0] = 1 - probs[:, :, 1]  # (B, T)
        
        else:
            probs[:, :, 1] = (cumsum + 1) / (t_vals + 2).float()
            probs[:, :, 0] = 1 - probs[:, :, 1]
        
        return probs