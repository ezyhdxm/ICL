import torch
from typing import Optional, List, Tuple
from config import CausalGraphConfig
import numpy as np

def dag_to_adj(dag, task_name="tree"):
    seq_len = len(dag)
    adj_mat = torch.zeros((seq_len, seq_len), dtype=torch.float)
    if task_name == "tree":
        py_dag = torch.tensor(dag)
        idx = torch.where(py_dag >= 0)[0]  # Indices where dag[i] >= 0 (has a parent)
        adj_mat[idx, py_dag[idx]] = 1  # Set adjacency matrix entries to 1
    else:
        for i in range(seq_len):
            if len(dag[i]) > 0:
                adj_mat[i, dag[i]] = 1 
    return adj_mat

def get_random_DAG(seq_len, p=0.5):
    # Generate random parent indices (excluding last node)
    random_parents = np.random.randint(np.zeros(seq_len-2), np.arange(1, seq_len - 1))  # Shape: (seq_len - 2,)
    
    # Insert -1 as the first parent
    dag = np.insert(random_parents, 0, -1)  # Shape: (seq_len - 1,)

    # Generate a mask: 1 with probability p, 0 otherwise
    mask = np.random.choice([True, False], size=seq_len-1, p=[p, 1 - p])

    # Apply the mask: set to -1 where mask == 1
    dag = np.where(mask, -1, dag)

    return dag

class InContextTreeTorch:
    def __init__(self, config:CausalGraphConfig)->None:
        self.dag = config.dag
        assert np.all(self.dag < np.arange(len(self.dag))), "Invalid DAG structure"
        self.vocab_size = config.vocab_size
        self.alpha = config.alpha.to(config.device)
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
        self.seed = config.seed

    def get_stationary(self, pi: torch.Tensor)->torch.Tensor:
        """
        Compute the stationary distribution of a batch of transition matrices. Cannot be jitted due to dynamic tensor shape
        Args:
            pi (torch.Tensor): Transition matrices of shape (batch_size, vocab_size, vocab_size).
        Returns:
            torch.Tensor: Stationary distributions of shape (batch_size, vocab_size).
        """
        pi_t = pi.transpose(1, 2)  # Transpose each matrix
        svd_input = pi_t - torch.eye(self.vocab_size, device=self.device).unsqueeze(0)
        _, _, v = torch.linalg.svd(svd_input)
        mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix
        return mu / mu.sum(dim=1, keepdim=True)

    def generate(self, mode="train")->Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of sequences from the InContextTree.
        Args:
            batch_size (int): Number of sequences to sample.
            seed (int, optional): Random seed for reproducibility.
        Returns:
            torch.Tensor: Sampled sequences of shape (batch_size, len(dag) + 1).
            torch.Tensor: Probabilities associated with the last token of each sequence. (batch_size, vocab_size).
        """
        # Set random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)

        num_samples = self.batch_size if mode == "train" else self.test_size

        # Create prior and sample Dirichlet-distributed transition matrices
        prior = self.alpha * torch.ones(self.vocab_size, device=self.device)
        dirichlet = torch.distributions.Dirichlet(prior)
        pi = dirichlet.sample((num_samples, self.vocab_size))  # Shape: (batch_size, vocab_size, vocab_size)
        pi /= pi.sum(dim=-1, keepdim=True)  # Normalize to make it a valid transition matrix

        # Compute stationary distributions for the batch
        mu = self.get_stationary(pi)

        # Initialize sequences (batch_size, sequence_length)
        seq_len = len(self.dag) + 1
        samples = torch.zeros((num_samples, seq_len), dtype=torch.long, device=self.device)

        # Iterate over the DAG nodes to sample tokens
        for i in range(len(self.dag)):
            if self.dag[i] == -1:  # Root node
                p = mu  # Use stationary distribution
            else:  # Child node
                parent_tokens = samples[:, self.dag[i]]  # Shape: (batch_size,)
                p = pi[torch.arange(num_samples), parent_tokens]  # Transition probabilities for parent tokens

            # Sample tokens for all sequences in the batch
            samples[:, i] = torch.multinomial(p, num_samples=1).squeeze()

        # Sample test tokens for the last position
        test_tokens = torch.randint(self.vocab_size, (num_samples,), device=self.device)
        samples[:, -1] = test_tokens
        y = pi[torch.arange(num_samples), test_tokens]  # Probabilities of test tokens

        return samples, y
    
    def bayes(self, samples): # samples: (B, seq_len)
        num_samples, seq_len = samples.shape
        test_tokens, seq = samples[:,-1], samples[:,:-1] # (B,), (B,seq_len-1)
        counts = torch.zeros((num_samples, self.vocab_size), device=self.device)
        for i in range(1, seq_len-1):
            parent_index = self.dag[i]
            if parent_index == -1:
                continue
            indices = (seq[:,parent_index] == test_tokens).nonzero(as_tuple=True)[0]
            values = torch.ones_like(indices, dtype=torch.float, device=self.device) 
            counts.index_put_((indices, seq[indices,i]), values, accumulate=True)
        counts += self.alpha
        return counts / counts.sum(dim=-1, keepdim=True)


class InContextDAGTorch:
    def __init__(self, config: CausalGraphConfig)->None:
        for i, p in enumerate(config.dag):
            assert max(p, default=-1) < i, "Invalid DAG structure"
        self.vocab_size = config.vocab_size
        self.dag = config.dag
        self.alpha = config.alpha
        self.num_parents = set(len(p) for p in self.dag)
        self.batch_size = config.batch_size
        self.test_size = config.test_size
        self.device = config.device
    
    def generate(self, mode: str="train")->Tuple[torch.Tensor, torch.Tensor]:
        num_samples = self.batch_size if mode == "train" else self.test_size
        pi = {}
        pi[0] = torch.ones((num_samples, self.vocab_size), device=self.device) / self.vocab_size # uniform initialization
        prior = self.alpha * torch.ones(self.vocab_size)
        dirichlet = torch.distributions.Dirichlet(prior)
        for k in self.num_parents:
            if k == 0:
                continue
            num_states_order = self.vocab_size ** k
            pi[k] = dirichlet.sample((num_samples, num_states_order)).to(self.device) # Shape: (batch_size, vocab_size**k, vocab_size)
            pi[k] /= pi[k].sum(dim=-1, keepdim=True)

        samples = torch.zeros((num_samples, len(self.dag)-1), dtype=torch.long).to(self.device)

        for i in range(len(self.dag)):
            k = len(self.dag[i])
            if k == 0:
                p = pi[0]
            else:
                parents = samples[:, self.dag[i]]
                parents_indices = torch.sum(parents * (self.vocab_size ** torch.arange(k-1, -1, -1, device=self.device)), dim=1)
                p = pi[k][torch.arange(num_samples), parents_indices]
            
            if i != len(self.dag)-1:
                samples[:, i] = torch.multinomial(p, num_samples=1).squeeze()
        
        return samples, p

    def bayes(self, samples): # samples: (B, seq_len)
        num_samples, seq_len = samples.shape
        last_parents = samples[:,self.dag[-1]]
        k = last_parents.shape[1]
        counts = torch.zeros((num_samples, self.vocab_size), device=self.device)
        for i in range(1, seq_len-1):
            parent_indexes = self.dag[i]
            if len(parent_indexes) == k:
                indices = (samples[:,parent_indexes] == last_parents).all(dim=-1).nonzero(as_tuple=True)[0]
                if len(indices) == 0:
                    continue
                values = torch.ones_like(indices, dtype=torch.float, device=self.device) 
                counts.index_put_((indices, samples[indices,i]), values, accumulate=True)
        counts += self.alpha
        return counts / counts.sum(dim=-1, keepdim=True)
