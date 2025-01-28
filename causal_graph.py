import torch

class InContextTreeTorch:
    def __init__(self, vocab_size, dag, alpha=0.1):
        assert torch.all(dag < torch.arange(len(dag))), "Invalid DAG structure"
        self.vocab_size = vocab_size
        self.dag = dag
        self.alpha = alpha

    def get_stationary(self, pi):
        """
        Compute the stationary distribution of a batch of transition matrices. Cannot be jitted due to dynamic tensor shape
        Args:
            pi (torch.Tensor): Transition matrices of shape (batch_size, vocab_size, vocab_size).
        Returns:
            torch.Tensor: Stationary distributions of shape (batch_size, vocab_size).
        """
        batch_size, vocab_size, _ = pi.shape
        pi_t = pi.transpose(1, 2)  # Transpose each matrix
        svd_input = pi_t - torch.eye(vocab_size, device=pi.device).unsqueeze(0)
        _, _, v = torch.linalg.svd(svd_input)
        mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix
        return mu / mu.sum(dim=1, keepdim=True)

    def sample_batch(self, batch_size, seed=None):
        """
        Sample a batch of sequences from the InContextTree.
        Args:
            batch_size (int): Number of sequences to sample.
            seed (int, optional): Random seed for reproducibility.
        Returns:
            torch.Tensor: Sampled sequences of shape (batch_size, len(dag) + 1).
            torch.Tensor: Probabilities associated with the last token of each sequence.
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Create prior and sample Dirichlet-distributed transition matrices
        prior = self.alpha * torch.ones(self.vocab_size, device=self.dag.device)
        dirichlet = torch.distributions.Dirichlet(prior)
        pi = dirichlet.sample((batch_size, self.vocab_size))  # Shape: (batch_size, vocab_size, vocab_size)
        pi /= pi.sum(dim=-1, keepdim=True)  # Normalize to make it a valid transition matrix

        # Compute stationary distributions for the batch
        mu = self.get_stationary(pi)

        # Initialize sequences (batch_size, sequence_length)
        seq_len = len(self.dag) + 1
        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.dag.device)

        # Iterate over the DAG nodes to sample tokens
        for i in range(len(self.dag)):
            if self.dag[i] == -1:  # Root node
                p = mu  # Use stationary distribution
            else:  # Child node
                parent_tokens = x[:, self.dag[i]]  # Shape: (batch_size,)
                p = pi[torch.arange(batch_size), parent_tokens]  # Transition probabilities for parent tokens

            # Sample tokens for all sequences in the batch
            x[:, i] = torch.multinomial(p, num_samples=1).squeeze()

        # Sample test tokens for the last position
        test_tokens = torch.randint(self.vocab_size, (batch_size,), device=self.dag.device)
        x[:, -1] = test_tokens
        y = pi[torch.arange(batch_size), test_tokens]  # Probabilities of test tokens

        return x, y

