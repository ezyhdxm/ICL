import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple
from config import *
from linear.lr_transformer import GPT2Model, GPT2Config
from linear.lr_utils import to_seq, seq_to_targets

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, TransformerLin):
        return "Transformer"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################



class TransformerLin(nn.Module):
    def __init__(self, n_dims: int, n_points: int, n_layer: int, n_embd: int, n_head: int, seed: int, dtype: Any):
        super().__init__()
        self.n_points = n_points
        self.dtype = dtype
        self.input_dim = n_dims+1

        # GPT-style config (assuming your custom GPT2Model/GPT2Config implementation)
        config = GPT2Config(
            block_size=2 * n_points,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dtype=dtype,
            device= "cuda" if torch.cuda.is_available() else "cpu",
        )

        torch.manual_seed(seed)

        self.input_proj = nn.Linear(self.input_dim, config.n_embd, bias=False).to(device=config.device)
        self.transformer = GPT2Model(config)
        self.output_proj = nn.Linear(config.n_embd, 1, bias=False).to(device=config.device)
        self.device = config.device

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert input into sequential format
        input_seq = to_seq(data, targets).to(device=self.device)  # shape: (batch, seq_len=2*n_points, input_dim)
        embds = self.input_proj(input_seq)  # shape: (batch, seq_len, n_embd)
        outputs = self.transformer(embds)  # shape: (batch, seq_len, n_embd)
        preds = self.output_proj(outputs)  # shape: (batch, seq_len, 1)
        preds = seq_to_targets(preds)  # shape: (batch, n_points)
        return preds


########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################

class Ridge(nn.Module):
    def __init__(self, lam: float, dtype=torch.float32):
        super().__init__()
        self.lam = lam
        self.dtype = dtype
    
    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        if data.ndim == 4:
            data = data.squeeze(0)
        batch_size, n_points, _ = data.shape
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        targets = targets.unsqueeze(-1)
        preds = [torch.zeros(batch_size, dtype=self.dtype, device=data.device)]
        for i in range(1, n_points):
            pred_i = self.predict(
                data[:, :i],          # X: (batch, i, dim)
                targets[:, :i],       # Y: (batch, i, 1)
                data[:, i:i+1],       # test_x: (batch, 1, dim)
                self.lam
            )
            preds.append(pred_i)
        preds = torch.stack(preds, dim=1)  # (batch_size, n_points)
        return preds
    
    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Args:
            X: (batch_size, i, n_dims)
            Y: (batch_size, i, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            (batch_size,)
        """
        batch_size, i, n_dims = X.shape

        XT = X.transpose(1, 2)                          # (batch_size, n_dims, i)
        XT_Y = torch.bmm(XT, Y)                         # (batch_size, n_dims, 1)

        # Ridge matrix: X^T X + λI
        eye = torch.eye(n_dims, dtype=self.dtype, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
        ridge_matrix = torch.bmm(XT, X) + lam * eye     # (batch_size, n_dims, n_dims)

        # Solve (XT X + λI) w = XT Y
        ws = torch.linalg.solve(ridge_matrix, XT_Y)     # (batch_size, n_dims, 1)

        # Predict: test_x @ w
        pred = torch.bmm(test_x, ws)                    # (batch_size, 1, 1)
        return pred[:, 0, 0]          

########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    def __init__(self, scale: float, task_pool: torch.Tensor, dtype=torch.float32):
        """
        Args:
            scale: noise std
            task_pool: Tensor of shape (n_tasks, n_dims, 1)
        """
        super().__init__()
        self.scale = scale
        self.dtype = dtype

        assert task_pool.ndim == 3 and task_pool.shape[2] == 1
        self.task_pool = task_pool.to(dtype)

        # Preprocess: squeeze and transpose for use in prediction
        self.W = task_pool.squeeze(-1).T  # shape: (n_dims, n_tasks)

    def forward(self, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: (batch_size, n_points, n_dims)
            targets: (batch_size, n_points)
        Returns:
            preds: (batch_size, n_points)
        """
        if data.ndim == 4:
            data = data.squeeze(0)
        if targets.ndim == 3:
            targets = targets.squeeze(0)
        batch_size, n_points, _ = data.shape
        targets = targets.unsqueeze(-1)  # (batch_size, n_points, 1)

        # Initial prediction with mean of W
        w_mean = self.W.mean(dim=1)  # (n_dims,)
        preds = [torch.matmul(data[:, 0], w_mean)]  # list of (batch_size,)

        # Iterative MMSE predictions
        for i in range(1, n_points):
            pred_i = self.predict(data[:, :i], targets[:, :i], data[:, i:i+1])
            preds.append(pred_i)

        return torch.stack(preds, dim=1)  # (batch_size, n_points)

    def predict(self, X: torch.Tensor, Y: torch.Tensor, test_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch_size, i, n_dims)
            Y: (batch_size, i, 1)
            test_x: (batch_size, 1, n_dims)
        Returns:
            pred: (batch_size,)
        """
        # shape: (batch_size, i, n_tasks)
        XW = torch.matmul(X, self.W)  # broadcasting matmul

        # shape: (batch_size, i, n_tasks)
        diff = Y - XW

        # Compute log-likelihood under Gaussian noise model
        log_probs = -0.5 * ((diff / self.scale) ** 2 + 2 * torch.log(torch.tensor(self.scale)))  # log N(·; 0, scale^2)
        alpha = log_probs.sum(dim=1)  # (batch_size, n_tasks)

        # Softmax weights over task pool
        weights = F.softmax(alpha, dim=1)  # (batch_size, n_tasks)

        # Compute MMSE weights: (batch_size, n_dims)
        w_mmse = torch.bmm(weights.unsqueeze(1), self.W.T.unsqueeze(0).expand(X.size(0), -1, -1))  # (batch, 1, n_dims)
        w_mmse = w_mmse.transpose(1, 2)  # (batch_size, n_dims, 1)

        # Predict: test_x @ w_mmse → (batch, 1, 1)
        pred = torch.bmm(test_x, w_mmse)
        return pred[:, 0, 0]  # (batch_size,)

Model = Ridge | DiscreteMMSE

def get_model(name: str, **kwargs) -> Model:
    models = {"ridge": Ridge, "discrete_mmse": DiscreteMMSE, "transformer": TransformerLin}
    return models[name](**kwargs)