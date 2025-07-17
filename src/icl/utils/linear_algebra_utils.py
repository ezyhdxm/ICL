import numpy as np
import torch
from torch import nn

########################
# Linear Algebra Utils #
#########################

def effective_rank(A):
    # See https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-12))  # Add small epsilon to avoid log(0)
    return np.exp(entropy)

def stable_rank(A):
    frob_norm = np.linalg.norm(A, ord='fro')
    op_norm = np.linalg.norm(A, ord=2)  # spectral norm
    return (frob_norm ** 2) / (op_norm ** 2)


def get_stationary(P: torch.Tensor)->torch.Tensor:
    if P.ndim == 2:
        P = P.unsqueeze(0)
    assert P.ndim == 3, "P should be a 3D tensor"
    P = P.transpose(1, 2)  # Transpose each matrix, Shape: (num_samples, num_states, num_states_order)
    num_states = P.shape[1]
    svd_input = P - torch.eye(num_states, device=P.device).unsqueeze(0)
    _, _, v = torch.linalg.svd(svd_input)
    mu = torch.abs(v[:, -1, :])  # Last singular vector for each matrix, Shape: (num_samples, num_states)
    mu = mu / mu.sum(dim=-1, keepdim=True)  # Normalize
    if mu.size(0) == 1:
        mu = mu.squeeze(0)
    return mu