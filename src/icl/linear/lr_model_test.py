import torch
import torch.nn as nn
from icl.linear.lr_models import Mixture, Ridge
import pytest  # optional, for test runner

def test_mixture_model_random_forward():
    # Parameters
    batch_size = 4
    n_points = 10
    n_dims = 6
    n_tasks = 3
    tau = 0.5
    noise_scale = 1
    p0 = 0.1

    # Task pool: shape (n_tasks, n_dims, 1)
    task_pool = torch.randn(n_tasks, n_dims, 1)

    # Model
    model = Mixture(tau=tau, task_pool=task_pool, p0=p0, noise_scale=noise_scale)

    # Dummy input data
    data = torch.randn(batch_size, n_points, n_dims)
    targets = torch.randn(batch_size, n_points)

    # Forward pass
    preds = model(data, targets)

    # Check output shape
    assert preds.shape == (batch_size, n_points), f"Expected shape {(batch_size, n_points)}, got {preds.shape}"
    print("✅ Mixture model forward pass test passed.")


def test_mixture_model_in_distribution():
    # Parameters
    batch_size = 4
    n_points = 64
    n_dims = 6
    n_tasks = 3
    tau = 0.5
    noise_scale = 1
    p0 = 0.1

    # Task pool: shape (n_tasks, n_dims, 1)
    task_pool = torch.randn(n_tasks, n_dims, 1)

    # Model
    model = Mixture(tau=tau, task_pool=task_pool, p0=p0, noise_scale=noise_scale)

    # Dummy input data
    data = torch.randn(batch_size, n_points, n_dims)
    ws_inds = torch.randint(0, n_tasks, (batch_size,))  # Random task assignments
    ws = task_pool[ws_inds]  # Shape (batch_size, n_dims, 1)
    oracle = torch.bmm(data, ws).squeeze(-1)  # Shape (batch_size, n_points)
    targets = torch.bmm(data, ws).squeeze(-1) + noise_scale * torch.randn(batch_size, n_points)  # Shape (batch_size, n_points)

    # Forward pass
    preds = model(data, targets)

    # Check output shape
    torch.testing.assert_close(preds[:,-1], oracle[:,-1], rtol=1e-1, atol=1e-1)
    print("✅ Mixture model forward pass test passed.")


def test_mixture_model_out_of_distribution_ridge():
    # Parameters
    batch_size = 2
    n_points = 20
    n_dims = 6
    n_tasks = 2
    tau = 1
    noise_scale = 0.5
    p0 = 0.999

    # Task pool: shape (n_tasks, n_dims, 1)
    task_pool = torch.zeros(n_tasks, n_dims, 1)

    # Model
    model = Mixture(tau=tau, task_pool=task_pool, p0=p0, noise_scale=noise_scale)
    RE = Ridge(lam=tau**2)

    # Dummy input data
    data = torch.randn(batch_size, n_points, n_dims)
    ws = torch.randn(batch_size, n_dims, 1)  # Random task assignments
    targets = torch.bmm(data, ws).squeeze(-1) + noise_scale * torch.randn(batch_size, n_points)  # Shape (batch_size, n_points)

    # Forward pass
    preds = model(data, targets)
    ridge_preds = RE(data, targets)
    print(preds[:,-1], ridge_preds[:,-1], targets[:,-1])

    # Check output shape
    torch.testing.assert_close(preds, ridge_preds, rtol=1e-3, atol=1e-3)
    print("✅ Mixture model forward pass test passed.")

def test_mixture_model_out_of_distribution_final():
    # Parameters
    batch_size = 128
    n_points = 64
    n_dims = 6
    n_tasks = 3
    tau = 1
    noise_scale = 0.5
    p0 = 0.1

    # Task pool: shape (n_tasks, n_dims, 1)
    task_pool = torch.zeros(n_tasks, n_dims, 1)

    # Model
    model = Mixture(tau=tau, task_pool=task_pool, p0=p0, noise_scale=noise_scale)
    RE = Ridge(lam=tau**2)

    # Dummy input data
    data = torch.randn(batch_size, n_points, n_dims)
    ws = torch.randn(batch_size, n_dims, 1)  # Random task assignments
    targets = torch.bmm(data, ws).squeeze(-1) + noise_scale * torch.randn(batch_size, n_points)  # Shape (batch_size, n_points)

    # Forward pass
    preds = model(data, targets)
    ridge_preds = RE(data, targets)
    print(preds[:,-1], ridge_preds[:,-1], targets[:,-1])

    # Check output shape
    torch.testing.assert_close(preds[:,-1], ridge_preds[:,-1], rtol=1e-3, atol=1e-3)
    print("✅ Mixture model forward pass test passed.")

if __name__ == "__main__":
    test_mixture_model_out_of_distribution_ridge()
    print("All tests passed successfully.")