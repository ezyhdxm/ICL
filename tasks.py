import math
import torch

def squared_error(y_pred, y_true):
    return (y_pred - y_true).square()

def mean_squared_error(y_pred, y_true):
    return squared_error(y_pred, y_true).mean()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean()

sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()

def cross_entropy(y_pred, y_true):
    return bce_loss(sigmoid(y_pred), (y_true+1)/2)


def get_task_sampler(task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "linear_classification": LinearClassification,
        "sparse_linear_regression": SparseLinearRegression,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
    }
    if task_name in task_names_to_classes:
        task_class = task_names_to_classes[task_name]
        if num_tasks is not None:
            pool_dict = task_class.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_class(n_dims, batch_size, pool_dict=pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class Task:
    def __init_(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
    
    def evaluate(self, xs):
        raise NotImplementedError
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError
    
    @staticmethod
    def get_training_metric():
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, scale=1, seeds=None):
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        if pool_dict is None and seeds is None:
            self.w = torch.randn(self.batch_size, n_dims, 1)
        else:
            indices = torch.randperm(len(pool_dict["w"]))[:self.batch_size]
            self.w = pool_dict["w"][indices]
    
    def evaluate(self, xs): # xs: (batch_size, n_sample, n_dims)
        w = self.w.to(xs.device)
        ys = self.scale * (xs @ w).squeeze(-1)
        return ys
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "w": torch.randn(num_tasks, n_dims, 1),
        }
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error

class SparseLinearRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, scale=1, sparsity=3, valid_coords=None):
        super(SparseLinearRegression, self).__init__(n_dims, batch_size, pool_dict, scale)
        self.sparsity = sparsity
        if not valid_coords: valid_coords = n_dims
        for w in self.w:
            indices = torch.randperm(valid_coords)[:sparsity]
            w[indices] = 0
    
class LinearClassification(LinearRegression):
    
    def evaluate(self, xs):
        ys = super(LinearClassification, self).evaluate(xs)
        return ys.sign()
    
    @staticmethod
    def get_metric():
        return accuracy
    
    @staticmethod
    def get_training_metric():
        return cross_entropy

class NoisyLinearRegression(LinearRegression):
    def __init_(self, n_dims, batch_size, pool_dict=None, scale=1, noise_std=1):
        super().__init_(n_dims, batch_size, pool_dict, scale)
        self.noise_std = noise_std
    
    def evaluate(self, xs):
        ys = super().evaluate(xs)
        ys_noisy = ys + torch.randn_like(ys) * self.noise_std
        return ys_noisy

class QuadraticRegression(LinearRegression):
    def evaluate(self, xs):
        w = self.w.to(xs.device)
        ys_q = self.scale * ((xs**2)@w).squeeze(-1) / math.sqrt(3)
        return ys_q


