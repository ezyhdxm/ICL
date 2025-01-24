import math
import torch

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims
    
    def sample(self):
        raise NotImplementedError

class GaussianDataSampler(DataSampler):
    def __init__(self, n_dims, mean=None, std=None):
        super(GaussianDataSampler, self).__init__(n_dims)
        self.mean = mean if mean is not None else torch.zeros(n_dims)
        self.std = std if std is not None else torch.ones(n_dims)
    
    def sample(self, n_samples, batch_size, n_dims_truncated=None):
        xs = torch.normal(mean=self.mean, std=self.std, size=(batch_size, n_samples, self.n_dims))
        if n_dims_truncated is not None:
            xs[:,:,n_dims_truncated:] = 0
        return xs