import torch
import torch.nn as nn
import dataclasses
from typing import Optional, Tuple, Any, List, Callable

from icl.linear.lr_models import get_model

# Adapted from https://github.com/mansheej/icl-task-diversity/blob/main/icl/tasks.py

Sampler = Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

@dataclasses.dataclass
class SingleIndexRegression:
    """
    Single-index regression model: y = f(w^T x) + noise
    where f is a nonlinear link function.
    """
    n_tasks: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    link_function: str = 'relu'  # Link function type
    link_scale: float = 1.0  # Scale factor for link function output
    dtype: Any = torch.float32

    def __post_init__(self):
        self.data_gen = torch.Generator().manual_seed(self.data_seed)
        self.task_gen = torch.Generator().manual_seed(self.task_seed)
        self.noise_gen = torch.Generator().manual_seed(self.noise_seed)
        self.task_pool: Optional[torch.Tensor] = self.generate_task_pool() if self.n_tasks > 0 else None
        
        # Set up link function
        self.link_fn = self._get_link_function()

    def _get_link_function(self) -> Callable:
        """Return the link function based on the specified type."""
        link_functions = {
            'identity': lambda x: x,
            'relu': torch.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'square': lambda x: x ** 2,
            'abs': torch.abs,
            'sin': torch.sin,
            'cos': torch.cos,
            'leaky_relu': torch.nn.functional.leaky_relu,
            'elu': torch.nn.functional.elu,
            'softplus': torch.nn.functional.softplus,
            'exp': torch.exp,
            'log1p': torch.log1p,  # log(1 + x)
            'cube': lambda x: x ** 3,
            'sign': torch.sign,
            'arctan': torch.arctan,
            # Highly nonlinear functions that break linear structure
            'step': lambda x: (x > 0).float(),  # Heaviside step function
            'periodic': lambda x: torch.sin(3 * x) + 0.5 * torch.cos(5 * x),  # Multi-frequency periodic
            'bump': lambda x: torch.exp(-1 / (1 - x**2).clamp(min=1e-7)) * ((x > -1) & (x < 1)).float(),  # Smooth bump
            'sawtooth': lambda x: x - torch.floor(x + 0.5),  # Sawtooth wave
            'gaussian': lambda x: torch.exp(-x**2),  # Gaussian
            'laplace': lambda x: torch.exp(-torch.abs(x)),  # Laplace distribution shape
            'cauchy': lambda x: 1 / (1 + x**2),  # Cauchy/Lorentzian
            'sinc': lambda x: torch.sinc(x / torch.pi),  # Normalized sinc function
            'chirp': lambda x: torch.sin(x + x**2) * torch.exp(-x**2 / 10),  # Frequency-modulated sine
            'rational': lambda x: x / (1 + torch.abs(x)),  # Rational function
            'hermite': lambda x: (2 * x**2 - 1) * torch.exp(-x**2),  # Hermite polynomial
            'bessel_like': lambda x: torch.sin(x) / (torch.abs(x) + 1e-7),  # Bessel-like oscillating
        }
        
        if self.link_function not in link_functions:
            raise ValueError(f"Unknown link function: {self.link_function}. "
                           f"Available options: {list(link_functions.keys())}")
        
        return link_functions[self.link_function]

    @property
    def name(self) -> str:
        return f"SingleIdxReg({self.n_tasks})_{self.link_function}"
    
    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "SingleIndexRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task
    
    def generate_task_pool(self) -> torch.Tensor:
        # generate a pool of index vectors w1, w2, ..., wN, where N = n_tasks
        # w_i ~ N(0, task_scale^2 * I), where I is the identity matrix of size D = n_dims
        shape = (self.n_tasks, self.n_dims, 1)
        return torch.randn(shape, generator=self.task_gen, dtype=self.dtype) * self.task_scale

    def sample_data(self, step: int) -> torch.Tensor:
        # generate a batch of data points x1, x2, ..., xN, where N = n_points
        # x_i ~ N(0, data_scale^2 * I), where I is the identity matrix of size D = n_dims
        gen = torch.Generator().manual_seed(self.data_seed + step)
        shape = (self.batch_size, self.n_points, self.n_dims)
        return torch.randn(shape, generator=gen, dtype=self.dtype) * self.data_scale
    
    def sample_tasks(self, step: int) -> torch.Tensor:
        # sample a batch of index vectors w1, w2, ..., wB from the task pool, where B = batch_size
        gen = torch.Generator().manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(low=0, high=self.n_tasks, size=(self.batch_size,), generator=gen)
            tasks = self.task_pool[idxs]
        else:
            # infinite task pool
            shape = (self.batch_size, self.n_dims, 1)
            tasks = torch.randn(shape, generator=gen, dtype=self.dtype) * self.task_scale
        return tasks
    
    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        # data: (batch, n_points, n_dims)
        # tasks: (batch, n_dims, 1)
        
        # Compute linear index: w^T x
        linear_index = (data @ tasks).squeeze(-1)  # (batch, n_points)
        
        # Apply link function: f(w^T x)
        targets = self.link_fn(linear_index) * self.link_scale
        
        # Add noise
        gen = torch.Generator().manual_seed(self.noise_seed + step)
        noise = torch.randn(targets.shape, dtype=targets.dtype, device=targets.device, generator=gen) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.sample_data(step)
        tasks = self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)
        return data, tasks, targets
    
    def sample_from_task(self, task: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of data and noisy targets for a specific given task.

        Args:
            task: Tensor of shape [n_dims, 1] or [1, n_dims, 1].
            step: Integer step for deterministic sampling.

        Returns:
            data: [batch_size, n_points, n_dims]
            targets: [batch_size, n_points]
        """
        data = self.sample_data(step)  # [B, T, D]
        if task.ndim == 2:
            task = task.unsqueeze(0)  # [1, D, 1]
        assert task.shape == (1, self.n_dims, 1), f"Task shape should be [1, {self.n_dims}, 1], got {task.shape}"

        tasks = task.expand(self.batch_size, -1, -1)  # [B, D, 1]
        targets = self.evaluate(data, tasks, step)    # [B, T]
        return data, targets

    def evaluate_oracle(self, data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        """Oracle evaluation that applies link function without noise."""
        linear_index = (data @ tasks).squeeze(-1)
        return self.link_fn(linear_index) * self.link_scale

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> List["SingleIndexRegression"]:
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config.update(dict(
            batch_size=batch_size,
            task_seed=task_seed,
            data_seed=data_seed,
            noise_seed=noise_seed,
            n_tasks=0,
        ))
        eval_tasks = [SingleIndexRegression(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(SingleIndexRegression.from_task_pool(task_pool=self.task_pool.clone(), **config))
        return eval_tasks

    def get_default_eval_models(self) -> List[Any]:
        # For single-index models, ridge regression might not be optimal
        # You may want to implement specific models for single-index regression
        models = []
        if self.link_function == 'identity':
            # For identity link, ridge regression is appropriate
            models.append(get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype))
        else:
            # For nonlinear links, you might need specialized models
            # This is a placeholder - you'll need to implement appropriate models
            pass
        
        if self.n_tasks > 0 and self.link_function == 'identity':
            assert self.task_scale == 1.0  # TODO
            models.append(
                get_model(
                    name="discrete_mmse", scale=self.noise_scale, task_pool=self.task_pool.clone(), dtype=self.dtype
                )
            )
        return models


########################################################################################################################
# Get Task                                                                                                             #
########################################################################################################################

Task = SingleIndexRegression  # Default task type


def get_task(name: str, **kwargs) -> Any:
    tasks = {
        "single_index_regression": SingleIndexRegression,
    }
    return tasks[name](**kwargs)

def get_task_name(task: Any) -> str:
    if hasattr(task, 'name'):
        if task.name.endswith("(0)"):
            return "Latent"
        else:
            return "Pretrain"
    return "Unknown"
