import torch
import torch.nn as nn
import dataclasses
from typing import Optional, Tuple, Any, List, Callable, Union

from icl.linear.lr_models import get_model


@dataclasses.dataclass
class DiscreteInputLinearRegression:
    """
    Linear regression with discrete inputs from a finite set of vectors in R^6.
    """
    n_tasks: int
    n_discrete_vectors: int  # Size of the discrete set
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any = torch.float32
    n_dims: int = 6

    def __post_init__(self):
        self.data_gen = torch.Generator().manual_seed(self.data_seed)
        self.task_gen = torch.Generator().manual_seed(self.task_seed)
        self.noise_gen = torch.Generator().manual_seed(self.noise_seed)
        
        # Generate the discrete set of vectors
        self.discrete_vectors = self._generate_discrete_vectors()
        
        # Generate task pool
        self.task_pool: Optional[torch.Tensor] = self.generate_task_pool() if self.n_tasks > 0 else None

    @property
    def name(self) -> str:
        return f"DiscreteInputLinReg({self.n_tasks})"
    
    def _generate_discrete_vectors(self) -> torch.Tensor:
        """Generate a fixed set of discrete vectors in R^6."""
        shape = (self.n_discrete_vectors, self.n_dims)
        return torch.randn(shape, generator=self.data_gen, dtype=self.dtype) * self.data_scale
    
    @classmethod
    def from_task_pool(cls, task_pool: torch.Tensor, **kwargs) -> "DiscreteInputLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task
    
    def generate_task_pool(self) -> torch.Tensor:
        shape = (self.n_tasks, self.n_dims, 1)
        return torch.randn(shape, generator=self.task_gen, dtype=self.dtype) * self.task_scale

    def sample_data(self, step: int) -> torch.Tensor:
        """
        Sample data points from the discrete set of vectors.
        """
        gen = torch.Generator().manual_seed(self.data_seed + step)
        
        # Sample indices for each data point
        indices = torch.randint(
            low=0, 
            high=self.n_discrete_vectors, 
            size=(self.batch_size, self.n_points), 
            generator=gen
        )
        
        # Select vectors from the discrete set
        data = self.discrete_vectors[indices]
        
        return data
    
    def sample_tasks(self, step: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(low=0, high=self.n_tasks, size=(self.batch_size,), generator=gen)
            tasks = self.task_pool[idxs]
        else:
            shape = (self.batch_size, self.n_dims, 1)
            tasks = torch.randn(shape, generator=gen, dtype=self.dtype) * self.task_scale
        return tasks
    
    def evaluate(self, data: torch.Tensor, tasks: torch.Tensor, step: int) -> torch.Tensor:
        targets = (data @ tasks).squeeze(-1)
        gen = torch.Generator().manual_seed(self.noise_seed + step)
        noise = torch.randn(targets.shape, dtype=targets.dtype, device=targets.device, generator=gen) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.sample_data(step)
        tasks = self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)
        return data, tasks, targets
    
    def sample_from_task(self, task: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.sample_data(step)
        if task.ndim == 2:
            task = task.unsqueeze(0)
        assert task.shape == (1, self.n_dims, 1), f"Task shape should be [1, {self.n_dims}, 1], got {task.shape}"

        tasks = task.expand(self.batch_size, -1, -1)
        targets = self.evaluate(data, tasks, step)
        return data, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        return (data @ tasks).squeeze(-1)

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> List["DiscreteInputLinearRegression"]:
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
        eval_tasks = [DiscreteInputLinearRegression(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(DiscreteInputLinearRegression.from_task_pool(task_pool=self.task_pool.clone(), **config))
        return eval_tasks

    def get_default_eval_models(self) -> List[Any]:
        models = [get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
        if self.n_tasks > 0:
            assert self.task_scale == 1.0
            models.append(
                get_model(
                    name="discrete_mmse", scale=self.noise_scale, task_pool=self.task_pool.clone(), dtype=self.dtype
                )
            )
        return models


@dataclasses.dataclass
class MixtureOfGaussiansRegression:
    """
    Linear regression with inputs from a mixture of two Gaussian distributions in R^6.
    Each component has its own centroid and task-specific weight vectors.
    """
    n_tasks: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float  # Scale for the Gaussian components
    task_scale: float
    noise_scale: float
    mixture_prob: float = 0.5  # Probability of selecting first component
    centroid1: Optional[torch.Tensor] = None  # Centroid for first Gaussian
    centroid2: Optional[torch.Tensor] = None  # Centroid for second Gaussian
    dtype: Any = torch.float32
    n_dims: int = 6  # Fixed to R^6 as specified

    def __post_init__(self):
        self.data_gen = torch.Generator().manual_seed(self.data_seed)
        self.task_gen = torch.Generator().manual_seed(self.task_seed)
        self.noise_gen = torch.Generator().manual_seed(self.noise_seed)
        
        # Initialize centroids if not provided
        if self.centroid1 is None:
            self.centroid1 = torch.randn(self.n_dims, generator=self.data_gen, dtype=self.dtype)
        if self.centroid2 is None:
            self.centroid2 = torch.randn(self.n_dims, generator=self.data_gen, dtype=self.dtype)
        
        # Generate task pools for both components
        self.task_pool1: Optional[torch.Tensor] = None
        self.task_pool2: Optional[torch.Tensor] = None
        if self.n_tasks > 0:
            self.task_pool1, self.task_pool2 = self.generate_task_pools()

    @property
    def name(self) -> str:
        return f"MixtureGaussianReg({self.n_tasks})"
    
    @classmethod
    def from_task_pools(cls, task_pool1: torch.Tensor, task_pool2: torch.Tensor, **kwargs) -> "MixtureOfGaussiansRegression":
        assert kwargs["n_tasks"] == task_pool1.shape[0] == task_pool2.shape[0]
        task = cls(**kwargs)
        task.task_pool1 = task_pool1
        task.task_pool2 = task_pool2
        return task
    
    def generate_task_pools(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (self.n_tasks, self.n_dims, 1)
        # Different weight vectors for each component
        pool1 = torch.randn(shape, generator=self.task_gen, dtype=self.dtype) * self.task_scale
        pool2 = torch.randn(shape, generator=self.task_gen, dtype=self.dtype) * self.task_scale
        return pool1, pool2

    def sample_data(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mixture of Gaussian data and return both data and component labels.
        Returns:
            data: (batch_size, n_points, n_dims)
            component_labels: (batch_size, n_points) - 0 for component 1, 1 for component 2
        """
        gen = torch.Generator().manual_seed(self.data_seed + step)
        shape = (self.batch_size, self.n_points, self.n_dims)
        
        # Sample mixture components for each data point
        component_labels = (torch.rand(self.batch_size, self.n_points, generator=gen) >= self.mixture_prob).long()
        
        # Generate Gaussian noise
        gaussian_noise = torch.randn(shape, generator=gen, dtype=self.dtype) * self.data_scale
        
        # Initialize data tensor
        data = torch.zeros(shape, dtype=self.dtype)
        
        # Apply mixture logic
        mask_comp1 = (component_labels == 0).unsqueeze(-1)
        mask_comp2 = (component_labels == 1).unsqueeze(-1)
        
        data[mask_comp1.expand_as(data)] = (self.centroid1 + gaussian_noise)[mask_comp1.expand_as(data)]
        data[mask_comp2.expand_as(data)] = (self.centroid2 + gaussian_noise)[mask_comp2.expand_as(data)]
        
        return data, component_labels
    
    def sample_tasks(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tasks for both components.
        Returns:
            tasks1: (batch_size, n_dims, 1) - weight vectors for component 1
            tasks2: (batch_size, n_dims, 1) - weight vectors for component 2
        """
        gen = torch.Generator().manual_seed(self.task_seed + step)
        if self.n_tasks > 0:
            idxs = torch.randint(low=0, high=self.n_tasks, size=(self.batch_size,), generator=gen)
            tasks1 = self.task_pool1[idxs]
            tasks2 = self.task_pool2[idxs]
        else:
            shape = (self.batch_size, self.n_dims, 1)
            tasks1 = torch.randn(shape, generator=gen, dtype=self.dtype) * self.task_scale
            tasks2 = torch.randn(shape, generator=gen, dtype=self.dtype) * self.task_scale
        return tasks1, tasks2
    
    def evaluate(self, data: torch.Tensor, tasks1: torch.Tensor, tasks2: torch.Tensor, 
                 component_labels: torch.Tensor, step: int) -> torch.Tensor:
        """
        Evaluate targets using component-specific weight vectors.
        """
        B, T, D = data.shape

        # Expand tasks to match data: (B, T, D)
        tasks1_exp = tasks1.squeeze(-1)[:, None, :].expand(-1, T, -1)
        tasks2_exp = tasks2.squeeze(-1)[:, None, :].expand(-1, T, -1)

        # Select which task vector to use for each element
        tasks = torch.where(component_labels[..., None] == 0, tasks1_exp, tasks2_exp)

        # Compute dot product: (B, T, D) ⋅ (B, T, D) → (B, T)
        targets = (data * tasks).sum(dim=-1).to(dtype=self.dtype)
        
        # Add noise
        gen = torch.Generator().manual_seed(self.noise_seed + step)
        noise = torch.randn(targets.shape, dtype=targets.dtype, device=targets.device, generator=gen) * self.noise_scale
        return targets + noise

    def sample_batch(self, step: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns:
            data: (batch_size, n_points, n_dims)
            tasks: Tuple of (tasks1, tasks2) each of shape (batch_size, n_dims, 1)
            targets: (batch_size, n_points)
        """
        data, component_labels = self.sample_data(step)
        tasks1, tasks2 = self.sample_tasks(step)
        targets = self.evaluate(data, tasks1, tasks2, component_labels, step)
        return (data, component_labels), (tasks1, tasks2), targets
    
    def sample_from_task(self, task1: torch.Tensor, task2: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, component_labels = self.sample_data(step)
        if task1.ndim == 2:
            task1 = task1.unsqueeze(0)
            task2 = task2.unsqueeze(0)
        assert task1.shape == (1, self.n_dims, 1), f"Task shape should be [1, {self.n_dims}, 1], got {task1.shape}"
        assert task2.shape == (1, self.n_dims, 1), f"Task shape should be [1, {self.n_dims}, 1], got {task2.shape}"

        tasks1 = task1.expand(self.batch_size, -1, -1)
        tasks2 = task2.expand(self.batch_size, -1, -1)
        targets = self.evaluate(data, tasks1, tasks2, component_labels, step)
        return data, component_labels, targets

    @staticmethod
    def evaluate_oracle(data: torch.Tensor, tasks1: torch.Tensor, tasks2: torch.Tensor, 
                       component_labels: torch.Tensor) -> torch.Tensor:
        """Oracle evaluation without noise."""
        B, T, D = data.shape

        # Expand tasks to shape (B, T, D)
        tasks1_exp = tasks1.squeeze(-1)[:, None, :].expand(-1, T, -1)
        tasks2_exp = tasks2.squeeze(-1)[:, None, :].expand(-1, T, -1)

        # Select appropriate task vector for each position
        tasks = torch.where(component_labels[..., None] == 0, tasks1_exp, tasks2_exp)

        # Element-wise product and sum across D (dot product): (B, T)
        targets = (data * tasks).sum(dim=-1)
        
        return targets

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> List["MixtureOfGaussiansRegression"]:
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
        eval_tasks = [MixtureOfGaussiansRegression(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(MixtureOfGaussiansRegression.from_task_pools(
                task_pool1=self.task_pool1.clone(), 
                task_pool2=self.task_pool2.clone(), 
                **config
            ))
        return eval_tasks

    def get_default_eval_models(self) -> List[Any]:
        # Note: This would need custom models that handle component-specific weights
        models = []
        # Standard ridge regression as baseline (won't be optimal for this task)
        models.append(get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype))
        # You would need to implement component-aware models here
        return models


# Example usage and testing
if __name__ == "__main__":
    print("Testing DiscreteInputLinearRegression:")
    print("-" * 50)
    
    # Create discrete input task
    discrete_task = DiscreteInputLinearRegression(
        n_tasks=10,
        n_discrete_vectors=20,  # 20 discrete vectors to choose from
        n_points=100,
        batch_size=32,
        data_seed=42,
        task_seed=43,
        noise_seed=44,
        data_scale=1.0,
        task_scale=1.0,
        noise_scale=0.1
    )
    
    # Sample a batch
    data, tasks, targets = discrete_task.sample_batch(step=0)
    print(f"Data shape: {data.shape}")
    print(f"Tasks shape: {tasks.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Number of unique vectors in batch: {len(torch.unique(data.reshape(-1, 6), dim=0))}")
    print()
    
    print("Testing MixtureOfGaussiansRegression:")
    print("-" * 50)
    
    # Create mixture of Gaussians task
    mixture_task = MixtureOfGaussiansRegression(
        n_tasks=10,
        n_points=100,
        batch_size=32,
        data_seed=42,
        task_seed=43,
        noise_seed=44,
        data_scale=0.5,
        task_scale=1.0,
        noise_scale=0.1,
        mixture_prob=0.5
    )
    
    # Sample a batch
    (data, component_labels), (tasks1, tasks2), targets = mixture_task.sample_batch(step=0)
    print(f"Data shape: {data.shape}")
    print(f"Tasks1 shape: {tasks1.shape}")
    print(f"Tasks2 shape: {tasks2.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Centroid 1: {mixture_task.centroid1}")
    print(f"Centroid 2: {mixture_task.centroid2}")
    
    # Check data distribution
    data_flat = data.reshape(-1, 6)
    mean = data_flat.mean(dim=0)
    print(f"Overall data mean: {mean}")
    print(f"Distance from centroid1: {torch.norm(mean - mixture_task.centroid1):.3f}")
    print(f"Distance from centroid2: {torch.norm(mean - mixture_task.centroid2):.3f}")