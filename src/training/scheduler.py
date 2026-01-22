"""Learning rate scheduler factory using PyTorch's built-in schedulers."""

from typing import Optional
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class SchedulerFactory:
    """Factory for creating learning rate schedulers from configuration."""
    
    @staticmethod
    def create(scheduler_name: str, optimizer: Optimizer, total_steps: int,
               **kwargs) -> LRScheduler:
        """
        Create a learning rate scheduler using PyTorch's built-in schedulers.
        
        Args:
            scheduler_name: Name of the scheduler (e.g., 'cosine', 'constant')
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            **kwargs: Additional arguments specific to the scheduler
                - learning_rate: Peak learning rate
                - warmup_ratio: Fraction of steps for warmup (default: 0.05)
                - min_lr_ratio: Minimum LR for cosine (default: 0.0)
                - decay_factor: For step decay (default: 0.1)
                - decay_steps: For step decay (default: total_steps / 3)
                - decay_rate: For exponential decay (default: 0.9)
        
        Returns:
            LRScheduler: PyTorch scheduler instance
        
        Raises:
            ValueError: If scheduler name is not supported
            TypeError: If required arguments are missing
        """
        scheduler_name = scheduler_name.lower()
        
        if scheduler_name not in SchedulerFactory.SCHEDULERS:
            supported = ', '.join(SchedulerFactory.SCHEDULERS.keys())
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. "
                f"Supported schedulers: {supported}"
            )
        
        learning_rate = kwargs.get('learning_rate')
        if learning_rate is None:
            raise TypeError(
                f"Scheduler '{scheduler_name}' requires 'learning_rate' parameter"
            )
        
        # Filter kwargs based on scheduler type
        if scheduler_name == 'constant':
            return SchedulerFactory._create_constant(optimizer, learning_rate)
        elif scheduler_name == 'linear_warmup':
            warmup_ratio = kwargs.get('warmup_ratio', 0.05)
            return SchedulerFactory._create_linear_warmup(optimizer, learning_rate, total_steps, warmup_ratio)
        elif scheduler_name == 'cosine':
            warmup_ratio = kwargs.get('warmup_ratio', 0.05)
            min_lr_ratio = kwargs.get('min_lr_ratio', 0.0)
            return SchedulerFactory._create_cosine(optimizer, learning_rate, total_steps, warmup_ratio, min_lr_ratio)
        elif scheduler_name == 'step_decay':
            warmup_ratio = kwargs.get('warmup_ratio', 0.05)
            decay_factor = kwargs.get('decay_factor', 0.1)
            decay_steps = kwargs.get('decay_steps', None)
            return SchedulerFactory._create_step_decay(optimizer, learning_rate, total_steps, warmup_ratio, decay_factor, decay_steps)
        elif scheduler_name == 'exponential_decay':
            warmup_ratio = kwargs.get('warmup_ratio', 0.05)
            decay_rate = kwargs.get('decay_rate', 0.9)
            return SchedulerFactory._create_exponential_decay(optimizer, learning_rate, total_steps, warmup_ratio, decay_rate)
        
        # This should never be reached due to the earlier check
        raise ValueError(f"Scheduler '{scheduler_name}' not handled")
    
    @staticmethod
    def _create_constant(optimizer: Optimizer, learning_rate: float) -> LRScheduler:
        """Create a constant learning rate scheduler."""
        # Return a no-op scheduler (LambdaLR with constant factor)
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    
    @staticmethod
    def _create_linear_warmup(optimizer: Optimizer, learning_rate: float, 
                             total_steps: int, warmup_ratio: float) -> LRScheduler:
        """Create a linear warmup followed by constant learning rate."""
        warmup_steps = int(total_steps * warmup_ratio)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    @staticmethod
    def _create_cosine(optimizer: Optimizer, learning_rate: float, 
                       total_steps: int, warmup_ratio: float, 
                       min_lr_ratio: float) -> LRScheduler:
        """Create a cosine annealing scheduler with linear warmup."""
        warmup_steps = int(total_steps * warmup_ratio)
        cosine_steps = total_steps - warmup_steps
        min_lr_factor = min_lr_ratio
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                progress = (step - warmup_steps) / cosine_steps if cosine_steps > 0 else 1.0
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    @staticmethod
    def _create_step_decay(optimizer: Optimizer, learning_rate: float, 
                           total_steps: int, warmup_ratio: float, 
                           decay_factor: float, decay_steps: Optional[int]) -> LRScheduler:
        """Create a step decay scheduler with linear warmup."""
        warmup_steps = int(total_steps * warmup_ratio)
        actual_decay_steps = decay_steps or max(1, (total_steps - warmup_steps) // 3)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                steps_since_warmup = step - warmup_steps
                decay_count = steps_since_warmup // actual_decay_steps
                return decay_factor ** decay_count
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    @staticmethod
    def _create_exponential_decay(optimizer: Optimizer, learning_rate: float, 
                                  total_steps: int, warmup_ratio: float, 
                                  decay_rate: float) -> LRScheduler:
        """Create an exponential decay scheduler with linear warmup."""
        warmup_steps = int(total_steps * warmup_ratio)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps if warmup_steps > 0 else 1.0
            else:
                steps_since_warmup = step - warmup_steps
                return decay_rate ** steps_since_warmup
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Mapping of scheduler names
    SCHEDULERS = {
        'constant': 'constant',
        'linear_warmup': 'linear_warmup',
        'cosine': 'cosine',
        'step_decay': 'step_decay',
        'exponential_decay': 'exponential_decay',
    }
    
    @staticmethod
    def get_supported_schedulers() -> list:
        """Get list of supported scheduler names."""
        return list(SchedulerFactory.SCHEDULERS.keys())
