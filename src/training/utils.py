from torch.nn import Module

from src.config.base import OptimizerConfig

def create_optimizer(model: Module, optimizer_config: OptimizerConfig):
    if optimizer_config.name == "adam":
        from torch.optim import Adam
        optimizer = Adam(
            model.parameters(),
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        return optimizer
    
    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")