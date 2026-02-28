from torch.nn import Module
import logging

from src.config.base import OptimizerConfig

logger = logging.getLogger(__name__)

def create_optimizer(model: Module, optimizer_config: OptimizerConfig, learning_rate: float):
    # Parameters that should NOT have weight decay applied
    # Embeddings are lookup tables, not computational weights
    # Biases and layer norms don't benefit from weight decay
    # Projection layers that share weights with embeddings should also not be decayed
    no_decay = ["embedding", "bias", "norm", "projection"]

    # Split parameters into two groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Log parameter group sizes for debugging
    logger.info(f"Optimizer parameter groups: {len(decay_params)} with decay, {len(no_decay_params)} without decay")

    optimizer_grouped_parameters = []

    # Only add groups if they have parameters
    if decay_params:
        optimizer_grouped_parameters.append({
            "params": decay_params,
            "weight_decay": optimizer_config.weight_decay,
        })

    if no_decay_params:
        optimizer_grouped_parameters.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
        })

    if not optimizer_grouped_parameters:
        raise ValueError("Model has no trainable parameters!")

    if optimizer_config.name == "adam":
        from torch.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=optimizer_config.betas,
            eps=optimizer_config.epsilon,
        )
        return optimizer

    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")