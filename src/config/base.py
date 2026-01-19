from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Architecture configuration class for the model.
    """
    n_block: int
    n_head: int
    d_model: int
    d_ff: int
    dropout_rate: float

@dataclass
class OptimizerConfig:
    """
    Optimizer configuration class for the model.
    """
    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple
    epsilon: float
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0

@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration class for the model.
    """
    name: str
    warmup_ratio: float = 0.05

@dataclass
class TrainingConfig:
    """
    Training configuration class for the model.
    """
    num_epochs: int
    scheduler_config: SchedulerConfig
    optimizer_config: OptimizerConfig
    label_smoothing: float = 0.1
    tensorboard_log_dir: str = "logs/"
    tensorboard_flush_frequency: int = 100
    logging_verbosity: int = 1

@dataclass
class CheckpointConfig:
    """
    Checkpoint configuration class for the model.
    """
    save_dir: str
    save_frequency: int

@dataclass
class SplitConfig:
    """
    Data split configuration class for the model.
    """
    train_split: float
    val_split: float
    test_split: float

@dataclass
class DataConfig:
    """
    Data configuration class for the model.
    """
    dataset_path: str
    batch_size: int
    splits: SplitConfig
    shuffle: bool = True

@dataclass
class ExperimentConfig:
    """
    Full experiment configuration class for the model.
    """
    experiment_name: str
    model_config: ModelConfig
    training_config: TrainingConfig
    checkpoint_config: CheckpointConfig
