from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Any, List
from pathlib import Path

if TYPE_CHECKING:
    from ..data.config import MultiCorpusConfig, PreprocessingConfig

@dataclass
class ModelConfig:
    """
    Architecture configuration class for the model.
    Supports both encoder_decoder and decoder_only architectures.
    """
    architecture: str  # "encoder_decoder" or "decoder_only"
    n_block: int
    n_head: int
    d_model: int
    d_ff: int
    dropout_rate: float
    use_flash_attention: bool = True
    sequence_length: Optional[int] = None  # Required for decoder_only, optional for encoder_decoder

@dataclass
class OptimizerConfig:
    """
    Optimizer configuration class for the model.
    """
    name: str
    weight_decay: float
    betas: tuple
    epsilon: float
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_clipping: float = 1.0  # Gradient norm clipping threshold

@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration class for the model.
    """
    name: str
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0
    decay_factor: float = 0.1
    decay_steps: Optional[int] = None
    decay_rate: float = 0.9

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
    logging_config: Optional["LoggingConfig"] = None
    use_mixed_precision: bool = False  # Use fp16/bf16 for training

@dataclass
class CheckpointConfig:
    """
    Checkpoint configuration class for the model.
    """
    save_dir: str
    save_frequency: int


@dataclass
class RunPodConfig:
    """
    RunPod-specific configuration for spot instance support.
    """
    enabled: bool = False
    base_path: str = "/workspace"
    emergency_checkpoint_on_signal: bool = True
    checkpoint_every_n_steps: int = 100
    auto_resume: bool = True

@dataclass
class SplitConfig:
    """
    Data split configuration class for the model.
    """
    train_split: float
    val_split: float
    test_split: float

@dataclass
class LoggingConfig:
    """
    Enhanced TensorBoard logging configuration.
    """
    # Periodic metrics frequencies (in steps)
    per_layer_metrics_frequency: int = 1000
    attention_visualization_frequency: int = 1000
    weight_norm_frequency: int = 1000

    # Validation sample settings
    num_qualitative_samples: int = 10
    num_random_val_samples: int = 5
    max_validation_translations: int = 500

    # Sample sentences configuration
    sample_sentences_path: Optional[Path] = None

@dataclass
class DataConfig:
    """
    Data configuration class for the model.
    """
    batch_size: int
    tokenizer_path: str = "models/tokenizers/europarl_tokenizer.json"


@dataclass
class GenerationConfig:
    """
    Configuration for generation sampling during training.
    """
    enabled: bool = False
    sample_every_n_steps: int = 100
    num_samples: int = 1
    temperatures: Optional[List[float]] = None
    max_new_tokens: int = 50
    prompts: Optional[List[str]] = None

    def __post_init__(self):
        if self.temperatures is None:
            object.__setattr__(self, 'temperatures', [0.8, 1.0])
        if self.prompts is None:
            object.__setattr__(self, 'prompts', ["Once upon a time"])


@dataclass
class ExperimentConfig:
    """
    Full experiment configuration class for the model.
    """
    experiment_name: str
    model_config: ModelConfig
    training_config: TrainingConfig
    checkpoint_config: CheckpointConfig
    data_config: DataConfig
    runpod_config: Optional[RunPodConfig] = None
    multi_corpus_config: Optional["MultiCorpusConfig"] = None
    generative_config: Optional[Any] = None  # For decoder-only tasks
    generative_preprocessing_config: Optional["PreprocessingConfig"] = None  # For decoder-only preprocessing
    generation_config: Optional[GenerationConfig] = None  # For generation sampling
