"""Tests for config base classes."""
from dataclasses import is_dataclass, fields
from src.config.base import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    CheckpointConfig,
    ExperimentConfig,
)


class TestConfigStructures:
    """Tests for config dataclass structures."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            n_block=6,
            n_head=8,
            d_model=512,
            d_ff=2048,
            dropout_rate=0.1,
        )
        assert config.n_block == 6
        assert config.n_head == 8
        assert config.d_model == 512
        assert config.d_ff == 2048
        assert config.dropout_rate == 0.1

    def test_all_configs_are_dataclasses(self):
        """Test that all config classes are dataclasses."""
        assert is_dataclass(ModelConfig)
        assert is_dataclass(OptimizerConfig)
        assert is_dataclass(SchedulerConfig)
        assert is_dataclass(TrainingConfig)
        assert is_dataclass(CheckpointConfig)
        assert is_dataclass(ExperimentConfig)


class TestOptimizerConfigStructure:
    """Tests for OptimizerConfig dataclass."""

    def test_optimizer_config_creation(self):
        """Test creating an OptimizerConfig instance."""
        config = OptimizerConfig(
            name="adam",
            learning_rate=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )
        assert config.name == "adam"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-5
        assert config.betas == (0.9, 0.999)


class TestSchedulerConfigStructure:
    """Tests for SchedulerConfig dataclass."""

    def test_scheduler_config_with_defaults(self):
        """Test creating SchedulerConfig with default warmup_ratio."""
        config = SchedulerConfig(name="cosine")
        assert config.name == "cosine"
        assert config.warmup_ratio == 0.05

    def test_scheduler_config_with_custom_warmup(self):
        """Test creating SchedulerConfig with custom warmup_ratio."""
        config = SchedulerConfig(name="cosine", warmup_ratio=0.1)
        assert config.warmup_ratio == 0.1


class TestCheckpointConfigStructure:
    """Tests for CheckpointConfig dataclass."""

    def test_checkpoint_config_creation(self):
        """Test creating a CheckpointConfig instance."""
        config = CheckpointConfig(save_dir="./checkpoints", save_frequency=1000)
        assert config.save_dir == "./checkpoints"
        assert config.save_frequency == 1000
