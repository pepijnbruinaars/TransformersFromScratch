"""Tests for config loader functionality."""
import pytest
from pathlib import Path
from src.config.loader import ConfigLoader
from src.config.base import ExperimentConfig


class TestConfigLoaderBasics:
    """Tests for ConfigLoader basic functionality."""

    @pytest.fixture
    def loader(self):
        """Create a ConfigLoader instance."""
        return ConfigLoader()

    def test_loader_instance_creation(self, loader):
        """Test creating a ConfigLoader instance."""
        assert isinstance(loader, ConfigLoader)

    def test_loader_has_from_yaml_method(self, loader):
        """Test ConfigLoader has from_yaml method."""
        assert hasattr(loader, "from_yaml")
        assert callable(loader.from_yaml)

    def test_loader_has_to_yaml_method(self, loader):
        """Test ConfigLoader has to_yaml method."""
        assert hasattr(loader, "to_yaml")
        assert callable(loader.to_yaml)


class TestYAMLLoading:
    """Tests for YAML loading functionality."""

    @pytest.fixture
    def loader(self):
        """Create a ConfigLoader instance."""
        return ConfigLoader()

    def test_from_yaml_returns_experiment_config(self, loader, sample_yaml_config):
        """Test from_yaml returns an ExperimentConfig instance."""
        config = loader.from_yaml(str(sample_yaml_config))
        assert isinstance(config, ExperimentConfig)

    def test_from_yaml_deserializes_model_config(self, loader, sample_yaml_config):
        """Test from_yaml deserializes model configuration correctly."""
        config = loader.from_yaml(str(sample_yaml_config))

        assert config.model_config.n_block == 4
        assert config.model_config.n_head == 4
        assert config.model_config.d_model == 256
        assert config.model_config.d_ff == 1024
        assert config.model_config.dropout_rate == 0.1

    def test_from_yaml_deserializes_training_config(self, loader, sample_yaml_config):
        """Test from_yaml deserializes training configuration correctly."""
        config = loader.from_yaml(str(sample_yaml_config))

        assert config.training_config.num_epochs == 10
        assert config.training_config.scheduler_config.name == "cosine"
        assert config.training_config.logging_verbosity == 1

    def test_from_yaml_experiment_name(self, loader, sample_yaml_config):
        """Test from_yaml preserves experiment name."""
        config = loader.from_yaml(str(sample_yaml_config))
        assert config.experiment_name == "test_experiment"

    def test_from_yaml_nested_structure(self, loader, sample_yaml_config):
        """Test from_yaml creates proper nested dataclass structure."""
        config = loader.from_yaml(str(sample_yaml_config))

        assert hasattr(config, "model_config")
        assert hasattr(config, "training_config")
        assert hasattr(config, "checkpoint_config")
        assert hasattr(config.training_config, "scheduler_config")
        assert hasattr(config.training_config, "optimizer_config")


class TestYAMLLoadingErrors:
    """Tests for YAML loading error handling."""

    @pytest.fixture
    def loader(self):
        """Create a ConfigLoader instance."""
        return ConfigLoader()

    def test_from_yaml_file_not_found(self, loader):
        """Test from_yaml raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.from_yaml("non_existent_file.yaml")

    def test_from_yaml_empty_file(self, loader, temp_dir):
        """Test from_yaml raises error for empty YAML file."""
        empty_yaml_file = temp_dir / "empty.yaml"
        empty_yaml_file.write_text("")

        with pytest.raises(ValueError, match="empty or contains only comments"):
            loader.from_yaml(str(empty_yaml_file))

    def test_from_yaml_only_comments(self, loader, temp_dir):
        """Test from_yaml raises error for YAML with only comments."""
        comments_yaml_file = temp_dir / "comments.yaml"
        comments_yaml_file.write_text(
            """
# This is a comment
# Another comment
"""
        )

        with pytest.raises(ValueError, match="empty or contains only comments"):
            loader.from_yaml(str(comments_yaml_file))

    def test_from_yaml_malformed_yaml(self, loader, temp_dir):
        """Test from_yaml handles malformed YAML (bad indentation)."""
        malformed_yaml_file = temp_dir / "malformed.yaml"
        malformed_yaml_file.write_text(
            """
model:
  n_blocks: 6
   d_model: 512
"""
        )

        with pytest.raises(Exception):  # YAML parsing error
            loader.from_yaml(str(malformed_yaml_file))

    def test_from_yaml_with_none_values(self, loader, temp_dir):
        """Test from_yaml with missing fields results in None values (validation elsewhere)."""
        missing_field_file = temp_dir / "missing_model.yaml"
        missing_field_file.write_text(
            """
experiment_name: "test"
model:
  n_blocks: 4
training:
  scheduler:
    type: "cosine"
checkpoint:
  save_dir: "./checkpoints"
"""
        )

        config = loader.from_yaml(str(missing_field_file))
        # Missing fields should be None - validation happens elsewhere
        assert config.model_config.d_model is None
        assert config.model_config.d_ff is None


class TestYAMLSaving:
    """Tests for YAML saving functionality."""

    @pytest.fixture
    def loader(self):
        """Create a ConfigLoader instance."""
        return ConfigLoader()

    def test_to_yaml_creates_file_from_dict(self, loader, temp_dir):
        """Test to_yaml creates a file from dictionary."""
        config_dict = {
            "experiment_name": "test_experiment",
            "model": {
                "n_blocks": 6,
                "n_heads": 8,
                "d_model": 512,
                "d_ff": 2048,
                "dropout_rate": 0.1,
            },
            "training": {
                "n_epochs": 20,
                "scheduler": {"type": "cosine"},
            },
            "checkpoint": {
                "save_dir": "./checkpoints",
                "save_frequency": 1000,
            },
        }

        output_file = temp_dir / "output_config.yaml"
        loader.to_yaml(config_dict, str(output_file))

        assert output_file.exists()

    def test_to_yaml_creates_file_from_dataclass(self, loader, temp_dir):
        """Test to_yaml creates a file from ExperimentConfig dataclass."""
        from src.config.base import (
            ModelConfig,
            SchedulerConfig,
            OptimizerConfig,
            TrainingConfig,
            CheckpointConfig,
        )

        model_config = ModelConfig(
            n_block=6,
            n_head=8,
            d_model=512,
            d_ff=2048,
            dropout_rate=0.1,
        )
        scheduler = SchedulerConfig(name="cosine")
        optimizer = OptimizerConfig(
            name="adam",
            learning_rate=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            epsilon=1e-8,
        )
        training_config = TrainingConfig(
            num_epochs=20,
            scheduler_config=scheduler,
            optimizer_config=optimizer,
        )
        checkpoint_config = CheckpointConfig(
            save_dir="./checkpoints",
            save_frequency=1000,
        )
        experiment = ExperimentConfig(
            experiment_name="test",
            model_config=model_config,
            training_config=training_config,
            checkpoint_config=checkpoint_config,
        )

        output_file = temp_dir / "output_config.yaml"
        loader.to_yaml(experiment, str(output_file))

        assert output_file.exists()

    def test_to_yaml_file_content_valid_yaml(self, loader, temp_dir):
        """Test to_yaml writes valid YAML content."""
        config_dict = {
            "experiment_name": "test_experiment",
            "model": {
                "n_blocks": 4,
                "n_heads": 4,
                "d_model": 256,
                "d_ff": 1024,
                "dropout_rate": 0.1,
            },
            "training": {
                "n_epochs": 10,
                "scheduler": {"type": "cosine"},
            },
            "checkpoint": {
                "save_dir": "./checkpoints",
                "save_frequency": 500,
            },
        }

        output_file = temp_dir / "output_config.yaml"
        loader.to_yaml(config_dict, str(output_file))

        # Read back and verify it's valid YAML
        content = output_file.read_text()
        assert "d_model" in content
        assert "n_epochs" in content
        assert "experiment_name" in content

    def test_round_trip_yaml_experiment_config(self, loader, sample_yaml_config, temp_dir):
        """Test loading config, saving, and loading again preserves data."""
        # Load as ExperimentConfig
        original_config = loader.from_yaml(str(sample_yaml_config))

        # Save to new file
        output_file = temp_dir / "roundtrip_config.yaml"
        loader.to_yaml(original_config, str(output_file))

        # Load again as ExperimentConfig
        reloaded_config = loader.from_yaml(str(output_file))

        # Compare values - verify round-trip consistency
        assert original_config.experiment_name == reloaded_config.experiment_name
        assert original_config.model_config.n_block == reloaded_config.model_config.n_block
        assert original_config.model_config.n_head == reloaded_config.model_config.n_head
        assert original_config.model_config.d_model == reloaded_config.model_config.d_model
        assert original_config.model_config.d_ff == reloaded_config.model_config.d_ff
        assert (
            original_config.training_config.num_epochs
            == reloaded_config.training_config.num_epochs
        )
        assert (
            original_config.training_config.scheduler_config.name
            == reloaded_config.training_config.scheduler_config.name
        )
