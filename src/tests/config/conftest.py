"""Pytest configuration and shared fixtures."""
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML config file for testing."""
    config_content = """
experiment_name: "test_experiment"
model:
  architecture: "encoder_decoder"
  n_blocks: 4
  d_model: 256
  n_heads: 4
  d_ff: 1024
  dropout_rate: 0.1
training:
  n_epochs: 10
  scheduler:
    type: "cosine"
    peak_lr: 1e-4
  warmup_step_percentage: 5
data:
  batch_size: 32
checkpoint:
  save_every_n_minutes: 5
  save_best_only: true
"""
    config_file = temp_dir / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file
