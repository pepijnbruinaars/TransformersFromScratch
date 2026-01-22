"""Checkpoint resume functionality for training continuation.

Provides functions to load checkpoints and restore full training state,
enabling seamless resume after spot instance preemption or interruption.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

import torch

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint dictionary from disk.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., last_state.pt)

    Returns:
        Checkpoint dictionary containing:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict (if available)
        - step, epoch, best_val_loss
        - rng_state, cuda_rng_state (if available)
        - timestamp (if available)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load to CPU first, then move to appropriate device
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    logger.info(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  - Step: {checkpoint.get('step', 'unknown')}")
    logger.info(f"  - Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")

    return checkpoint


def restore_training_state(trainer: "Trainer", checkpoint: Dict[str, Any]) -> None:
    """Restore all training state from checkpoint.

    This includes:
    - Model weights
    - Optimizer state (including momentum buffers)
    - Training state (epoch, step, best_val_loss)
    - RNG states for reproducibility
    - Scheduler state (if available)

    Args:
        trainer: The Trainer instance to restore state into
        checkpoint: Checkpoint dictionary from load_checkpoint()
    """
    # 1. Restore model weights
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Restored model weights")

    # 2. Restore optimizer state
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Restored optimizer state")

    # 3. Restore training state
    trainer.training_state.epoch = checkpoint["epoch"]
    trainer.training_state.step = checkpoint["step"]
    trainer.training_state.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    logger.info(
        f"Restored training state: epoch={trainer.training_state.epoch}, "
        f"step={trainer.training_state.step}"
    )

    # 4. Restore RNG states for reproducibility
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
        logger.info("Restored PyTorch RNG state")

    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        logger.info("Restored CUDA RNG state")

    # 5. Restore scheduler state if present
    if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored scheduler state")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory.

    Priority order:
    1. last_state.pt (always most recent)
    2. Highest numbered epoch_N.pt
    3. Highest numbered checkpoint_step_N.pt

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        logger.debug(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Priority 1: last_state.pt
    last_state = path / "last_state.pt"
    if last_state.exists():
        logger.info(f"Found last_state.pt: {last_state}")
        return str(last_state)

    # Priority 2: epoch checkpoints (epoch_0.pt, epoch_1.pt, etc.)
    epoch_checkpoints = list(path.glob("epoch_*.pt"))
    if epoch_checkpoints:
        # Sort by epoch number (descending)
        sorted_epochs = sorted(
            epoch_checkpoints,
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        logger.info(f"Found epoch checkpoint: {sorted_epochs[0]}")
        return str(sorted_epochs[0])

    # Priority 3: step checkpoints (checkpoint_step_100.pt, etc.)
    step_checkpoints = list(path.glob("checkpoint_step_*.pt"))
    if step_checkpoints:
        # Sort by step number (descending)
        sorted_steps = sorted(
            step_checkpoints,
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )
        logger.info(f"Found step checkpoint: {sorted_steps[0]}")
        return str(sorted_steps[0])

    logger.debug(f"No checkpoints found in: {checkpoint_dir}")
    return None


def find_latest_checkpoint_across_runs(base_checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint across all timestamped run directories.

    Searches through all subdirectories (which are timestamped run folders)
    and returns the most recently modified checkpoint.

    Args:
        base_checkpoint_dir: Base checkpoint directory containing run subdirs

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    import os

    base_path = Path(base_checkpoint_dir)
    if not base_path.exists():
        logger.info(f"Base checkpoint directory does not exist: {base_checkpoint_dir}")
        return None

    all_checkpoints = []

    # Search all timestamped subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            latest = find_latest_checkpoint(str(subdir))
            if latest:
                all_checkpoints.append(latest)

    if not all_checkpoints:
        logger.info(f"No checkpoints found in any subdirectory of: {base_checkpoint_dir}")
        return None

    # Return the most recently modified checkpoint
    latest = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Selected most recent checkpoint: {latest}")
    return latest
