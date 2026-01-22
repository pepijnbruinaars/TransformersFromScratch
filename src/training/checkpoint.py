"""Checkpoint saving functionality with atomic writes and emergency saves.

Provides robust checkpoint saving for RunPod Spot instances with:
- Atomic writes to prevent corruption during preemption
- Scheduler state preservation for exact training resume
- Emergency checkpoint saves on termination signals
- Last state checkpoint for easy resume
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def create_and_save_checkpoint(
    trainer: "Trainer", checkpoint_filename: str, atomic: bool = True
) -> None:
    """Create checkpoint dictionary and save to disk.

    Args:
        trainer: The Trainer instance
        checkpoint_filename: Name of the checkpoint file
        atomic: If True, write to temp file first then rename (default: True)
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(trainer.checkpoint_config.save_dir) / trainer.timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dictionary with all state needed for resume
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": (
            trainer.scheduler.state_dict() if trainer.scheduler else None
        ),
        "step": trainer.training_state.step,
        "epoch": trainer.training_state.epoch,
        "best_val_loss": trainer.training_state.best_val_loss,
        "rng_state": torch.get_rng_state(),
        "timestamp": trainer.timestamp,  # For resume identification
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    checkpoint_path = checkpoint_dir / checkpoint_filename

    if atomic:
        # Atomic write: save to temp file in same directory, then rename
        # Using same directory ensures same filesystem for atomic os.replace
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.replace(checkpoint_path)  # Atomic on POSIX and Windows
        logger.info(f"Checkpoint saved atomically: {checkpoint_path}")
    else:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def save_time_based_checkpoint(trainer: "Trainer") -> None:
    """Save checkpoint at regular time intervals based on save_frequency."""
    current_time = time.time()

    # Check if enough time has passed since last checkpoint
    if (
        current_time - trainer.training_state.last_checkpoint_time
        < trainer.checkpoint_config.save_frequency
    ):
        return

    create_and_save_checkpoint(
        trainer, f"checkpoint_step_{trainer.training_state.step}.pt"
    )
    trainer.training_state.last_checkpoint_time = current_time


def save_epoch_based_checkpoint(trainer: "Trainer") -> None:
    """Save checkpoint at the end of each epoch."""
    create_and_save_checkpoint(trainer, f"epoch_{trainer.training_state.epoch}.pt")


def save_last_state_checkpoint(trainer: "Trainer") -> None:
    """Save/overwrite the 'last_state.pt' checkpoint for easy resume.

    This checkpoint is always overwritten and represents the most recent
    training state. Use this for resuming interrupted training.
    """
    create_and_save_checkpoint(trainer, "last_state.pt", atomic=True)


def save_emergency_checkpoint(trainer: "Trainer") -> None:
    """Save emergency checkpoint immediately during shutdown.

    Called when a termination signal is received. Saves both an
    emergency checkpoint with timestamp and updates last_state.pt.
    Uses atomic writes to prevent corruption.
    """
    emergency_filename = f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    logger.warning(f"Saving emergency checkpoint: {emergency_filename}")

    # Save emergency checkpoint
    create_and_save_checkpoint(trainer, emergency_filename, atomic=True)

    # Also update last_state for easy resume
    create_and_save_checkpoint(trainer, "last_state.pt", atomic=True)

    logger.warning("Emergency checkpoint saved successfully")
