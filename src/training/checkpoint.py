import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


def create_and_save_checkpoint(trainer: "Trainer", checkpoint_filename: str) -> None:
    """Create checkpoint dictionary and save to disk.
    
    Args:
        trainer: The Trainer instance
        checkpoint_filename: Name of the checkpoint file (e.g., 'epoch_0.pt', 'checkpoint_step_100.pt')
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(trainer.checkpoint_config.save_dir) / trainer.timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Build checkpoint dictionary
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "step": trainer.training_state.step,
        "epoch": trainer.training_state.epoch,
        "best_val_loss": trainer.training_state.best_val_loss,
        "rng_state": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / checkpoint_filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def save_time_based_checkpoint(trainer: "Trainer") -> None:
    """Save checkpoint at regular time intervals based on save_frequency."""
    current_time = time.time()
    
    # Check if enough time has passed since last checkpoint
    if current_time - trainer.training_state.last_checkpoint_time < trainer.checkpoint_config.save_frequency:
        return
    
    create_and_save_checkpoint(trainer, f"checkpoint_step_{trainer.training_state.step}.pt")
    trainer.training_state.last_checkpoint_time = current_time


def save_epoch_based_checkpoint(trainer: "Trainer") -> None:
    """Save checkpoint at the end of each epoch."""
    create_and_save_checkpoint(trainer, f"epoch_{trainer.training_state.epoch}.pt")

