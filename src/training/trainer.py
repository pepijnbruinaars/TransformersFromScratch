from dataclasses import dataclass
from datetime import datetime
import logging
import time
from pathlib import Path
import torch
from torch.nn import Module
from tqdm import tqdm

from src.config.base import CheckpointConfig, TrainingConfig
from src.constants import PAD_TOKEN
from src.tokenization.tokenizer import CustomTokenizer
from src.training.utils import create_optimizer
from src.utils.device import get_device
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    epoch: int
    step: int
    best_val_loss: float
    last_checkpoint_time: float
    accumulation_counter: int

class Trainer():
    def __init__(self, model: Module, 
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 tokenizer: CustomTokenizer,
                 training_config: TrainingConfig,
                 checkpoint_config: CheckpointConfig,
                 tensorboard_writer: SummaryWriter):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.device = get_device()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_state = TrainingState(
            epoch=0,
            step=0,
            best_val_loss=float('inf'),
            last_checkpoint_time=0.0,
            accumulation_counter=0,
        )
        self.model = model.to(self.device)
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.accumulation_steps = training_config.optimizer_config.accumulation_steps
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id(PAD_TOKEN),
                                                       label_smoothing=training_config.label_smoothing,
                                                       ).to(self.device)
        self.optimizer = create_optimizer(model, training_config.optimizer_config)
        
        self.tensorboard_writer = tensorboard_writer
        logger.info(f"Using device: {self.device}")
    
    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            # 0. Update training state
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.training_state.epoch = epoch
            
            # 1. Set model in training mode
            self.model.train()
            
            # 2. Train and validate
            self._train_epoch()
            self._validate_epoch()
            self._save_epoch_based_checkpoint()
    
    def resume(self, num_epochs: int, checkpoint_path: str):
        pass

    def _train_epoch(self):
        batch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {self.training_state.epoch + 1} Training")
        epoch_losses = []
        
        for step, batch in enumerate(batch_iterator):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            source_mask = batch["source_mask"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            label = batch["label"].to(self.device)
            
            loss = self._training_step(step, source, target, source_mask, target_mask, label)
            batch_iterator.set_postfix({"Loss": f"{loss:.4f}", "LR": self.optimizer.param_groups[0]['lr']})
            self.tensorboard_writer.add_scalar("Train/Loss", loss, self.training_state.step)
            epoch_losses.append(loss)
            
            self.training_state.step += 1
            
            if step % self.training_config.tensorboard_flush_frequency == 0:
                self.tensorboard_writer.flush()
    
    def _validate_epoch(self):
        pass
    
    def _training_step(self, step: int, source: torch.Tensor, target: torch.Tensor,
                       source_mask: torch.Tensor, target_mask: torch.Tensor, label: torch.Tensor) -> float:        
        # Determine the appropriate dtype based on device capabilities
        dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        with torch.autocast(device_type=self.device, dtype=dtype):
            # Forward pass
            output = self.model(source, target, source_mask, target_mask)
            
            # Compute loss
            loss = self.loss_function(output.view(-1, output.size(-1)), label.view(-1))
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()
            
        # Update parameters if accumulation step is reached
        if (step + 1) % self.accumulation_steps == 0:
            # Clip gradients
            grad_norm_value = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.training_config.optimizer_config.max_grad_norm)
            # Update parameters and reset gradients
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Log gradient norm
            self.tensorboard_writer.add_scalar("Train/Grad_Norm", grad_norm_value, self.training_state.step)
            
        return loss.item()
    
    def _create_and_save_checkpoint(self, checkpoint_filename: str) -> None:
        """Create checkpoint dictionary and save to disk.
        
        Args:
            checkpoint_filename: Name of the checkpoint file (e.g., 'epoch_0.pt', 'checkpoint_step_100.pt')
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.checkpoint_config.save_dir) / self.timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.training_state.step,
            "epoch": self.training_state.epoch,
            "best_val_loss": self.training_state.best_val_loss,
            "rng_state": torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / checkpoint_filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_time_based_checkpoint(self):
        """Save checkpoint at regular time intervals based on save_frequency."""
        current_time = time.time()
        
        # Check if enough time has passed since last checkpoint
        if current_time - self.training_state.last_checkpoint_time < self.checkpoint_config.save_frequency:
            return
        
        self._create_and_save_checkpoint(f"checkpoint_step_{self.training_state.step}.pt")
        self.training_state.last_checkpoint_time = current_time
    
    def _save_epoch_based_checkpoint(self):
        """Save checkpoint at the end of each epoch."""
        self._create_and_save_checkpoint(f"epoch_{self.training_state.epoch}.pt")

