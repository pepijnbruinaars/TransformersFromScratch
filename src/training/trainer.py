from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional
import torch
from torch.nn import Module
from tqdm import tqdm
import subprocess
import socket
import webbrowser
import os
import time

from src.config.base import CheckpointConfig, TrainingConfig
from src.constants import PAD_TOKEN
from src.tokenization.tokenizer import CustomTokenizer
from src.training.utils import create_optimizer
from src.training.checkpoint import save_time_based_checkpoint, save_epoch_based_checkpoint
from src.training.logger import TrainingLogger
from src.training.scheduler import SchedulerFactory
from src.utils.device import get_device
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional

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
                 experiment_name: str = "default_experiment"
                 ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.device = get_device()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
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
        self.optimizer = create_optimizer(model, training_config.optimizer_config, 
                                         training_config.scheduler_config.learning_rate)
        self.scheduler: Optional[LRScheduler] = None  # Initialized in train(), depends on num_epochs
        
        # Create experiment-specific log directory with timestamp for unique runs
        import os
        experiment_log_dir = os.path.join(training_config.tensorboard_log_dir, experiment_name, self.timestamp)
        self.tb_logger = TrainingLogger(log_dir=experiment_log_dir)
        logger.info(f"Using device: {self.device}")
        logger.info(f"Logging to: {experiment_log_dir}")
    
    def train(self, num_epochs: int):
        # Calculate and log total token count (sample-based for speed)
        total_batches = len(self.train_dataloader)
        sample_size = min(10, total_batches)  # Sample first 10 batches for speed
        sampled_tokens = 0
        
        for i, batch in enumerate(self.train_dataloader):
            if i >= sample_size:
                break
            # Count non-padding tokens in source
            source_mask = batch.get("source_mask", None)
            if source_mask is not None:
                sampled_tokens += source_mask.sum().item() if source_mask.dim() > 2 else source_mask.sum().item()
            else:
                label = batch.get("label", None)
                if label is not None:
                    pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
                    sampled_tokens += (label != pad_id).sum().item()
        
        # Extrapolate from sample
        avg_tokens_per_batch = sampled_tokens / sample_size if sample_size > 0 else 0
        estimated_total_tokens = int(avg_tokens_per_batch * total_batches)
        
        logger.info(f"Training set statistics:")
        logger.info(f"  - Total batches: {total_batches}")
        logger.info(f"  - Estimated tokens per batch: {avg_tokens_per_batch:.0f}")
        logger.info(f"  - Estimated total tokens: {estimated_total_tokens:,}")
        logger.info(f"  - Total epochs: {num_epochs}")
        logger.info(f"  - Estimated total training tokens: {estimated_total_tokens * num_epochs:,}")
        
        # Initialize scheduler now that we know the number of epochs
        total_steps = len(self.train_dataloader) * num_epochs
        self.scheduler = SchedulerFactory.create(
            scheduler_name=self.training_config.scheduler_config.name,
            optimizer=self.optimizer,
            total_steps=total_steps,
            learning_rate=self.training_config.scheduler_config.learning_rate,
            warmup_ratio=self.training_config.scheduler_config.warmup_ratio,
            min_lr_ratio=self.training_config.scheduler_config.min_lr_ratio,
            decay_factor=self.training_config.scheduler_config.decay_factor,
            decay_steps=self.training_config.scheduler_config.decay_steps,
            decay_rate=self.training_config.scheduler_config.decay_rate,
        )
        logger.info(f"Using scheduler: {self.training_config.scheduler_config.name}")
        logger.info(f"Warmup steps: {int(total_steps * self.training_config.scheduler_config.warmup_ratio)}")
        
        for epoch in range(num_epochs):
            # 0. Update training state
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.training_state.epoch = epoch
            
            # 1. Set model in training mode
            self.model.train()
            
            # 2. Train and validate
            self._train_epoch()
            self._validate_epoch()
            save_epoch_based_checkpoint(self)
    
    def resume(self, num_epochs: int, checkpoint_path: str):
        pass

    def _train_epoch(self):
        batch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {self.training_state.epoch + 1} Training")
        epoch_losses = []
        
        for step, batch in enumerate(batch_iterator):
            # Prepare inputs
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            source_mask = batch["source_mask"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            label = batch["label"].to(self.device)
            
            loss = self._training_step(step, source, target, source_mask, target_mask, label)
            batch_iterator.set_postfix({"Loss": f"{loss:.4f}", "LR": self.optimizer.param_groups[0]['lr']})
            self.tb_logger.log_training_step(loss, self.training_state.step, self.optimizer.param_groups[0]['lr'])
            epoch_losses.append(loss)
            
            self.training_state.step += 1
            
            if step % self.training_config.tensorboard_flush_frequency == 0:
                self.tb_logger.flush()
    
    def _validate_epoch(self):
        pass
    
    def _training_step(self, step: int, source: torch.Tensor, target: torch.Tensor,
                       source_mask: torch.Tensor, target_mask: torch.Tensor, label: torch.Tensor) -> float:        
        # Save time-based checkpoint if needed
        save_time_based_checkpoint(self)
        
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
            if self.scheduler is None:
                raise ValueError("Scheduler not initialized.")    
            self.scheduler.step()   
            
            self.optimizer.zero_grad()
            
            # Log gradient norm
            self.tb_logger.log_gradient_norm(grad_norm_value.item(), self.training_state.step)
            
        return loss.item()

