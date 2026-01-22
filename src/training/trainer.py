"""Training module with RunPod Spot instance support.

Provides robust training with:
- Graceful shutdown on SIGTERM/SIGINT
- Checkpoint resume capability
- Periodic last_state.pt saves for easy resume
- Enhanced TensorBoard logging
"""
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from src.config.base import CheckpointConfig, LoggingConfig, TrainingConfig, RunPodConfig
from src.constants import END_TOKEN, PAD_TOKEN, START_TOKEN
from src.tokenization.tokenizer import CustomTokenizer
from src.training.utils import create_optimizer
from src.training.checkpoint import (
    save_time_based_checkpoint,
    save_epoch_based_checkpoint,
    save_last_state_checkpoint,
    save_emergency_checkpoint,
)
from src.training.logger import TrainingLogger
from src.training.metrics import Metrics
from src.training.sample_loader import SampleSentences
from src.training.scheduler import SchedulerFactory
from src.training.signals import GracefulKiller
from src.training.resume import (
    load_checkpoint,
    restore_training_state,
    find_latest_checkpoint_across_runs,
)
from src.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    epoch: int
    step: int
    best_val_loss: float
    last_checkpoint_time: float
    accumulation_counter: int


class Trainer:
    def __init__(
        self,
        model: Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer: CustomTokenizer,
        training_config: TrainingConfig,
        checkpoint_config: CheckpointConfig,
        experiment_name: str = "default_experiment",
        runpod_config: Optional[RunPodConfig] = None,
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device = get_device()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.training_state = TrainingState(
            epoch=0,
            step=0,
            best_val_loss=float("inf"),
            last_checkpoint_time=0.0,
            accumulation_counter=0,
        )
        self.model = model.to(self.device)
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.runpod_config = runpod_config
        self.accumulation_steps = training_config.optimizer_config.accumulation_steps

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer

        self.loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.token_to_id(PAD_TOKEN),
            label_smoothing=training_config.label_smoothing,
        ).to(self.device)
        self.optimizer = create_optimizer(
            model,
            training_config.optimizer_config,
            training_config.scheduler_config.learning_rate,
        )
        self.scheduler: Optional[LRScheduler] = None

        # Initialize signal handler for graceful shutdown (RunPod spot instance support)
        self.killer = GracefulKiller.get_instance()

        # Determine checkpoint frequency (use RunPod config if available)
        self._checkpoint_every_n_steps = (
            runpod_config.checkpoint_every_n_steps
            if runpod_config and runpod_config.enabled
            else 2000  # Default: every 2000 steps
        )

        # Create experiment-specific log directory
        experiment_log_dir = os.path.join(
            training_config.tensorboard_log_dir, experiment_name, self.timestamp
        )
        self.tb_logger = TrainingLogger(log_dir=experiment_log_dir)

        # Initialize metrics calculator
        self.metrics = Metrics()

        # Initialize sample sentences for evaluation (load later via load_sample_sentences)
        self.sample_sentences: Optional[SampleSentences] = None

        # Get logging config with defaults
        self.logging_config = training_config.logging_config or LoggingConfig()

        logger.info(f"Using device: {self.device}")
        logger.info(f"Logging to: {experiment_log_dir}")

    def _initialize_scheduler(self, num_epochs: int) -> None:
        """Initialize the learning rate scheduler."""
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
        logger.info(
            f"Warmup steps: {int(total_steps * self.training_config.scheduler_config.warmup_ratio)}"
        )

    def load_sample_sentences(self, path: Optional[Path] = None) -> None:
        """Load sample sentences for qualitative evaluation.

        Args:
            path: Optional path to YAML config. If None, uses defaults.
        """
        self.sample_sentences = SampleSentences.load(path)
        logger.info(
            f"Loaded {len(self.sample_sentences.evaluation_pairs)} evaluation pairs "
            f"and {len(self.sample_sentences.attention_probes)} attention probes"
        )

    def _compute_padding_ratio(self, batch: dict) -> float:
        """Compute the ratio of padding tokens in a batch.

        Note: Should be called before moving batch to device for efficiency.

        Args:
            batch: Batch dictionary with 'source' and 'target' tensors.

        Returns:
            Ratio of padding tokens (0.0 to 1.0).
        """
        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)

        source = batch["source"]
        target = batch["target"]

        source_padding = (source == pad_id).sum().item()
        target_padding = (target == pad_id).sum().item()

        total_padding = source_padding + target_padding
        total_tokens = source.numel() + target.numel()

        return total_padding / total_tokens if total_tokens > 0 else 0.0

    def _count_non_padding_tokens(self, batch: dict) -> int:
        """Count non-padding tokens in batch for throughput calculation.

        Note: Should be called before moving batch to device for efficiency.

        Args:
            batch: Batch dictionary with 'source' and 'target' tensors.

        Returns:
            Number of non-padding tokens.
        """
        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)

        source = batch["source"]
        target = batch["target"]

        source_tokens = (source != pad_id).sum().item()
        target_tokens = (target != pad_id).sum().item()

        return source_tokens + target_tokens

    def _create_source_mask(self, source: torch.Tensor) -> torch.Tensor:
        """Create attention mask for source sequence.

        Args:
            source: Source tensor of shape (batch, seq_len).

        Returns:
            Mask tensor of shape (batch, 1, 1, seq_len).
        """
        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        mask = (source != pad_id).unsqueeze(1).unsqueeze(1)
        return mask.to(self.device)

    def _create_decoder_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder (autoregressive).

        Args:
            seq_len: Length of the target sequence.

        Returns:
            Causal mask tensor.
        """
        mask = torch.triu(
            torch.ones(1, seq_len, seq_len, device=self.device),
            diagonal=1,
        ).bool()
        mask = ~mask
        return mask.unsqueeze(0)

    def _translate_single(self, source_text: str, max_length: int = 512) -> str:
        """Translate a single sentence using greedy decoding.

        Args:
            source_text: Source text to translate.
            max_length: Maximum output length.

        Returns:
            Translated text.
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize
            source_tokens = self.tokenizer.encode(source_text)
            source_tensor = torch.tensor(
                [source_tokens], dtype=torch.int64, device=self.device
            )

            # Create mask
            source_mask = self._create_source_mask(source_tensor)

            # Encode
            encoder_output = self.model.encode(source_tensor, source_mask)

            # Initialize decoder with START token
            start_id = self.tokenizer.token_to_id(START_TOKEN)
            end_id = self.tokenizer.token_to_id(END_TOKEN)
            decoder_input = torch.tensor([[start_id]], device=self.device)

            generated_tokens = []

            for _ in range(max_length):
                decoder_mask = self._create_decoder_mask(decoder_input.size(1))

                decoder_output = self.model.decode(
                    decoder_input, encoder_output, source_mask, decoder_mask
                )

                projection = self.model.project(decoder_output)
                next_token = projection[:, -1, :].argmax(dim=-1)
                next_token_id = next_token.item()

                if next_token_id == end_id:
                    break

                generated_tokens.append(next_token_id)
                decoder_input = torch.cat(
                    [decoder_input, next_token.unsqueeze(0)], dim=1
                )

            translation = self.tokenizer.decode(generated_tokens)

        self.model.train()
        return translation

    def _extract_attentions_for_sentence(
        self, source_text: str, max_length: int = 128
    ) -> tuple[list, dict, list[str], list[str]]:
        """Extract attention maps for a single sentence via autoregressive decoding.

        Args:
            source_text: Source text to analyze.
            max_length: Maximum decoding length.

        Returns:
            Tuple of (encoder_attentions, decoder_attentions, source_tokens, target_tokens).
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize source
            source_tokens = self.tokenizer.encode(source_text)
            source_tensor = torch.tensor(
                [source_tokens], dtype=torch.int64, device=self.device
            )

            source_mask = self._create_source_mask(source_tensor)

            # Encode with attention (encoder attentions captured once)
            encoder_output, encoder_attentions = self.model.encode(
                source_tensor, source_mask, return_attentions=True
            )

            # Autoregressive decoding to build up target sequence
            start_id = self.tokenizer.token_to_id(START_TOKEN)
            end_id = self.tokenizer.token_to_id(END_TOKEN)
            decoder_input = torch.tensor([[start_id]], device=self.device)
            generated_token_ids = [start_id]

            # We'll collect the final cross-attention after full decoding
            final_decoder_attentions = None

            for _ in range(max_length):
                decoder_mask = self._create_decoder_mask(decoder_input.size(1))

                _, decoder_attentions = self.model.decode(
                    decoder_input,
                    encoder_output,
                    source_mask,
                    decoder_mask,
                    return_attentions=True,
                )

                # Get projection and next token
                decoder_output = self.model.decode(
                    decoder_input, encoder_output, source_mask, decoder_mask
                )
                projection = self.model.project(decoder_output)
                next_token = projection[:, -1, :].argmax(dim=-1)
                next_token_id = next_token.item()

                if next_token_id == end_id:
                    # Capture final attention state before stopping
                    final_decoder_attentions = decoder_attentions
                    break

                generated_token_ids.append(next_token_id)
                decoder_input = torch.cat(
                    [decoder_input, next_token.unsqueeze(0)], dim=1
                )
                final_decoder_attentions = decoder_attentions

            # Get token strings
            source_token_strs = [
                self.tokenizer.id_to_token(t) for t in source_tokens
            ]
            target_token_strs = [
                self.tokenizer.id_to_token(t) for t in generated_token_ids
            ]

        self.model.train()

        return encoder_attentions, final_decoder_attentions or {}, source_token_strs, target_token_strs

    def _log_periodic_metrics(self, step: int) -> None:
        """Log metrics that are expensive to compute (every N steps).

        Args:
            step: Current training step.
        """
        logger.info(f"Logging periodic metrics at step {step}")

        # Layer-wise gradient norms
        self.tb_logger.log_layer_gradient_norms(self.model, step)

        # Weight norms
        self.tb_logger.log_weight_norms(self.model, step)

        # Full weight histograms (expensive: CPU transfer for every parameter)
        self.tb_logger.log_weight_histograms(self.model, step)

        # Attention visualization for probe sentences
        if self.sample_sentences:
            for i, sentence in enumerate(self.sample_sentences.attention_probes[:3]):
                try:
                    enc_attn, dec_attn, src_tokens, tgt_tokens = (
                        self._extract_attentions_for_sentence(sentence)
                    )
                    self.tb_logger.log_attention_maps(
                        enc_attn,
                        dec_attn,
                        src_tokens,
                        tgt_tokens,
                        step=step,
                        sentence_idx=i,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log attention for sentence {i}: {e}")

    def _log_qualitative_samples(self, step: int) -> None:
        """Log qualitative translation samples.

        Args:
            step: Current training step.
        """
        if self.sample_sentences is None:
            return

        samples = []

        # Fixed evaluation pairs
        num_fixed = min(
            self.logging_config.num_qualitative_samples,
            len(self.sample_sentences.evaluation_pairs),
        )
        for pair in self.sample_sentences.evaluation_pairs[:num_fixed]:
            try:
                prediction = self._translate_single(pair.source)
                samples.append((pair.source, pair.target, prediction))
            except Exception as e:
                logger.warning(f"Failed to translate: {pair.source[:50]}... Error: {e}")

        # Random validation samples (if validation dataset supports indexing)
        try:
            val_dataset = self.val_dataloader.dataset
            len_func = getattr(val_dataset, "__len__", None)
            getitem_func = getattr(val_dataset, "__getitem__", None)
            if len_func is not None and getitem_func is not None:
                dataset_len: int = len_func()
                num_random = min(
                    self.logging_config.num_random_val_samples, dataset_len
                )
                if num_random > 0:
                    indices = random.sample(range(dataset_len), num_random)
                    for idx in indices:
                        item = getitem_func(idx)
                        # Check if raw text is available
                        if "source_text" in item and "target_text" in item:
                            source_text = item["source_text"]
                            target_text = item["target_text"]
                            prediction = self._translate_single(source_text)
                            samples.append((source_text, target_text, prediction))
        except Exception as e:
            logger.warning(f"Could not sample from validation set: {e}")

        if samples:
            self.tb_logger.log_text_samples(samples, step)

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Train the model for the specified number of epochs.

        Args:
            num_epochs: Total number of epochs to train
            start_epoch: Epoch to start from (for resume support)
        """
        # Calculate and log total token count (sample-based for speed)
        total_batches = len(self.train_dataloader)
        sample_size = min(10, total_batches)
        sampled_tokens = 0

        for i, batch in enumerate(self.train_dataloader):
            if i >= sample_size:
                break
            source_mask = batch.get("source_mask", None)
            if source_mask is not None:
                sampled_tokens += (
                    source_mask.sum().item()
                    if source_mask.dim() > 2
                    else source_mask.sum().item()
                )
            else:
                label = batch.get("label", None)
                if label is not None:
                    pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
                    sampled_tokens += (label != pad_id).sum().item()

        avg_tokens_per_batch = sampled_tokens / sample_size if sample_size > 0 else 0
        estimated_total_tokens = int(avg_tokens_per_batch * total_batches)

        logger.info("Training set statistics:")
        logger.info(f"  - Total batches: {total_batches}")
        logger.info(f"  - Estimated tokens per batch: {avg_tokens_per_batch:.0f}")
        logger.info(f"  - Estimated total tokens: {estimated_total_tokens:,}")
        logger.info(f"  - Total epochs: {num_epochs}")
        logger.info(
            f"  - Estimated total training tokens: {estimated_total_tokens * num_epochs:,}"
        )

        # Initialize scheduler if not already done (resume handles this separately)
        if self.scheduler is None:
            self._initialize_scheduler(num_epochs)

        for epoch in range(start_epoch, num_epochs):
            # Check for shutdown signal at epoch boundary
            if self.killer.should_stop():
                logger.warning(
                    "Shutdown signal received at epoch boundary, saving checkpoint..."
                )
                save_emergency_checkpoint(self)
                logger.info("Exiting training due to shutdown signal")
                sys.exit(0)

            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            self.training_state.epoch = epoch
            self.model.train()

            # Train epoch - returns True if we should stop
            should_stop = self._train_epoch()
            if should_stop:
                logger.info("Exiting training due to shutdown signal")
                sys.exit(0)

            self._validate_epoch()
            save_epoch_based_checkpoint(self)
            save_last_state_checkpoint(self)

        logger.info("Training completed successfully")

    def resume(self, num_epochs: int, checkpoint_path: Optional[str] = None) -> None:
        """Resume training from a checkpoint.

        Args:
            num_epochs: Total number of epochs to train (not remaining)
            checkpoint_path: Path to checkpoint. If None, finds latest automatically.
        """
        # Find checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint_across_runs(
                self.checkpoint_config.save_dir
            )
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_config.save_dir}"
                )
            logger.info(f"Auto-selected checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)

        # Restore timestamp to continue in same checkpoint directory
        if "timestamp" in checkpoint:
            self.timestamp = checkpoint["timestamp"]
            logger.info(f"Restored timestamp: {self.timestamp}")

        # Initialize scheduler before restoring state
        self._initialize_scheduler(num_epochs)

        # Restore all state (model, optimizer, scheduler, training state, RNG)
        restore_training_state(self, checkpoint)

        # Determine starting epoch (resume from next epoch after saved state)
        # If we saved at end of epoch N, we resume from epoch N+1
        start_epoch = self.training_state.epoch + 1

        logger.info(
            f"Resuming training from epoch {start_epoch + 1}, "
            f"step {self.training_state.step}"
        )

        # Continue training
        self.train(num_epochs, start_epoch=start_epoch)

    def _train_epoch(self) -> bool:
        """Train for one epoch.

        Returns:
            True if training should stop (shutdown signal received), False otherwise
        """
        batch_iterator = tqdm(
            self.train_dataloader, desc=f"Epoch {self.training_state.epoch + 1} Training"
        )
        epoch_losses = []

        for step, batch in enumerate(batch_iterator):
            step_start_time = time.time()

            # Check for shutdown signal
            if self.killer.should_stop():
                logger.warning("Shutdown signal received during training step")
                save_emergency_checkpoint(self)
                return True  # Signal to stop training

            # Compute metrics BEFORE moving to device (avoids GPU-CPU sync)
            padding_ratio = self._compute_padding_ratio(batch)
            num_tokens = self._count_non_padding_tokens(batch)

            # Prepare inputs
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            source_mask = batch["source_mask"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            label = batch["label"].to(self.device)

            loss, grad_norm = self._training_step(
                step, source, target, source_mask, target_mask, label
            )

            current_step = self.training_state.step

            # Update progress bar
            batch_iterator.set_postfix(
                {"Loss": f"{loss:.4f}", "LR": self.optimizer.param_groups[0]["lr"]}
            )

            # === PER-STEP LOGGING ===
            self.tb_logger.log_training_step(
                loss, current_step, self.optimizer.param_groups[0]["lr"]
            )

            # Perplexity
            self.tb_logger.log_perplexity(loss, current_step)

            # Padding ratio
            self.tb_logger.log_padding_ratio(padding_ratio, current_step)

            # Throughput
            step_time = time.time() - step_start_time
            if step_time > 0:
                iters_per_sec = 1.0 / step_time
                tokens_per_sec = num_tokens / step_time
                self.tb_logger.log_throughput(current_step, iters_per_sec, tokens_per_sec)

            # Clip factor (if gradients were clipped this step)
            if grad_norm is not None:
                max_grad_norm = self.training_config.optimizer_config.max_grad_norm
                self.tb_logger.log_clip_factor(grad_norm, max_grad_norm, current_step)

            epoch_losses.append(loss)

            # === PERIODIC LOGGING (EVERY N STEPS) ===
            if (
                current_step > 0
                and current_step % self.logging_config.per_layer_metrics_frequency == 0
            ):
                self._log_periodic_metrics(current_step)

            self.training_state.step += 1

            # Save last_state periodically for spot instance resilience
            if self.training_state.step % self._checkpoint_every_n_steps == 0:
                save_last_state_checkpoint(self)

            if step % self.training_config.tensorboard_flush_frequency == 0:
                self.tb_logger.flush()

        return False  # Continue training

    def _validate_epoch(self) -> None:
        """Validate the model for one epoch.

        Computes and logs:
        - Average validation loss
        - Perplexity
        - Top-1 and Top-5 accuracy
        - BLEU and chrF scores
        - Length ratio (predicted/target)
        - Qualitative translation samples
        """
        self.model.eval()

        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_tokens = 0
        num_batches = 0

        # For BLEU/chrF calculation
        all_predictions: list[str] = []
        all_references: list[str] = []
        total_pred_length = 0
        total_ref_length = 0

        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        max_translations = self.logging_config.max_validation_translations

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)
                source_mask = batch["source_mask"].to(self.device)
                target_mask = batch["target_mask"].to(self.device)
                label = batch["label"].to(self.device)

                # Forward pass
                output = self.model(source, target, source_mask, target_mask)

                # Compute loss
                loss = self.loss_function(
                    output.view(-1, output.size(-1)), label.view(-1)
                )
                total_loss += loss.item()
                num_batches += 1

                # Compute top-1 and top-5 accuracy
                # Mask out padding positions
                non_pad_mask = (label != pad_id).view(-1)
                num_valid_tokens = non_pad_mask.sum().item()

                if num_valid_tokens > 0:
                    flat_output = output.view(-1, output.size(-1))
                    flat_label = label.view(-1)

                    # Top-1 accuracy
                    predictions = flat_output.argmax(dim=-1)
                    correct_top1 = (
                        (predictions == flat_label) & non_pad_mask
                    ).sum().item()
                    total_correct_top1 += correct_top1

                    # Top-5 accuracy
                    top5_preds = torch.topk(flat_output, k=min(5, flat_output.size(-1)), dim=-1).indices
                    correct_top5 = (
                        (top5_preds == flat_label.unsqueeze(-1)).any(dim=-1) & non_pad_mask
                    ).sum().item()
                    total_correct_top5 += correct_top5

                    total_tokens += num_valid_tokens

                # Generate translations for BLEU/chrF (limited for speed)
                if len(all_predictions) < max_translations:
                    # Check if raw text is available in batch
                    has_text = "source_text" in batch and "target_text" in batch

                    for i in range(source.size(0)):
                        if len(all_predictions) >= max_translations:
                            break

                        try:
                            if has_text:
                                source_text = batch["source_text"][i]
                                reference = batch["target_text"][i]
                            else:
                                # Decode from tokens if text not available
                                source_tokens = source[i].tolist()
                                source_tokens = [t for t in source_tokens if t != pad_id]
                                source_text = self.tokenizer.decode(source_tokens)

                                ref_tokens = label[i].tolist()
                                ref_tokens = [t for t in ref_tokens if t != pad_id]
                                reference = self.tokenizer.decode(ref_tokens)

                            prediction = self._translate_single(source_text)

                            all_predictions.append(prediction)
                            all_references.append(reference)

                            # Track lengths for ratio calculation
                            pred_tokens = len(self.tokenizer.encode(prediction))
                            ref_tokens_count = len(self.tokenizer.encode(reference))
                            total_pred_length += pred_tokens
                            total_ref_length += ref_tokens_count

                        except Exception as e:
                            logger.warning(f"Translation failed during validation: {e}")

        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        top1_accuracy = (
            (total_correct_top1 / total_tokens * 100) if total_tokens > 0 else 0.0
        )
        top5_accuracy = (
            (total_correct_top5 / total_tokens * 100) if total_tokens > 0 else 0.0
        )
        length_ratio = (
            total_pred_length / total_ref_length if total_ref_length > 0 else 1.0
        )

        # Compute BLEU and chrF
        bleu_score = 0.0
        chrf_score = 0.0
        if all_predictions and all_references:
            try:
                bleu_score = self.metrics.bleu(all_references, all_predictions)
                chrf_score = self.metrics.chrf(all_references, all_predictions)
            except Exception as e:
                logger.warning(f"Failed to compute BLEU/chrF: {e}")

        # Log all validation metrics
        step = self.training_state.step
        self.tb_logger.log_validation_step(avg_loss, step)
        self.tb_logger.log_perplexity(avg_loss, step, prefix="Metrics/Val")
        self.tb_logger.log_validation_metrics(
            step=step,
            bleu=bleu_score,
            chrf=chrf_score,
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
            length_ratio=length_ratio,
        )

        # Log qualitative samples
        self._log_qualitative_samples(step)

        # Update best validation loss
        if avg_loss < self.training_state.best_val_loss:
            self.training_state.best_val_loss = avg_loss
            logger.info(f"New best validation loss: {avg_loss:.4f}")

        logger.info(
            f"Validation - Loss: {avg_loss:.4f}, PPL: {math.exp(min(avg_loss, 100)):.2f}, "
            f"BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, "
            f"Top-1: {top1_accuracy:.1f}%, Top-5: {top5_accuracy:.1f}%"
        )

        self.model.train()

    def _training_step(
        self,
        step: int,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[float, Optional[float]]:
        """Execute a single training step.

        Returns:
            Tuple of (loss, grad_norm). grad_norm is None if not an accumulation step.
        """
        # Save time-based checkpoint if needed
        save_time_based_checkpoint(self)

        # Determine the appropriate dtype based on device capabilities
        dtype = (
            torch.bfloat16
            if self.device == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )

        grad_norm_value: Optional[float] = None

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
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.optimizer_config.max_grad_norm,
            )
            grad_norm_value = grad_norm_tensor.item()

            # Update parameters and reset gradients
            self.optimizer.step()
            if self.scheduler is None:
                raise ValueError("Scheduler not initialized.")
            self.scheduler.step()

            self.optimizer.zero_grad()

            # Log gradient norm
            self.tb_logger.log_gradient_norm(grad_norm_value, self.training_state.step)

        return loss.item(), grad_norm_value
