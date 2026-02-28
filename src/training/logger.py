"""Enhanced TensorBoard logging for transformer training.

Provides comprehensive logging including:
- Per-step metrics: loss, perplexity, throughput, padding ratio, clip factor
- Periodic metrics: layer-wise gradient/weight norms, attention maps
- Validation metrics: BLEU, chrF, accuracy, text samples
"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


class TrainingLogger:
    """TensorBoard logger with enhanced metrics for transformer training."""

    def __init__(self, log_dir: str):
        """Initialize the logger.

        Args:
            log_dir: Directory for TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    # ============== PER-STEP METRICS ==============

    def log_training_step(self, loss: float, step: int, lr: float) -> None:
        """Log training loss and learning rate.

        Args:
            loss: Training loss value.
            step: Current training step.
            lr: Current learning rate.
        """
        self.writer.add_scalar("Loss/train", loss, step)
        self.writer.add_scalar("Train/Learning_Rate", lr, step)

    def log_perplexity(self, loss: float, step: int, prefix: str = "Metrics") -> None:
        """Log perplexity (exp of loss).

        Args:
            loss: Loss value to convert to perplexity.
            step: Current training step.
            prefix: Tag prefix for TensorBoard (default "Metrics").
        """
        # Cap loss to prevent overflow (exp(100) is already astronomical)
        capped_loss = min(loss, 100.0)
        ppl = math.exp(capped_loss)
        self.writer.add_scalar(f"{prefix}/PPL", ppl, step)

    def log_throughput(
        self,
        step: int,
        iters_per_sec: float,
        tokens_per_sec: float,
    ) -> None:
        """Log training throughput metrics.

        Args:
            step: Current training step.
            iters_per_sec: Training iterations per second.
            tokens_per_sec: Tokens processed per second.
        """
        self.writer.add_scalar("Throughput/Iters_per_sec", iters_per_sec, step)
        self.writer.add_scalar("Throughput/Tokens_per_sec", tokens_per_sec, step)

    def log_padding_ratio(self, padding_ratio: float, step: int) -> None:
        """Log padding ratio (efficiency metric).

        Args:
            padding_ratio: Ratio of padding tokens in the batch.
            step: Current training step.
        """
        self.writer.add_scalar("Efficiency/Padding_Ratio", padding_ratio, step)

    def log_clip_factor(
        self, original_norm: float, max_norm: float, step: int
    ) -> None:
        """Log gradient clipping factor.

        Args:
            original_norm: Original gradient norm before clipping.
            max_norm: Maximum allowed gradient norm.
            step: Current training step.
        """
        clip_factor = original_norm / max_norm if max_norm > 0 else 0.0
        self.writer.add_scalar("Health/Clip_Factor", clip_factor, step)
        # Also log whether clipping occurred (1.0 if clipped, 0.0 if not)
        self.writer.add_scalar("Health/Clipped", float(clip_factor > 1.0), step)

    def log_gradient_norm(self, grad_norm: float, step: int) -> None:
        """Log overall gradient norm.

        Args:
            grad_norm: Gradient norm value.
            step: Current training step.
        """
        self.writer.add_scalar("Health/Grad_Norm", grad_norm, step)

    # ============== PERIODIC METRICS (EVERY N STEPS) ==============

    def log_layer_gradient_norms(self, model: nn.Module, step: int) -> None:
        """Log gradient norms per layer group.

        Separates gradients into embeddings, encoder, and decoder groups.

        Args:
            model: The transformer model.
            step: Current training step.
        """
        embedding_norms = []
        encoder_norms = []
        decoder_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                if "embedding" in name.lower():
                    embedding_norms.append(grad_norm)
                elif "encoder" in name.lower():
                    encoder_norms.append(grad_norm)
                elif "decoder" in name.lower() or "projection" in name.lower():
                    decoder_norms.append(grad_norm)

        # Log average norms per group
        if embedding_norms:
            avg_norm = sum(embedding_norms) / len(embedding_norms)
            self.writer.add_scalar("Health/Grad_Norm/Embeddings", avg_norm, step)

        if encoder_norms:
            avg_norm = sum(encoder_norms) / len(encoder_norms)
            self.writer.add_scalar("Health/Grad_Norm/Encoder", avg_norm, step)

        if decoder_norms:
            avg_norm = sum(decoder_norms) / len(decoder_norms)
            self.writer.add_scalar("Health/Grad_Norm/Decoder", avg_norm, step)

    def log_weight_norms(self, model: nn.Module, step: int) -> None:
        """Log L2 norm of weights per layer group.

        Args:
            model: The transformer model.
            step: Current training step.
        """
        embedding_norms = []
        encoder_norms = []
        decoder_norms = []

        for name, param in model.named_parameters():
            weight_norm = param.data.norm().item()

            if "embedding" in name.lower():
                embedding_norms.append(weight_norm)
            elif "encoder" in name.lower():
                encoder_norms.append(weight_norm)
            elif "decoder" in name.lower() or "projection" in name.lower():
                decoder_norms.append(weight_norm)

        if embedding_norms:
            avg_norm = sum(embedding_norms) / len(embedding_norms)
            self.writer.add_scalar("Health/Weight_Norm/Embeddings", avg_norm, step)

        if encoder_norms:
            avg_norm = sum(encoder_norms) / len(encoder_norms)
            self.writer.add_scalar("Health/Weight_Norm/Encoder", avg_norm, step)

        if decoder_norms:
            avg_norm = sum(decoder_norms) / len(decoder_norms)
            self.writer.add_scalar("Health/Weight_Norm/Decoder", avg_norm, step)

    def log_weight_histograms(self, model: nn.Module, step: int) -> None:
        """Log weight histograms for all parameters.

        Note: This is expensive due to CPU transfer. Use sparingly.

        Args:
            model: The transformer model.
            step: Current training step.
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param.data.cpu().numpy(), step)

    def log_attention_maps(
        self,
        encoder_attentions: list,
        decoder_attentions: dict,
        source_tokens: list[str],
        target_tokens: list[str],
        step: int,
        sentence_idx: int = 0,
    ) -> None:
        """Log attention maps to TensorBoard as images.

        Args:
            encoder_attentions: List of attention tensors per encoder layer.
                Each tensor shape: (batch, n_heads, src_len, src_len)
            decoder_attentions: Dict with 'self_attentions' and 'cross_attentions'.
                Each list contains tensors per decoder layer.
            source_tokens: Tokenized source sentence as strings.
            target_tokens: Tokenized target sentence as strings.
            step: Current training step.
            sentence_idx: Index of the probe sentence (for naming).
        """
        # Log encoder self-attention (average across heads, final layer)
        if encoder_attentions:
            final_enc_attn = encoder_attentions[-1]
            if final_enc_attn is not None and len(final_enc_attn) > 0:
                # Handle both list and tensor formats
                if isinstance(final_enc_attn, list):
                    attn_tensor = final_enc_attn[0]
                else:
                    attn_tensor = final_enc_attn

                if attn_tensor is not None and attn_tensor.numel() > 0:
                    # Shape: (batch, n_heads, src_len, src_len) -> average heads
                    avg_attn = attn_tensor[0].mean(dim=0)  # (src_len, src_len)
                    fig = self._create_attention_figure(
                        avg_attn.detach().cpu(),
                        source_tokens,
                        source_tokens,
                        f"Encoder Self-Attention (Sentence {sentence_idx})",
                    )
                    self.writer.add_figure(
                        f"Attention/Encoder_Self/Sentence_{sentence_idx}",
                        fig,
                        step,
                    )
                    plt.close(fig)

        # Log decoder cross-attention (average across heads, final layer)
        cross_attns = decoder_attentions.get("cross_attentions", [])
        if cross_attns:
            final_cross_attn = cross_attns[-1]
            if final_cross_attn is not None and final_cross_attn.numel() > 0:
                # Shape: (batch, n_heads, tgt_len, src_len) -> average heads
                avg_attn = final_cross_attn[0].mean(dim=0)  # (tgt_len, src_len)
                fig = self._create_attention_figure(
                    avg_attn.detach().cpu(),
                    source_tokens,
                    target_tokens,
                    f"Decoder Cross-Attention (Sentence {sentence_idx})",
                )
                self.writer.add_figure(
                    f"Attention/Decoder_Cross/Sentence_{sentence_idx}",
                    fig,
                    step,
                )
                plt.close(fig)

    def log_decoder_only_attention_maps(
        self,
        attentions: list,
        tokens: list[str],
        step: int,
        prompt_idx: int = 0,
    ) -> None:
        """Log decoder-only self-attention maps to TensorBoard.

        Args:
            attentions: List of attention tensors per layer.
                Each tensor shape: (batch, n_heads, seq_len, seq_len)
            tokens: Token strings for axis labels.
            step: Current training step.
            prompt_idx: Index of the prompt (for naming).
        """
        if not attentions:
            return

        # Log final layer attention (averaged across heads)
        final_attn = attentions[-1]
        if final_attn is not None and final_attn.numel() > 0:
            avg_attn = final_attn[0].mean(dim=0)  # (seq_len, seq_len)
            fig = self._create_attention_figure(
                avg_attn.detach().cpu(),
                tokens,
                tokens,
                f"Self-Attention Final Layer (Prompt {prompt_idx})",
            )
            self.writer.add_figure(
                f"Attention/Decoder_Self_Avg/Prompt_{prompt_idx}",
                fig,
                step,
            )
            plt.close(fig)

        # Log per-head attention for final layer (first 4 heads)
        if final_attn is not None and final_attn.numel() > 0:
            n_heads = min(4, final_attn.shape[1])
            for h in range(n_heads):
                head_attn = final_attn[0, h]  # (seq_len, seq_len)
                fig = self._create_attention_figure(
                    head_attn.detach().cpu(),
                    tokens,
                    tokens,
                    f"Self-Attention Head {h} (Prompt {prompt_idx})",
                )
                self.writer.add_figure(
                    f"Attention/Decoder_Head_{h}/Prompt_{prompt_idx}",
                    fig,
                    step,
                )
                plt.close(fig)

    def _create_attention_figure(
        self,
        attention: torch.Tensor,
        x_labels: list[str],
        y_labels: list[str],
        title: str,
    ) -> plt.Figure:
        """Create a matplotlib figure for attention visualization.

        Args:
            attention: 2D attention tensor (tgt_len, src_len).
            x_labels: Labels for x-axis (source positions).
            y_labels: Labels for y-axis (target positions).
            title: Figure title.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        attn_np = attention.numpy()
        im = ax.imshow(attn_np, aspect="auto", cmap="viridis")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Source Position")
        ax.set_ylabel("Target Position")

        # Limit number of ticks for readability
        max_ticks = 20
        if len(x_labels) > max_ticks:
            x_idx = list(range(0, len(x_labels), len(x_labels) // max_ticks))
        else:
            x_idx = list(range(len(x_labels)))

        if len(y_labels) > max_ticks:
            y_idx = list(range(0, len(y_labels), len(y_labels) // max_ticks))
        else:
            y_idx = list(range(len(y_labels)))

        ax.set_xticks(x_idx)
        ax.set_xticklabels(
            [x_labels[i][:10] if i < len(x_labels) else "" for i in x_idx],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(y_idx)
        ax.set_yticklabels(
            [y_labels[i][:10] if i < len(y_labels) else "" for i in y_idx],
            fontsize=8,
        )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        return fig

    # ============== VALIDATION METRICS ==============

    def log_validation_step(self, val_loss: float, step: int) -> None:
        """Log validation loss.

        Args:
            val_loss: Validation loss value.
            step: Current training step.
        """
        self.writer.add_scalar("Loss/val", val_loss, step)

    def log_validation_metrics(
        self,
        step: int,
        bleu: Optional[float] = None,
        chrf: Optional[float] = None,
        top1_accuracy: Optional[float] = None,
        top5_accuracy: Optional[float] = None,
        length_ratio: Optional[float] = None,
    ) -> None:
        """Log validation evaluation metrics.

        Args:
            step: Current training step.
            bleu: BLEU score (optional).
            chrf: chrF score (optional).
            top1_accuracy: Top-1 accuracy percentage (optional).
            top5_accuracy: Top-5 accuracy percentage (optional).
            length_ratio: Predicted/target length ratio (optional).
        """
        if bleu is not None:
            self.writer.add_scalar("Metrics/BLEU", bleu, step)
        if chrf is not None:
            self.writer.add_scalar("Metrics/chrF", chrf, step)
        if top1_accuracy is not None:
            self.writer.add_scalar("Metrics/Top1_Accuracy", top1_accuracy, step)
        if top5_accuracy is not None:
            self.writer.add_scalar("Metrics/Top5_Accuracy", top5_accuracy, step)
        if length_ratio is not None:
            self.writer.add_scalar("Metrics/Length_Ratio", length_ratio, step)

    def log_text_samples(
        self,
        samples: list[tuple[str, str, str]],
        step: int,
        tag_prefix: str = "Translations",
    ) -> None:
        """Log qualitative text samples as markdown table.

        Args:
            samples: List of (source, reference, prediction) tuples.
            step: Current training step.
            tag_prefix: Tag prefix for TensorBoard (default "Translations").
        """
        markdown_text = "| Source | Reference | Prediction |\n"
        markdown_text += "|--------|-----------|------------|\n"

        for source, reference, prediction in samples:
            # Escape pipe characters for markdown table
            source = source.replace("|", "\\|")
            reference = reference.replace("|", "\\|")
            prediction = prediction.replace("|", "\\|")
            markdown_text += f"| {source} | {reference} | {prediction} |\n"

        self.writer.add_text(f"{tag_prefix}/Samples", markdown_text, step)

    # ============== UTILITY METHODS ==============

    def flush(self) -> None:
        """Flush pending events to TensorBoard."""
        self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
