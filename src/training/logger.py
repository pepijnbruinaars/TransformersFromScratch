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

        Loss and LR are both written under the "Loss/" namespace so TensorBoard
        renders them in the same section with a shared step x-axis, making it
        easy to correlate LR schedule changes with loss behaviour.

        Args:
            loss: Training loss value.
            step: Current training step.
            lr: Current learning rate.
        """
        self.writer.add_scalar("Loss/train", loss, step)
        # LR in the same "Loss/" group → same step axis, same TensorBoard section
        self.writer.add_scalar("Loss/LR", lr, step)
        # Standalone chart for zooming into the LR schedule in detail
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

    def log_batch_stats(self, seq_len: int, unique_tokens: int, step: int) -> None:
        """Log per-batch sequence-length and vocabulary-diversity metrics.

        Sequence length tracks how much of the context window each batch
        actually uses (shorter = more padding waste with the standard dataset;
        always max_length with sequence packing).  Unique tokens per batch
        measures vocabulary diversity — a healthy batch should cover a broad
        slice of the vocabulary.

        Args:
            seq_len: Sequence length processed by the model this step
                     (= padded/packed length, i.e. input_ids.shape[1]).
            unique_tokens: Number of distinct token IDs in input_ids.
            step: Current training step.
        """
        self.writer.add_scalar("Efficiency/Seq_Length", seq_len, step)
        self.writer.add_scalar("Efficiency/Unique_Tokens", unique_tokens, step)

    def log_loss_spike(self, loss: float, ema_loss: float, step: int) -> None:
        """Log a loss-spike marker for anomalous training steps.

        Only called when the current loss is significantly above the recent
        exponential moving average.  Writing to the same step as the normal
        training loss allows TensorBoard to overlay the spike markers on the
        loss curve when both tags are viewed together.

        Args:
            loss: Raw loss at this step (the spike value).
            ema_loss: Smoothed EMA loss used as the baseline.
            step: Current training step.
        """
        self.writer.add_scalar("Loss/spike", loss, step)
        # Ratio > 1.0 = spike; visible at a glance in Health/
        self.writer.add_scalar("Health/Loss_Spike_Ratio", loss / max(ema_loss, 1e-8), step)

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

    # ============== OPTIMIZATION DYNAMICS ==============

    def log_update_ratios(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int
    ) -> None:
        """Log parameter update-to-weight ratios using Adam's moment estimates.

        The effective per-step update magnitude for Adam is approximately
        α * m̂_t / (√v̂_t + ε). Dividing by the parameter norm gives a
        scale-free ratio. Values around 1e-3 are healthy; consistently
        above 1e-2 suggests too-high LR, below 1e-4 suggests too-low LR.

        Uses the optimizer's stored exponential moving averages directly,
        so no parameter snapshot is needed.

        Args:
            model: The transformer model.
            optimizer: Adam/AdamW optimizer with stored moment estimates.
            step: Current training step.
        """
        param_map = {id(p): name for name, p in model.named_parameters()}

        group_ratios: dict[str, list[float]] = {}
        for group in optimizer.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state is None or "exp_avg" not in state:
                    continue
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                step_count = state.get("step", 1)
                # Bias-corrected estimates
                beta1 = group.get("betas", (0.9, 0.999))[0]
                beta2 = group.get("betas", (0.9, 0.999))[1]
                bc1 = 1.0 - beta1 ** step_count
                bc2 = 1.0 - beta2 ** step_count
                m_hat = m / bc1
                v_hat = v / bc2
                eps = group.get("eps", 1e-8)
                update_norm = (lr * m_hat / (v_hat.sqrt() + eps)).norm().item()
                weight_norm = p.data.norm().item()
                if weight_norm < 1e-12:
                    continue
                ratio = update_norm / weight_norm

                name = param_map.get(id(p), "unknown")
                if "embedding" in name:
                    group_ratios.setdefault("Embedding", []).append(ratio)
                elif "decoder_stack" in name:
                    group_ratios.setdefault("Decoder", []).append(ratio)
                else:
                    group_ratios.setdefault("Other", []).append(ratio)

        for group_name, ratios in group_ratios.items():
            avg = sum(ratios) / len(ratios)
            self.writer.add_scalar(f"Optimization/Update_Ratio/{group_name}", avg, step)

    def log_attention_entropy(self, attentions: list, step: int) -> None:
        """Log entropy of attention distributions per layer.

        Entropy H = -Σ p·log(p) over the attention weight distribution.
        Low entropy → focused head (attending to few positions).
        High entropy (≈ log(seq_len)) → diffuse/uniform head.
        Tracking this over training shows when heads specialise.

        Args:
            attentions: List of attention tensors per layer, each shape
                        (batch, n_heads, seq_len, seq_len). May be None
                        if flash attention was used (weights not returned).
            step: Current training step.
        """
        valid_layers = [a for a in attentions if a is not None and a.numel() > 0]
        if not valid_layers:
            return

        for layer_idx, attn in enumerate(valid_layers):
            # attn: (batch, n_heads, seq_len, seq_len)
            # Clamp to avoid log(0)
            attn_clamped = attn.clamp(min=1e-9)
            # Entropy per (batch, head, query position), then average over batch & position
            entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (batch, heads, seq_len)
            mean_entropy = entropy.mean().item()
            self.writer.add_scalar(f"Attention/Entropy/Layer_{layer_idx}", mean_entropy, step)

        # Per-head entropy for the final layer only
        final_attn = valid_layers[-1]
        final_clamped = final_attn.clamp(min=1e-9)
        per_head_entropy = -(final_clamped * final_clamped.log()).sum(dim=-1).mean(dim=(0, 2))
        for h, h_entropy in enumerate(per_head_entropy):
            self.writer.add_scalar(
                f"Attention/Entropy/FinalLayer_Head_{h}", h_entropy.item(), step
            )

    def log_per_position_loss(self, per_position_loss: torch.Tensor, step: int) -> None:
        """Log average loss bucketed by sequence position.

        Reveals whether the model struggles more at the beginning (lack of
        context) or end (long-range dependencies) of sequences.
        The tensor is bucketed into 8 equal-width groups so that the chart
        is readable regardless of sequence length.

        Args:
            per_position_loss: 1-D float tensor of length seq_len, already
                               averaged across the batch (CPU tensor).
            step: Current training step.
        """
        n = per_position_loss.shape[0]
        n_buckets = min(8, n)
        bucket_size = n // n_buckets
        for b in range(n_buckets):
            start = b * bucket_size
            end = start + bucket_size if b < n_buckets - 1 else n
            bucket_loss = per_position_loss[start:end].mean().item()
            self.writer.add_scalar(f"Loss/Position_Bucket_{b}", bucket_loss, step)

    def log_generation_stats(
        self, stats: dict[float, dict[str, float]], step: int
    ) -> None:
        """Log generation quality metrics (distinct-n) per temperature.

        Distinct-1 and distinct-2 measure vocabulary diversity in generated
        text: |unique n-grams| / |total n-grams|. Values near 1.0 are ideal;
        low values (< 0.3) indicate repetition or mode collapse.

        Args:
            stats: {temperature: {"distinct_1": float, "distinct_2": float}}
            step: Current training step.
        """
        for temp, metrics in stats.items():
            tag = f"Generation/Temp_{temp:.1f}"
            if "distinct_1" in metrics:
                self.writer.add_scalar(f"{tag}/Distinct_1", metrics["distinct_1"], step)
            if "distinct_2" in metrics:
                self.writer.add_scalar(f"{tag}/Distinct_2", metrics["distinct_2"], step)

    def log_gpu_memory(self, step: int) -> None:
        """Log GPU memory usage (allocated and reserved) in MB.

        Args:
            step: Current training step.
        """
        if not torch.cuda.is_available():
            return
        allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
        self.writer.add_scalar("System/GPU_Memory_Allocated_MB", allocated_mb, step)
        self.writer.add_scalar("System/GPU_Memory_Reserved_MB", reserved_mb, step)

    # ============== UTILITY METHODS ==============

    def flush(self) -> None:
        """Flush pending events to TensorBoard."""
        self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
