from datetime import datetime
import math
from pathlib import Path
import json

from tqdm import tqdm  # type: ignore
from ..constants import PAD_TOKEN, START_TOKEN, END_TOKEN
from .load_data import get_max_sequence_length, load_opus_data, get_sentences_from_data
from ..tokenizer import CustomTokenizer
from ..transformer import Transformer
from .dataset import CustomDataset
from .dataset import attention_mask
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import logging
import matplotlib.pyplot as plt
from .train_utils import greedy_decode_single, calculate_bleu_chrf, score_sample_from_validation

# Prefer TF32 on CUDA-capable GPUs to improve stability/performance on large matmuls.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Configure Python logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_or_load_tokenizer(
    dutch_sentences: list[str], english_sentences: list[str]
) -> CustomTokenizer:
    shared_tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")

    if not shared_tokenizer.trained:
        logger.info("Training shared tokenizer on both English and Dutch sentences...")
        combined_sentences = english_sentences + dutch_sentences
        shared_tokenizer.train(combined_sentences)
        shared_tokenizer.save("models/tokenizers/shared_tokenizer.json")

    logger.info("Testing shared tokenizer on English samples...")
    sample_english_sentences = english_sentences[:10]
    for sentence in sample_english_sentences:
        shared_tokenizer.print_tokens(sentence)

    logger.info("Testing shared tokenizer on Dutch samples...")
    sample_dutch_sentences = dutch_sentences[:10]
    for sentence in sample_dutch_sentences:
        shared_tokenizer.print_tokens(sentence)

    return shared_tokenizer


def learning_rate_lambda(current_step: int, total_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def greedy_decode(
    transformer: Transformer,
    encoder_output: torch.Tensor,
    src_mask: torch.Tensor,
    start_id: int,
    end_id: int,
    pad_id: int,
    device: str,
    max_len: int = 64,
):
    """Greedy decode using the transformer.decode + project API.

    Returns (generated_ids, last_decoder_attentions, ys_tensor)
    """
    ys = torch.tensor([[start_id]], dtype=torch.int64).to(device)
    generated_ids: list[int] = []
    last_dec_atts = None
    for _ in range(max_len):
        dec_mask = (ys != pad_id).unsqueeze(0).unsqueeze(0).int() & attention_mask(ys.size(1)).to(device)
        dec_result = transformer.decode(ys, encoder_output, src_mask, dec_mask, return_attentions=True)
        if isinstance(dec_result, tuple):
            dec_out, last_dec_atts = dec_result
        else:
            dec_out = dec_result
            last_dec_atts = None

        logits = transformer.project(dec_out)
        next_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        next_id_item = int(next_id.item())
        if next_id_item == end_id:
            break
        generated_ids.append(next_id_item)
        ys = torch.cat([ys, next_id], dim=1)

    return generated_ids, last_dec_atts, ys


def plot_per_head_cross_attention(
    last_layer_attn: torch.Tensor,
    src_seq: torch.Tensor,
    ys: torch.Tensor,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
):
    """Create a matplotlib figure with one subplot per head showing tgt_len x src_len cross-attention.

    `last_layer_attn` expected shape: (batch, n_heads, tgt_len, src_len)
    """
    attn = last_layer_attn[0].detach().cpu()  # (n_heads, tgt_len, src_len)
    source_pad_id = source_tokenizer.token_to_id(PAD_TOKEN)
    src_len = int((src_seq != source_pad_id).sum().item())
    tgt_len = ys.size(1)

    attn_trimmed = attn[:, :tgt_len, :src_len]
    num_heads = attn_trimmed.shape[0]

    fig, axes = plt.subplots(nrows=num_heads, ncols=1, figsize=(8, 2 * num_heads))
    if num_heads == 1:
        axes = [axes]

    im = None
    # Prepare token labels (trimmed)
    src_tokens = [source_tokenizer.id_to_token(int(t)) for t in src_seq[:src_len]]
    tgt_tokens = [target_tokenizer.id_to_token(int(t)) for t in ys.squeeze(0).tolist()]

    # Thinning function to avoid too many ticks
    def thin_indices(n: int, max_ticks: int = 24):
        if n <= max_ticks:
            return list(range(n))
        step = max(1, n // max_ticks)
        return list(range(0, n, step))

    x_idx = thin_indices(src_len)
    y_idx = thin_indices(tgt_len)

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_trimmed[h].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"head {h}")
        ax.set_xlabel("source position")
        ax.set_ylabel("target position")
        # Set tick labels (thin if needed)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([src_tokens[i] for i in x_idx], rotation=90, fontsize=8)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([tgt_tokens[i] for i in y_idx], fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02)

    return fig


def log_sample_and_attention(
    writer: SummaryWriter,
    transformer: Transformer,
    sample: dict,
    tokenizer: CustomTokenizer,
    device: str,
    epoch: int,
    sample_idx: int,
):
    """Log one sample's SRC/TGT/PRED and per-head cross-attention to TensorBoard."""
    src = sample["source"].unsqueeze(0).to(device)
    src_mask = sample["source_mask"].to(device)

    encoder_result = transformer.encode(src, src_mask, return_attentions=True)
    if isinstance(encoder_result, tuple):
        encoder_output, _ = encoder_result
    else:
        encoder_output = encoder_result

    start_id = tokenizer.token_to_id(START_TOKEN)
    end_id = tokenizer.token_to_id(END_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)

    generated_ids, dec_atts, ys = greedy_decode(
        transformer, encoder_output, src_mask, start_id, end_id, pad_id, device
    )

    pred_text = tokenizer.decode(generated_ids)
    src_text = sample.get("source_text", "")
    tgt_text = sample.get("target_text", "")

    writer.add_text(f"samples/{sample_idx}", f"SRC: {src_text}\nTGT: {tgt_text}\nPRED: {pred_text}", global_step=epoch)

    if dec_atts is not None and isinstance(dec_atts, dict):
        cross_atts = dec_atts.get("cross_attentions")
        if cross_atts:
            last_layer_attn = cross_atts[-1]
            fig = plot_per_head_cross_attention(last_layer_attn, sample["source"], ys, tokenizer, tokenizer)
            writer.add_figure(f"attention/sample_{sample_idx}", fig, global_step=epoch)
            plt.close(fig)


def train_model(
    transformer: Transformer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    tokenizer: CustomTokenizer,
    model_config: dict,
    nr_epochs: int = 20,
    learning_rate: float = 5e-5,
) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")
    transformer.to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"models/transformer/{timestamp}"
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models/transformer").mkdir(parents=True, exist_ok=True)

    config_path = f"{model_folder}/model_config.json"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Model configuration saved to {config_path}")

    writer = SummaryWriter(log_dir=model_folder)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)

    total_steps = nr_epochs * len(train_dataloader)
    warmup_steps = int(0.05 * total_steps)
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warm-up steps: {warmup_steps}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_lambda(step, total_steps, warmup_steps),
    )

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id(PAD_TOKEN),
        label_smoothing=0.1,
    ).to(device)

    # Create GradScaler once outside the training loop
    loss_scaler = torch.GradScaler("cuda") if device == "cuda" else None

    iterations = 0
    loss_history = []
    epoch_loss_history = []
    validation_loss_history = []
    best_val_loss = float('inf')

    for epoch in range(nr_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{nr_epochs}")
        transformer.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{nr_epochs}")
        epoch_losses = []
        for batch in batch_iterator:
            source, target, source_mask, target_mask, label = (
                batch["source"].to(device),
                batch["target"].to(device),
                batch["source_mask"].to(device),
                batch["target_mask"].to(device),
                batch["label"].to(device),
            )
            if loss_scaler is not None:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    projection = transformer(
                        source, target, source_mask, target_mask
                    )
                    loss = loss_function(
                        projection.view(-1, tokenizer.vocabulary_size), label.view(-1)
                    )

                loss_scaler.scale(loss).backward()
                loss_scaler.unscale_(optimizer)

                loss_value = loss.item()
                # Clip and capture gradient norm before optimizer step / zero_grad
                grad_norm_value = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                projection = transformer(
                    source, target, source_mask, target_mask
                )
                loss = loss_function(
                    projection.view(-1, tokenizer.vocabulary_size), label.view(-1)
                )
                loss.backward()
                
                loss_value = loss.item()
                grad_norm_value = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()

            ips = float(batch_iterator.format_dict.get("rate", 0.0) or 0.0)
            iter_time_ms = 1000.0 / ips if ips > 0 else 0.0

            batch_iterator.set_postfix({"loss": f"{loss_value:.6f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            
            # Log to TensorBoard only every 10 iterations to avoid I/O bottleneck
            if iterations % 10 == 0:
                writer.add_scalar("train/loss", loss_value, iterations)
                writer.add_scalar("speed/iters_per_sec", ips, iterations)
                writer.add_scalar("speed/iter_time_ms", iter_time_ms, iterations)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], iterations)
                writer.add_scalar("train/perplexity", math.exp(loss_value), iterations)
                writer.add_scalar("train/grad_norm", float(grad_norm_value), iterations)
                writer.flush()
            
            epoch_losses.append(loss_value)

            iterations += 1

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        epoch_loss_history.append({"epoch": epoch + 1, "avg_loss": avg_epoch_loss})
        
        transformer.eval()
        val_losses = []
        
        with torch.no_grad():
            val_iterator = tqdm(validation_dataloader, desc=f"Validation {epoch + 1}/{nr_epochs}", leave=False)
            for val_batch in val_iterator:
                val_source = val_batch["source"].to(device)
                val_target = val_batch["target"].to(device)
                val_source_mask = val_batch["source_mask"].to(device)
                val_target_mask = val_batch["target_mask"].to(device)
                val_label = val_batch["label"].to(device)
                
                # Calculate validation loss
                val_projection = transformer(
                    val_source, val_target, val_source_mask, val_target_mask
                )
                val_loss = loss_function(
                    val_projection.view(-1, tokenizer.vocabulary_size), val_label.view(-1)
                )
                
                if torch.isfinite(val_loss):
                    val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        validation_loss_history.append({"epoch": epoch + 1, "val_loss": avg_val_loss})
        
        # Calculate BLEU and chrF scores on a random sample (much faster!)
        bleu_score, chrf_score = score_sample_from_validation(
            transformer, validation_dataloader, tokenizer, device, num_samples=150
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BLEU: {bleu_score:.4f} | chrF: {chrf_score:.4f}")
        writer.add_scalars("loss/epoch", {"train": avg_epoch_loss, "validation": avg_val_loss}, epoch)
        writer.add_scalar("validation/bleu_score", bleu_score, epoch)
        writer.add_scalar("validation/chrf_score", chrf_score, epoch)
        writer.flush()

        # --- Log sample translations + attention maps (refactored) ---
        try:
            n_samples = 3
            with torch.no_grad():
                for i in range(min(n_samples, len(validation_dataloader.dataset))):
                    sample = validation_dataloader.dataset[i]
                    log_sample_and_attention(
                        writer,
                        transformer,
                        sample,
                        tokenizer,
                        device,
                        epoch + 1,
                        i,
                    )
        except Exception as e:
            logger.warning(f"Could not log sample translations/attentions: {e}")
        
        model_path = f"{model_folder}/transformer_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model_state_dict": transformer.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_path,
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"{model_folder}/transformer_best.pt"
            torch.save(
                {
                    "model_state_dict": transformer.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                best_model_path,
            )
            logger.info(f"✓ Best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            logger.info(f"Model checkpoint saved for epoch {epoch + 1}")

    final_model_path = f"{model_folder}/transformer_final.pt"
    torch.save(
        {
            "model_state_dict": transformer.state_dict(),
            "epoch": nr_epochs,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )
    
    loss_history_path = f"{model_folder}/loss_history.json"
    epoch_loss_history_path = f"{model_folder}/epoch_loss_history.json"
    validation_loss_history_path = f"{model_folder}/validation_loss_history.json"
    
    with open(loss_history_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    
    with open(epoch_loss_history_path, "w") as f:
        json.dump(epoch_loss_history, f, indent=2)
    
    with open(validation_loss_history_path, "w") as f:
        json.dump(validation_loss_history, f, indent=2)
    
    logger.info(f"\nLoss history saved to {loss_history_path}")
    logger.info(f"Epoch loss history saved to {epoch_loss_history_path}")
    logger.info(f"Validation loss history saved to {validation_loss_history_path}")

    logger.info("Training complete!")


def main() -> None:
    full_dataset = load_opus_data(1.0, 0.0, 0.0)
    _, english_sentences, dutch_sentences = get_sentences_from_data(full_dataset[0])

    tokenizer = train_or_load_tokenizer(
        dutch_sentences, english_sentences
    )

    train_raw, validation_raw, test_raw = load_opus_data(0.7, 0.15, 0.15)

    source_length, target_length = get_max_sequence_length(
        full_dataset[0], tokenizer, tokenizer
    )

    max_sequence_length = min(
        max(source_length, target_length), 512
    )

    train = CustomDataset(
        train_raw,  # type: ignore
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        sequence_length=max_sequence_length,
    )
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        sequence_length=max_sequence_length,
    )

    # On Windows, num_workers=0 is faster. Use prefetch_factor for async data loading
    train_dataloader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    validation_dataloader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    n_blocks = 6
    d_model = 512
    d_ff = 2048
    n_heads = 8
    dropout = 0.1
    
    model_config = {
        "n_blocks": n_blocks,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "dropout": dropout,
        "source_length": max_sequence_length,
        "target_length": max_sequence_length,
        "vocabulary_size": tokenizer.vocabulary_size,
    }
    
    transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        source_length=max_sequence_length,
        target_length=max_sequence_length,
        source_vocabulary_size=tokenizer.vocabulary_size,
        target_vocabulary_size=tokenizer.vocabulary_size,
    )
    
    logger.info(transformer)

    train_model(
        transformer=transformer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        tokenizer=tokenizer,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
