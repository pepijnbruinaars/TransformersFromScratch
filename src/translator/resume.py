from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from ..tokenization.tokenizer import CustomTokenizer
from ..models import Transformer
from ..constants import PAD_TOKEN
from .train_utils import (
    load_checkpoint,
    restore_rng_states,
    find_latest_checkpoint,
    find_latest_run,
    build_dataloaders,
    create_optimizer,
    create_scheduler,
    create_loss_function,
    score_sample_from_validation,
    log_training_iteration,
    log_epoch_metrics,
    load_loss_histories,
    save_loss_histories,
)
from ..utils import get_device

def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training from a checkpoint")
    parser.add_argument("--run-folder", type=str, default=None, help="Path to the run folder (models/transformer/...)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename inside run folder (e.g. transformer_epoch_7.pt). If omitted, the latest checkpoint is used.")
    parser.add_argument("--additional-epochs", type=int, default=5, help="Number of additional epochs to run")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu). If omitted, auto-detected")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for dataloaders")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Initial learning rate (scheduler will be recomputed)")

    args = parser.parse_args()

    # Determine run folder
    run_folder = args.run_folder
    if run_folder is None:
        run_folder = find_latest_run()
        if run_folder is None:
            raise SystemExit("No run folder found in models/transformer")

    run_folder = str(run_folder)

    # Determine checkpoint path
    checkpoint_name = args.checkpoint
    if checkpoint_name is None:
        ckpt_path = find_latest_checkpoint(run_folder)
        if ckpt_path is None:
            raise SystemExit(f"No checkpoint found in {run_folder}")
    else:
        ckpt_path = str(Path(run_folder) / checkpoint_name)

    device = args.device or get_device()

    # Load model config
    config_path = Path(run_folder) / "model_config.json"
    if not config_path.exists():
        raise SystemExit(f"Model config not found at {config_path}")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    # Load tokenizer (shared)
    tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")

    # Build dataloaders
    sequence_length = model_config.get("source_length", model_config.get("target_length", 128))
    train_dataloader, validation_dataloader = build_dataloaders(tokenizer, sequence_length, batch_size=args.batch_size)

    # Build model
    transformer = Transformer(
        n_blocks=model_config.get("n_blocks", 6),
        d_model=model_config.get("d_model", 512),
        d_ff=model_config.get("d_ff", 2048),
        n_heads=model_config.get("n_heads", 8),
        dropout=model_config.get("dropout", 0.1),
        source_length=sequence_length,
        target_length=sequence_length,
        source_vocabulary_size=model_config.get("vocabulary_size", tokenizer.vocabulary_size),
        target_vocabulary_size=model_config.get("vocabulary_size", tokenizer.vocabulary_size),
        use_flash_attention=model_config.get("use_flash_attention", True),
    )

    transformer.to(device)

    # Optimizer / scheduler setup (matches train.py)
    optimizer = create_optimizer(transformer, learning_rate=args.learning_rate)

    accumulation_steps = 8
    total_batches = args.additional_epochs * len(train_dataloader)
    total_steps = (total_batches + accumulation_steps - 1) // accumulation_steps

    scheduler = create_scheduler(optimizer, total_steps, warmup_ratio=0.05)

    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path, device=device)

    if "model_state_dict" in ckpt:
        transformer.load_state_dict(ckpt["model_state_dict"])

    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("Warning: Could not fully load optimizer state. Optimizer will start fresh.")

    # Restore RNGs if present
    try:
        restore_rng_states(ckpt)
    except Exception:
        print("Warning: Could not restore RNG states from checkpoint.")

    start_epoch = int(ckpt.get("epoch", 0))
    additional = int(args.additional_epochs)

    # TensorBoard writer (append to existing run)
    writer = SummaryWriter(log_dir=run_folder)

    loss_function = create_loss_function(
        pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
        device=device,
        label_smoothing=0.1,
    )

    loss_scaler = torch.GradScaler("cuda") if device == "cuda" else None

    # Load existing loss histories
    epoch_loss_history, validation_loss_history = load_loss_histories(run_folder)
    
    # Calculate starting iteration based on checkpoint epoch
    # Each epoch has len(train_dataloader) batches, divided by accumulation_steps for optimizer steps
    batches_per_epoch = len(train_dataloader)
    steps_per_epoch = (batches_per_epoch + accumulation_steps - 1) // accumulation_steps
    iterations = start_epoch * steps_per_epoch

    # Resume training loop
    for extra_epoch in range(1, additional + 1):
        epoch = start_epoch + extra_epoch
        print(f"Resuming: Starting epoch {epoch}/{start_epoch + additional}")
        transformer.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Resumed Epoch {epoch}/{start_epoch + additional}")
        epoch_losses = []

        for i, batch in enumerate(batch_iterator):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            source_mask = batch["source_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            label = batch["label"].to(device)

            if loss_scaler is not None:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    projection = transformer(source, target, source_mask, target_mask)
                    loss = loss_function(projection.view(-1, tokenizer.vocabulary_size), label.view(-1))
                    loss = loss / accumulation_steps

                loss_scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    loss_scaler.unscale_(optimizer)
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Log gradient norm
                    writer.add_scalar("train/grad_norm", float(grad_norm_value), iterations)
                else:
                    grad_norm_value = 0.0
            else:
                projection = transformer(source, target, source_mask, target_mask)
                loss = loss_function(projection.view(-1, tokenizer.vocabulary_size), label.view(-1))
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Log gradient norm
                    writer.add_scalar("train/grad_norm", float(grad_norm_value), iterations)
                else:
                    grad_norm_value = 0.0

            loss_value = loss.item() * accumulation_steps
            ips = float(batch_iterator.format_dict.get("rate", 0.0) or 0.0)
            
            batch_iterator.set_postfix({"loss": f"{loss_value:.6f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            
            # Log to TensorBoard every 10 iterations
            log_training_iteration(writer, loss_value, iterations, scheduler, ips, log_every_n=10)
            
            epoch_losses.append(loss_value)
            iterations += 1

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # Validation
        transformer.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in tqdm(validation_dataloader, desc="Validation", leave=False):
                val_source = val_batch["source"].to(device)
                val_target = val_batch["target"].to(device)
                val_source_mask = val_batch["source_mask"].to(device)
                val_target_mask = val_batch["target_mask"].to(device)
                val_label = val_batch["label"].to(device)

                val_projection = transformer(val_source, val_target, val_source_mask, val_target_mask)
                val_loss = loss_function(val_projection.view(-1, tokenizer.vocabulary_size), val_label.view(-1))
                if torch.isfinite(val_loss):
                    val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        
        # Calculate BLEU and chrF scores
        bleu_score, chrf_score = score_sample_from_validation(
            transformer, validation_dataloader, tokenizer, device, num_samples=150
        )
        
        print(f"Epoch {epoch} completed. Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | BLEU: {bleu_score:.4f} | chrF: {chrf_score:.4f}")
        
        # Log to TensorBoard
        log_epoch_metrics(writer, epoch, avg_epoch_loss, avg_val_loss, bleu_score, chrf_score)
        
        # Append to loss histories
        epoch_loss_history.append({"epoch": epoch, "avg_loss": avg_epoch_loss})
        validation_loss_history.append({"epoch": epoch, "val_loss": avg_val_loss})
        
        # Save loss histories
        save_loss_histories(run_folder, epoch_loss_history, validation_loss_history)

        # Save checkpoint
        ckpt_save_path = Path(run_folder) / f"transformer_epoch_{epoch}.pt"
        torch.save({
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "iterations": iterations,
        }, ckpt_save_path)

        # Optionally update best
        best_path = Path(run_folder) / "transformer_best.pt"
        # If improvement, save best (simple heuristic: use val loss)
        try:
            prev_best = None
            if best_path.exists():
                prev = load_checkpoint(str(best_path), device=device)
                prev_best = prev.get("val_loss")
        except Exception:
            prev_best = None

        if prev_best is None or avg_val_loss < prev_best:
            torch.save({
                "model_state_dict": transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "iterations": iterations,
            }, best_path)
            print(f"Saved new best model to {best_path}")

    writer.close()
    print("Resumed training finished")


if __name__ == "__main__":
    main()
