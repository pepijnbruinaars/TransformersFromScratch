from datetime import datetime
import math
from pathlib import Path
import json

from tqdm import tqdm  # type: ignore
from ..constants import PAD_TOKEN
from .load_data import get_max_sequence_length, load_opus_data, get_sentences_from_data
from ..tokenizer import CustomTokenizer
from ..transformer import Transformer
from .dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import logging

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


def train_or_load_tokenizers(
    dutch_sentences: list[str], english_sentences: list[str]
) -> tuple[CustomTokenizer, CustomTokenizer]:
    dutch_tokenizer = CustomTokenizer("models/tokenizers/dutch_tokenizer.json")
    english_tokenizer = CustomTokenizer("models/tokenizers/english_tokenizer.json")

    if not dutch_tokenizer.trained:
        logger.info("Training Dutch tokenizer...")
        dutch_tokenizer.train(dutch_sentences)
        dutch_tokenizer.save("models/tokenizers/dutch_tokenizer.json")
    if not english_tokenizer.trained:
        logger.info("Training English tokenizer...")
        english_tokenizer.train(english_sentences)
        english_tokenizer.save("models/tokenizers/english_tokenizer.json")

    logger.info("Testing Dutch tokenizer...")
    sample_dutch_sentences = dutch_sentences[:20]
    for sentence in sample_dutch_sentences:
        dutch_tokenizer.print_tokens(sentence)

    logger.info("Testing English tokenizer...")
    sample_english_sentences = english_sentences[:20]
    for sentence in sample_english_sentences:
        english_tokenizer.print_tokens(sentence)

    return dutch_tokenizer, english_tokenizer


def learning_rate_lambda(current_step: int, total_steps: int, warmup_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(
    transformer: Transformer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    model_config: dict,
    nr_epochs: int = 20,
    learning_rate: float = 1e-4,
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

    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    total_steps = nr_epochs * len(train_dataloader)
    warmup_steps = int(0.05 * total_steps)
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warm-up steps: {warmup_steps}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_lambda(step, total_steps, warmup_steps),
    )

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.token_to_id(PAD_TOKEN),
        label_smoothing=0.1,
    ).to(device)

    iterations = 0
    loss_history = []
    epoch_loss_history = []
    validation_loss_history = []
    best_val_loss = float('inf')

    log_frequency = 10
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

            loss_scaler = torch.GradScaler("cuda")
            with torch.autocast(device_type=device, dtype=torch.float16):
                projection = transformer(
                    source, target, source_mask, target_mask
                )
                loss = loss_function(
                    projection.view(-1, target_tokenizer.vocabulary_size), label.view(-1)
                )

            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)

            loss_value = loss.item()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)

            loss_scaler.step(optimizer)
            loss_scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            ips = float(batch_iterator.format_dict.get("rate", 0.0) or 0.0)
            iter_time_ms = 1000.0 / ips if ips > 0 else 0.0

            batch_iterator.set_postfix({"loss": f"{loss_value:.6f}", "ips": f"{ips:.2f}"})
            writer.add_scalar("Loss/train", loss_value, iterations)
            writer.add_scalar("Speed/iters_per_sec", ips, iterations)
            writer.add_scalar("Speed/iter_time_ms", iter_time_ms, iterations)
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
                
                with torch.autocast(device_type=device, dtype=torch.float16):
                    val_projection = transformer(
                        val_source, val_target, val_source_mask, val_target_mask
                    )
                    val_loss = loss_function(
                        val_projection.view(-1, target_tokenizer.vocabulary_size), val_label.view(-1)
                    )
                
                if torch.isfinite(val_loss):
                    val_losses.append(val_loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        validation_loss_history.append({"epoch": epoch + 1, "val_loss": avg_val_loss})
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalars("Loss/epoch", {"train": avg_epoch_loss, "validation": avg_val_loss}, epoch + 1)
        writer.flush()
        
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

    dutch_tokenizer, english_tokenizer = train_or_load_tokenizers(
        dutch_sentences, english_sentences
    )

    train_raw, validation_raw, test_raw = load_opus_data(0.7, 0.15, 0.15)

    source_length, target_length = get_max_sequence_length(
        full_dataset[0], english_tokenizer, dutch_tokenizer
    )

    max_sequence_length = min(
        max(source_length, target_length), 512
    )

    train = CustomDataset(
        train_raw,  # type: ignore
        source_tokenizer=english_tokenizer,
        target_tokenizer=dutch_tokenizer,
        sequence_length=max_sequence_length,
    )
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=english_tokenizer,
        target_tokenizer=dutch_tokenizer,
        sequence_length=max_sequence_length,
    )

    train_dataloader = DataLoader(train, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=8, shuffle=False)

    n_blocks = 8
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
        "source_vocabulary_size": english_tokenizer.vocabulary_size,
        "target_vocabulary_size": dutch_tokenizer.vocabulary_size,
    }
    
    transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        dropout=dropout,
        source_length=max_sequence_length,
        target_length=max_sequence_length,
        source_vocabulary_size=english_tokenizer.vocabulary_size,
        target_vocabulary_size=dutch_tokenizer.vocabulary_size,
    )
    
    logger.info(transformer)

    train_model(
        transformer=transformer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        source_tokenizer=english_tokenizer,
        target_tokenizer=dutch_tokenizer,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
