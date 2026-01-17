import os
import glob
import json
import random
import math
from typing import Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore
from sacrebleu.metrics import CHRF  # type: ignore
from ..constants import START_TOKEN, END_TOKEN, PAD_TOKEN
from ..tokenizer import CustomTokenizer
from .dataset import CustomDataset
from .load_data import load_opus_data


def get_device() -> str:
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_splits(run_folder: str, train_raw: Any, validation_raw: Any, test_raw: Any) -> None:
    ensure_dir(run_folder)
    path = os.path.join(run_folder, "splits.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train_raw, "validation": validation_raw, "test": test_raw}, f, ensure_ascii=False)


def load_splits(run_folder: str):
    path = os.path.join(run_folder, "splits.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train"], data["validation"], data["test"]


def save_checkpoint(path: str, transformer: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int, extra: dict | None = None) -> None:
    ckpt = {
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str | None = None) -> dict:
    map_loc = None if device is None else device
    return torch.load(path, map_location=map_loc)


def restore_rng_states(ckpt: dict) -> None:
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])


def find_latest_run(base_dir: str = "models/transformer") -> str | None:
    if not os.path.isdir(base_dir):
        return None
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def find_latest_checkpoint(run_folder: str) -> str | None:
    patterns = [os.path.join(run_folder, "transformer_epoch_*.pt"), os.path.join(run_folder, "transformer_best.pt"), os.path.join(run_folder, "transformer_final.pt")]
    epoch_files = glob.glob(patterns[0])
    if epoch_files:
        def epoch_num(p: str) -> int:
            name = os.path.basename(p)
            try:
                return int(name.split("_")[-1].split(".")[0])
            except Exception:
                return -1

        return max(epoch_files, key=epoch_num)
    for p in patterns[1:]:
        files = glob.glob(p)
        if files:
            return files[0]
    return None


def build_dataloaders(
    tokenizer: CustomTokenizer,
    sequence_length: int,
    batch_size: int = 8,
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders.
    
    Args:
        tokenizer: The tokenizer to use
        sequence_length: Maximum sequence length
        batch_size: Batch size for dataloaders
        splits: (train_ratio, val_ratio, test_ratio) for data splitting
        
    Returns:
        Tuple of (train_dataloader, validation_dataloader)
    """
    train_raw, validation_raw, test_raw = load_opus_data(splits[0], splits[1], splits[2])

    train = CustomDataset(
        train_raw, source_tokenizer=tokenizer, target_tokenizer=tokenizer, sequence_length=sequence_length  # type: ignore
    )
    validation = CustomDataset(
        validation_raw, source_tokenizer=tokenizer, target_tokenizer=tokenizer, sequence_length=sequence_length  # type: ignore
    )

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_dataloader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_dataloader, validation_dataloader


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 5e-5,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """Create Adam optimizer with transformer-specific hyperparameters.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        betas: Adam beta parameters
        eps: Adam epsilon
        weight_decay: Weight decay coefficient
        
    Returns:
        Configured Adam optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)


def learning_rate_lambda(current_step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning rate schedule with warmup and cosine annealing.
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate multiplier
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with warmup and cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of training steps
        warmup_ratio: Fraction of steps to use for warmup
        
    Returns:
        Configured LambdaLR scheduler
    """
    warmup_steps = int(warmup_ratio * total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_lambda(step, total_steps, warmup_steps),
    )


def create_loss_function(
    pad_token_id: int,
    device: str,
    label_smoothing: float = 0.1,
) -> torch.nn.CrossEntropyLoss:
    """Create cross-entropy loss function with label smoothing.
    
    Args:
        pad_token_id: Token ID for padding (to ignore in loss)
        device: Device to place loss function on
        label_smoothing: Label smoothing factor
        
    Returns:
        Configured CrossEntropyLoss
    """
    return torch.nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=label_smoothing,
    ).to(device)


def log_training_iteration(
    writer: SummaryWriter,
    loss_value: float,
    iterations: int,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    iters_per_sec: float,
    log_every_n: int = 10,
) -> None:
    """Log training metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        loss_value: Current loss value
        iterations: Current iteration number
        scheduler: Learning rate scheduler
        iters_per_sec: Iterations per second
        log_every_n: Log every N iterations
    """
    if iterations % log_every_n == 0:
        iter_time_ms = 1000.0 / iters_per_sec if iters_per_sec > 0 else 0.0
        writer.add_scalar("train/loss", loss_value, iterations)
        writer.add_scalar("speed/iters_per_sec", iters_per_sec, iterations)
        writer.add_scalar("speed/iter_time_ms", iter_time_ms, iterations)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], iterations)
        writer.add_scalar("train/perplexity", math.exp(loss_value), iterations)
        writer.flush()


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    bleu_score: float,
    chrf_score: float,
) -> None:
    """Log epoch-level metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        avg_train_loss: Average training loss
        avg_val_loss: Average validation loss
        bleu_score: BLEU score
        chrf_score: chrF score
    """
    writer.add_scalars("loss/epoch", {"train": avg_train_loss, "validation": avg_val_loss}, epoch)
    writer.add_scalar("validation/bleu_score", bleu_score, epoch)
    writer.add_scalar("validation/chrf_score", chrf_score, epoch)
    writer.flush()


def load_loss_histories(run_folder: str) -> tuple[list[dict], list[dict]]:
    """Load existing loss history JSON files.
    
    Args:
        run_folder: Path to the run folder
        
    Returns:
        Tuple of (epoch_loss_history, validation_loss_history)
    """
    epoch_loss_history = []
    validation_loss_history = []
    
    epoch_loss_path = Path(run_folder) / "epoch_loss_history.json"
    val_loss_path = Path(run_folder) / "validation_loss_history.json"
    
    if epoch_loss_path.exists():
        with open(epoch_loss_path, "r") as f:
            epoch_loss_history = json.load(f)
    
    if val_loss_path.exists():
        with open(val_loss_path, "r") as f:
            validation_loss_history = json.load(f)
    
    return epoch_loss_history, validation_loss_history


def save_loss_histories(
    run_folder: str,
    epoch_loss_history: list[dict],
    validation_loss_history: list[dict],
) -> None:
    """Save loss history JSON files.
    
    Args:
        run_folder: Path to the run folder
        epoch_loss_history: List of epoch loss records
        validation_loss_history: List of validation loss records
    """
    epoch_loss_path = Path(run_folder) / "epoch_loss_history.json"
    val_loss_path = Path(run_folder) / "validation_loss_history.json"
    
    with open(epoch_loss_path, "w") as f:
        json.dump(epoch_loss_history, f, indent=2)
    with open(val_loss_path, "w") as f:
        json.dump(validation_loss_history, f, indent=2)


def greedy_decode_with_rollout(
    transformer: torch.nn.Module,
    encoder_output: torch.Tensor,
    source_mask: torch.Tensor,
    start_id: int,
    end_id: int,
    pad_id: int,
    device: str,
    max_len: int = 64,
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy decode while capturing all decoder cross-attention layers at final step.
    
    In autoregressive decoding, we generate one token at a time. To compute attention rollout,
    we only use the final forward pass where all tokens have been decoded.
    
    Args:
        transformer: The model to decode with
        encoder_output: (1, src_len, d_model)
        source_mask: (1, 1, src_len) or (1, src_len)
        start_id: START token ID
        end_id: END token ID
        pad_id: PAD token ID
        device: Device to use
        max_len: Maximum sequence length
        
    Returns:
        Tuple of (generated_ids, final_layer_cross_attentions)
        where final_layer_cross_attentions is a list of attention tensors from each layer
        at the final decoding step: each (1, n_heads, final_seq_len, src_len)
    """
    decoder_input = torch.tensor([[start_id]], dtype=torch.int64, device=device)
    generated_ids: list[int] = []
    final_attentions: list[torch.Tensor] = []
    
    for step in range(max_len):
        seq_len = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        decoder_mask = ~causal_mask
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        decoder_output, attn_dict = transformer.decode(
            decoder_input, encoder_output, source_mask, decoder_mask, return_attentions=True
        )
        
        logits = transformer.project(decoder_output)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        
        # Always capture final attention at the last step (when we output a token or hit end)
        cross_attentions = attn_dict.get("cross_attentions", [])
        
        if next_token == end_id:
            # This is the final step - save these attentions for rollout
            final_attentions = [attn for attn in cross_attentions]
            break
        
        generated_ids.append(next_token)
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], dtype=torch.int64, device=device)],
            dim=1,
        )
        
        # On last iteration, also save attention if we hit max_len
        if step == max_len - 1:
            final_attentions = [attn for attn in cross_attentions]
    
    return generated_ids, final_attentions


def greedy_decode_single(
    transformer: torch.nn.Module,
    encoder_output: torch.Tensor,
    source_mask: torch.Tensor,
    start_id: int,
    end_id: int,
    pad_id: int,
    device: str,
    max_len: int = 64,
) -> list[int]:
    """Greedy decode a single sample. Safe, simple, and reliable.
    
    Args:
        encoder_output: (1, src_len, d_model)
        source_mask: (1, 1, src_len) or (1, src_len)
        
    Returns:
        List of decoded token IDs
    """
    decoder_input = torch.tensor([[start_id]], dtype=torch.int64, device=device)
    generated_ids: list[int] = []
    
    for _ in range(max_len):
        seq_len = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        decoder_mask = ~causal_mask
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        decoder_output = transformer.decode(
            decoder_input, encoder_output, source_mask, decoder_mask
        )
        
        logits = transformer.project(decoder_output)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        
        if next_token == end_id:
            break
        
        generated_ids.append(next_token)
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], dtype=torch.int64, device=device)],
            dim=1,
        )
    
    return generated_ids


def calculate_bleu_chrf(
    generated_texts: list[str],
    reference_texts: list[str],
) -> tuple[float, float]:
    """Calculate BLEU and chrF scores.
    
    Args:
        generated_texts: List of generated translations
        reference_texts: List of reference translations
        
    Returns:
        Tuple of (bleu_score, chrf_score)
    """
    bleu_score = 0.0
    chrf_score = 0.0
    
    if not generated_texts or not reference_texts:
        return bleu_score, chrf_score
    
    try:
        # BLEU score
        references = [[ref.split()] for ref in reference_texts]
        hypotheses = [hyp.split() for hyp in generated_texts]
        smoothing_function = SmoothingFunction().method1
        bleu_score = corpus_bleu(
            references, hypotheses, smoothing_function=smoothing_function
        )
    except Exception as e:
        print(f"Warning: Could not calculate BLEU score: {e}")
    
    try:
        # chrF score
        chrf_metric = CHRF()
        chrf_result = chrf_metric.corpus_score(generated_texts, [reference_texts])
        chrf_score = chrf_result.score / 100.0
    except Exception as e:
        print(f"Warning: Could not calculate chrF score: {e}")
    
    return bleu_score, chrf_score  # type: ignore


def compute_attention_rollout(
    attention_matrices: list[torch.Tensor],
    normalize: bool = True,
) -> torch.Tensor:
    """Compute attention rollout by recursively multiplying attention matrices across layers.
    
    This method traces how information flows through the entire decoder network.
    
    Args:
        attention_matrices: List of attention tensors from each layer, shape (batch, n_heads, seq_len, seq_len)
        normalize: Whether to normalize attention matrices before multiplication
        
    Returns:
        Rollout attention matrix of shape (batch, seq_len, seq_len)
    """
    if not attention_matrices:
        return torch.tensor([])
    
    batch_size = attention_matrices[0].shape[0]
    seq_len = attention_matrices[0].shape[2]
    device = attention_matrices[0].device
    
    # Start with the last layer (L), average across heads
    # attention_matrices[-1] shape: (batch, n_heads, seq_len, seq_len)
    rollout = attention_matrices[-1].mean(dim=1)  # (batch, seq_len, seq_len)
    
    if normalize:
        rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-9)
    
    # Recursively multiply with previous layers (going backwards through the network)
    for i in range(len(attention_matrices) - 2, -1, -1):
        attn = attention_matrices[i].mean(dim=1)  # (batch, seq_len, seq_len)
        
        if normalize:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Matrix multiplication: (batch, seq_len, seq_len) @ (batch, seq_len, seq_len)
        rollout = torch.matmul(attn, rollout)
        
        if normalize:
            rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-9)
    
    return rollout


def attention_rollout_to_image(
    rollout: torch.Tensor,
    src_len: int,
    tgt_len: int,
) -> torch.Tensor:
    """Convert attention rollout matrix to an image tensor for TensorBoard.
    
    Args:
        rollout: Attention rollout of shape (batch, seq_len, seq_len)
        src_len: Source sequence length (for trimming)
        tgt_len: Target sequence length (for trimming)
        
    Returns:
        Image tensor of shape (3, tgt_len, src_len) for TensorBoard visualization
    """
    # Take first batch item and trim to actual lengths
    attn = rollout[0, :tgt_len, :src_len].cpu().detach()
    
    # Normalize to [0, 1]
    attn_min = attn.min()
    attn_max = attn.max()
    if attn_max > attn_min:
        attn = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn = torch.ones_like(attn)
    
    # Convert to 3-channel image (RGB) using viridis colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    
    attn_np = attn.numpy()
    attn_colored = cmap(attn_np)  # (tgt_len, src_len, 4)
    
    # Convert to (3, tgt_len, src_len) format for TensorBoard
    img = torch.from_numpy(attn_colored[:, :, :3]).permute(2, 0, 1).float()
    
    return img


def count_unique_unigrams(text: str) -> int:
    """Count unique unigrams (words) in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of unique unigrams
    """
    tokens = text.split()
    return len(set(tokens))


def count_unique_bigrams(text: str) -> int:
    """Count unique bigrams in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of unique bigrams
    """
    tokens = text.split()
    if len(tokens) < 2:
        return 0
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    return len(set(bigrams))


def log_weight_histogram(
    model: torch.nn.Module,
    writer: SummaryWriter,
    global_step: int,
) -> None:
    """Log weight histograms for all parameters in the model.
    
    Args:
        model: The model to log weights from
        writer: TensorBoard SummaryWriter
        global_step: Current training step
    """
    for name, param in model.named_parameters():
        if param.data is not None:
            writer.add_histogram(f"weights/{name}", param.data, global_step)


def score_sample_from_validation(
    transformer: torch.nn.Module,
    validation_dataloader,
    tokenizer,
    device: str,
    num_samples: int = 100,
) -> tuple[float, float, list[float], list[float], list[int], list[int]]:
    """Sample ~num_samples random items from validation set and compute BLEU/chrF.
    
    This is much faster than full validation pass and gives good estimates.
    Also tracks generated and target sequence lengths, plus n-gram diversity.
    
    Args:
        transformer: The model to score
        validation_dataloader: Validation DataLoader
        tokenizer: Tokenizer for encoding/decoding
        device: Device to use (cuda/cpu/mps)
        num_samples: Number of random samples to score (default 100)
        
    Returns:
        Tuple of (bleu_score, chrf_score, generated_lengths, target_lengths, unigram_counts, bigram_counts)
    """
    transformer.eval()
    
    # Collect all data from dataloader (needed for random sampling)
    all_sources: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_source_masks: list[torch.Tensor] = []
    
    with torch.no_grad():
        for batch in validation_dataloader:
            all_sources.append(batch["source"])
            all_targets.append(batch["target"])
            all_source_masks.append(batch["source_mask"])
    
    # Concatenate all batches
    sources: torch.Tensor = torch.cat(all_sources, dim=0)
    targets: torch.Tensor = torch.cat(all_targets, dim=0)
    source_masks: torch.Tensor = torch.cat(all_source_masks, dim=0)
    
    total_samples = sources.size(0)
    sample_size = min(num_samples, total_samples)
    
    # Random indices
    indices = torch.randperm(total_samples)[:sample_size]
    
    generated_texts: list[str] = []
    reference_texts: list[str] = []
    generated_lengths: list[float] = []
    target_lengths: list[float] = []
    unigram_counts: list[int] = []
    bigram_counts: list[int] = []
    
    start_id = tokenizer.token_to_id(START_TOKEN)
    end_id = tokenizer.token_to_id(END_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    special_ids = {start_id, end_id, pad_id}
    
    with torch.no_grad():
        for idx in indices:
            source = sources[idx].unsqueeze(0).to(device)
            target = targets[idx]
            source_mask = source_masks[idx].unsqueeze(0).to(device)
            
            # Encode
            encoder_result = transformer.encode(source, source_mask, return_attentions=False)
            if isinstance(encoder_result, tuple):
                encoder_output = encoder_result[0]
            else:
                encoder_output = encoder_result
            
            # Decode single sample
            generated_ids = greedy_decode_single(
                transformer,
                encoder_output,
                source_mask,
                start_id,
                end_id,
                pad_id,
                device,
                max_len=source.size(1),
            )
            
            generated_text = tokenizer.decode(generated_ids)
            clean_reference_ids = target[~target.unsqueeze(1).eq(torch.tensor(list(special_ids), device=target.device)).any(1)].tolist()
            reference_text = tokenizer.decode(clean_reference_ids)
            
            generated_texts.append(generated_text)
            reference_texts.append(reference_text)
            
            # Track sequence lengths
            generated_lengths.append(float(len(generated_ids)))
            target_lengths.append(float(len(clean_reference_ids)))
            
            # Track n-gram diversity
            unigram_counts.append(count_unique_unigrams(generated_text))
            bigram_counts.append(count_unique_bigrams(generated_text))
    
    bleu_score, chrf_score = calculate_bleu_chrf(generated_texts, reference_texts)
    return bleu_score, chrf_score, generated_lengths, target_lengths, unigram_counts, bigram_counts
