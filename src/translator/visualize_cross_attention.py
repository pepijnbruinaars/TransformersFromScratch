"""
Visualize cross-attention patterns from all decoder blocks for a sample translation.

Usage:
    python -m src.translator.visualize_cross_attention --model_folder models/transformer/20260113_135051
    python -m src.translator.visualize_cross_attention --model_folder models/transformer/20260113_135051 --sample_idx 42
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ..constants import PAD_TOKEN, START_TOKEN, END_TOKEN
from ..tokenizer import CustomTokenizer
from ..transformer import Transformer
from ..attention import (
    plot_cross_attention_for_block,
    plot_attention_overview,
    plot_block_comparison,
    plot_final_block_heads,
    save_attention_figure,
    display_attention_figure,
)
from .dataset import CustomDataset
from .load_data import load_opus_data, get_sentences_from_data, get_max_sequence_length
from .train_utils import greedy_decode_single


def load_model_from_folder(model_folder: str, device: str = "cpu") -> tuple[Transformer, dict, CustomTokenizer]:
    """Load model, config, and tokenizer from a checkpoint folder.
    
    Args:
        model_folder: Path to folder containing model_config.json and transformer_best.pt
        device: Device to load model onto
        
    Returns:
        Tuple of (model, config_dict, tokenizer)
    """
    config_path = Path(model_folder) / "model_config.json"
    model_path = Path(model_folder) / "transformer_best.pt"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Initialize tokenizer
    tokenizer = CustomTokenizer("models/tokenizers/shared_tokenizer.json")
    
    # Create model
    model = Transformer(
        n_blocks=config["n_blocks"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        source_length=config["source_length"],
        target_length=config["target_length"],
        source_vocabulary_size=config["vocabulary_size"],
        target_vocabulary_size=config["vocabulary_size"],
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config, tokenizer


def get_device() -> str:
    """Returns the device to be used."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def greedy_decode_with_attention(
    transformer: Transformer,
    encoder_output: torch.Tensor,
    src_mask: torch.Tensor,
    start_id: int,
    end_id: int,
    pad_id: int,
    device: str,
    max_len: int = 64,
) -> tuple[list[int], dict, torch.Tensor]:
    """Greedy decode and capture cross-attention from all decoder blocks.
    
    Returns:
        Tuple of (generated_ids, attention_dict, decoder_output_seq)
        where attention_dict is {block_idx: cross_attention_tensor}
    """
    decoder_input = torch.tensor([[start_id]], dtype=torch.int64, device=device)
    generated_ids: list[int] = []
    all_cross_attentions: dict[int, list[torch.Tensor]] = {}
    
    for _ in range(max_len):
        seq_len = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        decoder_mask = ~causal_mask
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        decoder_output, dec_atts = transformer.decode(
            decoder_input, encoder_output, src_mask, decoder_mask, return_attentions=True
        )
        
        # Store cross-attention from all blocks
        if dec_atts is not None and "cross_attentions" in dec_atts:
            for block_idx, cross_attn in enumerate(dec_atts["cross_attentions"]):
                if block_idx not in all_cross_attentions:
                    all_cross_attentions[block_idx] = []
                all_cross_attentions[block_idx].append(cross_attn.detach().cpu())
        
        logits = transformer.project(decoder_output)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        
        if next_token == end_id:
            break
        
        generated_ids.append(next_token)
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], dtype=torch.int64, device=device)],
            dim=1,
        )
    
    return generated_ids, all_cross_attentions, decoder_output


def plot_cross_attention_for_block(
    cross_attn: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    block_idx: int,
) -> plt.Figure:
    """Create a figure with subplots for each attention head in a single decoder block.
    
    cross_attn: shape (batch=1, n_heads, tgt_len, src_len)
    """
    attn = cross_attn[0].detach().cpu()  # (n_heads, tgt_len, src_len)
    num_heads = attn.shape[0]
    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    
    # Trim to actual lengths
    attn_trimmed = attn[:, :tgt_len, :src_len]
    
    # Create subplots
    fig, axes = plt.subplots(nrows=num_heads, ncols=1, figsize=(max(10, src_len * 0.5), 2 * num_heads))
    if num_heads == 1:
        axes = [axes]
    
    # Helper to thin indices for readability
    def thin_indices(n: int, max_ticks: int = 24):
        if n <= max_ticks:
            return list(range(n))
        step = max(1, n // max_ticks)
        return list(range(0, n, step))
    
    x_idx = thin_indices(src_len)
    y_idx = thin_indices(tgt_len)
    
    im = None
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_trimmed[h].numpy(), aspect="auto", cmap="viridis")
        ax.set_title(f"Block {block_idx} - Head {h}")
        ax.set_xlabel("Source Position")
        ax.set_ylabel("Target Position")
        ax.set_xticks(x_idx)
        ax.set_xticklabels([src_tokens[i] for i in x_idx], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([tgt_tokens[i] for i in y_idx], fontsize=8)
    
    if im is not None:
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    
def list_samples(model_folder: str, num_samples: int = 10) -> None:
    """List available sentence pairs from the validation dataset.
    
    Args:
        model_folder: Path to checkpoint folder
        num_samples: Number of samples to list
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {model_folder}...")
    model, config, tokenizer = load_model_from_folder(model_folder, device=device)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = load_opus_data(1.0, 0.0, 0.0)
    _, english_sentences, dutch_sentences = get_sentences_from_data(full_dataset[0])
    
    train_raw, validation_raw, test_raw = load_opus_data(0.7, 0.15, 0.15)
    
    source_length, target_length = get_max_sequence_length(
        full_dataset[0], tokenizer, tokenizer
    )
    max_sequence_length = min(max(source_length, target_length), 512)
    
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        sequence_length=max_sequence_length,
    )
    
    print(f"\nAvailable validation samples (showing first {min(num_samples, len(validation))}):")
    print("=" * 100)
    
    for i in range(min(num_samples, len(validation))):
        sample = validation[i]
        src_text = sample['source_text']
        tgt_text = sample['target_text']
        
        # Truncate long sentences for display
        src_display = src_text[:80] + "..." if len(src_text) > 80 else src_text
        tgt_display = tgt_text[:80] + "..." if len(tgt_text) > 80 else tgt_text
        
        print("2d")
    
    print(f"\nTotal validation samples: {len(validation)}")
    print(f"Use --sample_idx <number> to visualize a specific sample")
    print(f"Example: python -m src.translator.visualize_cross_attention --model_folder {model_folder} --sample_idx 5")


def visualize_sample(
    model_folder: str,
    sample_idx: int | None = None,
    save_dir: str | None = None,
    mode: str = "final",
) -> None:
    """Load model and visualize cross-attention patterns for a sample translation.
    
    Args:
        model_folder: Path to checkpoint folder
        sample_idx: Index of sample to visualize (None = random)
        save_dir: If provided, save figures to this directory
        mode: Visualization mode - "final" (all heads for final block), "overview" (averaged grid), "comparison" (side-by-side blocks), "detailed" (per-head per-block)
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_folder}...")
    model, config, tokenizer = load_model_from_folder(model_folder, device=device)
    print(f"Model config: {config}")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = load_opus_data(1.0, 0.0, 0.0)
    _, english_sentences, dutch_sentences = get_sentences_from_data(full_dataset[0])
    
    train_raw, validation_raw, test_raw = load_opus_data(0.7, 0.15, 0.15)
    
    source_length, target_length = get_max_sequence_length(
        full_dataset[0], tokenizer, tokenizer
    )
    max_sequence_length = min(max(source_length, target_length), 512)
    
    validation = CustomDataset(
        validation_raw,  # type: ignore
        source_tokenizer=tokenizer,
        target_tokenizer=tokenizer,
        sequence_length=max_sequence_length,
    )
    
    # Select sample
    if sample_idx is None:
        sample_idx = random.randint(0, len(validation) - 1)
    
    if sample_idx >= len(validation):
        raise ValueError(f"Sample index {sample_idx} out of range [0, {len(validation)-1}]")
    
    sample = validation[sample_idx]
    print(f"\nUsing sample index: {sample_idx}")
    print(f"Source: {sample['source_text']}")
    print(f"Target: {sample['target_text']}")
    
    # Prepare input
    src = sample["source"].unsqueeze(0).to(device)
    src_mask = sample["source_mask"].to(device)
    
    # Encode
    start_id = tokenizer.token_to_id(START_TOKEN)
    end_id = tokenizer.token_to_id(END_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        
        # Decode with attention capture
        generated_ids, all_cross_attentions, _ = greedy_decode_with_attention(
            model, encoder_output, src_mask, start_id, end_id, pad_id, device
        )
    
    # Decode prediction
    pred_text = tokenizer.decode(generated_ids)
    print(f"Prediction: {pred_text}\n")
    
    # Get token sequences
    src_len = int((src != pad_id).sum().item())
    src_tokens = [tokenizer.id_to_token(int(t)) for t in src[:, :src_len].squeeze(0).tolist()]
    tgt_tokens = [tokenizer.id_to_token(start_id)] + [tokenizer.id_to_token(t) for t in generated_ids]
    
    print(f"Source tokens ({len(src_tokens)}): {src_tokens}")
    print(f"Target tokens ({len(tgt_tokens)}): {tgt_tokens}\n")
    
    # Create and display/save figures
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    num_blocks = len(all_cross_attentions)
    print(f"Generating attention visualizations for {num_blocks} decoder blocks (mode: {mode})...\n")
    
    # Collect final attention matrices for all blocks
    final_attentions = []
    for block_idx in sorted(all_cross_attentions.keys()):
        final_cross_attn = all_cross_attentions[block_idx][-1]  # Last decoding step
        final_attentions.append(final_cross_attn)
    
    if mode == "overview":
        # Single overview plot showing all blocks averaged
        fig = plot_attention_overview(final_attentions, src_tokens, tgt_tokens)
        
        if save_dir:
            fig_path = save_path / "attention_overview.png"
            save_attention_figure(fig, str(fig_path))
        else:
            display_attention_figure(fig)
        
    elif mode == "comparison":
        # Side-by-side comparison of all blocks
        fig = plot_block_comparison(final_attentions, src_tokens, tgt_tokens)
        
        if save_dir:
            fig_path = save_path / "block_comparison.png"
            save_attention_figure(fig, str(fig_path))
        else:
            display_attention_figure(fig)
        
    elif mode == "final":
        # All heads for final decoder block only
        fig = plot_final_block_heads(final_attentions, src_tokens, tgt_tokens)
        
        if save_dir:
            fig_path = save_path / "final_block_heads.png"
            save_attention_figure(fig, str(fig_path))
        else:
            display_attention_figure(fig)
        
    elif mode == "detailed":
        # Original detailed view: per-head plots for each block
        for block_idx, final_cross_attn in enumerate(final_attentions):
            fig = plot_cross_attention_for_block(
                final_cross_attn,
                src_tokens,
                tgt_tokens,
                block_idx,
            )
            
            if save_dir:
                fig_path = save_path / f"block_{block_idx:02d}_heads.png"
                save_attention_figure(fig, str(fig_path))
            else:
                display_attention_figure(fig)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'overview', 'comparison', 'final', or 'detailed'")
    
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-attention patterns from all decoder blocks"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to checkpoint folder containing model_config.json and transformer_best.pt",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Index of sample to visualize (default: random)",
    )
    parser.add_argument(
        "--list_samples",
        type=int,
        default=None,
        help="List first N sentence pairs and exit (useful for finding good examples)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save figures (if not provided, displays with plt.show())",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="final",
        choices=["overview", "comparison", "final", "detailed"],
        help="Visualization mode: 'overview' (grid of averaged blocks), 'comparison' (side-by-side blocks), 'final' (all heads for final block), 'detailed' (per-head plots per block)",
    )
    
    args = parser.parse_args()
    
    # Handle list_samples mode
    if args.list_samples is not None:
        list_samples(args.model_folder, args.list_samples)
        return
    
    visualize_sample(
        model_folder=args.model_folder,
        sample_idx=args.sample_idx,
        save_dir=args.save_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
