"""
General attention visualization utilities.

These functions are model-agnostic and can be used to visualize
attention patterns from any transformer-like model.
"""

import matplotlib.pyplot as plt
import torch


def plot_cross_attention_for_block(
    cross_attn: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    block_idx: int,
    title_prefix: str = "Block",
) -> plt.Figure:
    """Create a figure with subplots for each attention head in a single decoder block.

    Args:
        cross_attn: shape (batch=1, n_heads, tgt_len, src_len)
        src_tokens: List of source token strings
        tgt_tokens: List of target token strings
        block_idx: Index of the decoder block
        title_prefix: Prefix for subplot titles (e.g., "Block", "Layer")

    Returns:
        Matplotlib figure with attention heatmaps
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
        ax.set_title(f"{title_prefix} {block_idx} - Head {h}")
        ax.set_xlabel("Source Position")
        ax.set_ylabel("Target Position")
        ax.set_xticks(x_idx)
        ax.set_xticklabels([src_tokens[i] for i in x_idx], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([tgt_tokens[i] for i in y_idx], fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)

    fig.tight_layout()
    return fig


def plot_attention_matrix(
    attn_matrix: torch.Tensor,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "Attention",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot a single attention matrix with custom labels.

    Args:
        attn_matrix: shape (y_len, x_len) or (n_heads, y_len, x_len)
        x_labels: Labels for x-axis (columns)
        y_labels: Labels for y-axis (rows)
        title: Plot title
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure
    """
    if attn_matrix.dim() == 3:
        # Multiple heads - take mean across heads for summary view
        attn_matrix = attn_matrix.mean(dim=0)

    attn = attn_matrix.detach().cpu().numpy()

    if figsize is None:
        figsize = (max(8, len(x_labels) * 0.3), max(6, len(y_labels) * 0.2))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")

    # Thin indices for readability
    def thin_indices(n: int, max_ticks: int = 20):
        if n <= max_ticks:
            return list(range(n))
        step = max(1, n // max_ticks)
        return list(range(0, n, step))

    x_idx = thin_indices(len(x_labels))
    y_idx = thin_indices(len(y_labels))

    ax.set_xticks(x_idx)
    ax.set_xticklabels([x_labels[i] for i in x_idx], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(y_idx)
    ax.set_yticklabels([y_labels[i] for i in y_idx], fontsize=8)

    fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    fig.tight_layout()

    return fig


def plot_attention_overview(
    attention_blocks: list[torch.Tensor],
    src_tokens: list[str],
    tgt_tokens: list[str],
    title_prefix: str = "Block",
) -> plt.Figure:
    """Create an overview plot showing averaged attention for all blocks in a grid.

    Args:
        attention_blocks: List of attention tensors, one per block
        src_tokens: Source token strings
        tgt_tokens: Target token strings
        title_prefix: Prefix for subplot titles

    Returns:
        Matplotlib figure with grid of attention matrices
    """
    n_blocks = len(attention_blocks)
    if n_blocks == 0:
        raise ValueError("No attention blocks provided")

    # Calculate grid dimensions
    cols = min(3, n_blocks)
    rows = (n_blocks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, attn_tensor in enumerate(attention_blocks):
        if i >= len(axes):
            break

        ax = axes[i]

        # Average across heads: (1, n_heads, tgt_len, src_len) -> (tgt_len, src_len)
        avg_attn = attn_tensor.mean(dim=1)[0].detach().cpu().numpy()

        # Trim to actual lengths
        src_len = len(src_tokens)
        tgt_len = len(tgt_tokens)
        avg_attn_trimmed = avg_attn[:tgt_len, :src_len]

        im = ax.imshow(avg_attn_trimmed, aspect="auto", cmap="viridis")
        ax.set_title(f"{title_prefix} {i}")

        # Simplified axis labels for overview
        if i % cols == 0:  # Leftmost column
            ax.set_ylabel("Target")
        if i >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel("Source")

    # Hide unused subplots
    for i in range(n_blocks, len(axes)):
        axes[i].set_visible(False)

    # Add colorbar to the right of the last plot
    if n_blocks > 0:
        fig.colorbar(im, ax=axes[:n_blocks], orientation="vertical", fraction=0.02, pad=0.04)

    fig.suptitle("Attention Overview (Averaged Across Heads)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_block_comparison(
    attention_blocks: list[torch.Tensor],
    src_tokens: list[str],
    tgt_tokens: list[str],
    block_indices: list[int] | None = None,
    title: str = "Block Comparison",
) -> plt.Figure:
    """Compare attention patterns across specific blocks side by side.

    Args:
        attention_blocks: List of attention tensors, one per block
        src_tokens: Source token strings
        tgt_tokens: Target token strings
        block_indices: Which blocks to compare (None = all)
        title: Plot title

    Returns:
        Matplotlib figure comparing blocks
    """
    if block_indices is None:
        block_indices = list(range(len(attention_blocks)))

    n_blocks = len(block_indices)
    if n_blocks == 0:
        raise ValueError("No blocks to compare")

    fig, axes = plt.subplots(1, n_blocks, figsize=(6 * n_blocks, 5))

    if n_blocks == 1:
        axes = [axes]

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)

    for i, block_idx in enumerate(block_indices):
        ax = axes[i]
        attn_tensor = attention_blocks[block_idx]

        # Average across heads
        avg_attn = attn_tensor.mean(dim=1)[0].detach().cpu().numpy()
        avg_attn_trimmed = avg_attn[:tgt_len, :src_len]

        im = ax.imshow(avg_attn_trimmed, aspect="auto", cmap="viridis")
        ax.set_title(f"Block {block_idx}")
        ax.set_xlabel("Source")
        ax.set_ylabel("Target")

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_final_block_heads(
    attention_blocks: list[torch.Tensor],
    src_tokens: list[str],
    tgt_tokens: list[str],
    title: str = "Final Block Attention Heads",
) -> plt.Figure:
    """Plot all attention heads for the final decoder block.

    Args:
        attention_blocks: List of attention tensors, one per block
        src_tokens: Source token strings
        tgt_tokens: Target token strings
        title: Plot title

    Returns:
        Matplotlib figure with all heads from final block
    """
    if not attention_blocks:
        raise ValueError("No attention blocks provided")

    # Use the final (last) block
    final_attn = attention_blocks[-1]  # (1, n_heads, tgt_len, src_len)

    return plot_cross_attention_for_block(
        final_attn,
        src_tokens,
        tgt_tokens,
        block_idx=len(attention_blocks) - 1,
        title_prefix="Final Block"
    )


def save_attention_figure(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 100,
    bbox_inches: str = "tight",
) -> None:
    """Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure
        filepath: Path to save figure
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved: {filepath}")


def display_attention_figure(fig: plt.Figure) -> None:
    """Display a matplotlib figure (blocks until closed).

    Args:
        fig: Matplotlib figure to display
    """
    plt.show()
