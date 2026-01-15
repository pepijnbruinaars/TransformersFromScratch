"""
Attention visualization utilities - empty init file for package.
"""

from .visualization import (
    plot_cross_attention_for_block,
    plot_attention_matrix,
    plot_attention_overview,
    plot_block_comparison,
    plot_final_block_heads,
    save_attention_figure,
    display_attention_figure,
)

from .utils import (
    extract_cross_attentions,
    extract_self_attentions,
    aggregate_attention_over_sequence,
    average_attention_heads,
    get_attention_at_position,
    find_max_attention_positions,
)

__all__ = [
    # Visualization
    "plot_cross_attention_for_block",
    "plot_attention_matrix",
    "plot_attention_overview",
    "plot_block_comparison",
    "plot_final_block_heads",
    "save_attention_figure",
    "display_attention_figure",

    # Processing
    "extract_cross_attentions",
    "extract_self_attentions",
    "aggregate_attention_over_sequence",
    "average_attention_heads",
    "get_attention_at_position",
    "find_max_attention_positions",
]
