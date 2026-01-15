"""
Example usage script for visualizing cross-attention patterns.

This script demonstrates how to use the visualize_cross_attention module
to generate and view/save attention heatmaps for all decoder blocks.
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Example 0: List available samples first
    print("=" * 80)
    print("Example 0: List first 10 sentence pairs")
    print("=" * 80)
    
    # Find the latest model folder
    model_base = Path("models/transformer")
    if not model_base.exists():
        print("No models found. Please train a model first.")
        return
    
    latest_model = max(model_base.iterdir(), key=lambda p: p.stat().st_mtime)
    print(f"Using model: {latest_model}\n")
    
    cmd = [
        sys.executable, "-m", "src.translator.visualize_cross_attention",
        "--model_folder", str(latest_model),
        "--list_samples", "10",
    ]
    
    subprocess.run(cmd)
    
    # Example 1: Final block heads - most informative (default)
    print("\n" + "=" * 80)
    print("Example 1: Final Block Heads (all heads for final decoder block)")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "src.translator.visualize_cross_attention",
        "--model_folder", str(latest_model),
        "--sample_idx", "0",
        "--mode", "final",
    ]
    
    subprocess.run(cmd)
    
    # Example 2: Overview mode - single plot showing all blocks averaged
    print("\n" + "=" * 80)
    print("Example 2: Overview mode (single plot, all blocks averaged)")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "src.translator.visualize_cross_attention",
        "--model_folder", str(latest_model),
        "--mode", "overview",
    ]
    
    subprocess.run(cmd)
    
    # Example 3: Comparison mode - side-by-side blocks
    print("\n" + "=" * 80)
    print("Example 3: Comparison mode (side-by-side blocks)")
    print("=" * 80)
    
    cmd = [
        sys.executable, "-m", "src.translator.visualize_cross_attention",
        "--model_folder", str(latest_model),
        "--mode", "comparison",
    ]
    
    subprocess.run(cmd)
    
    # Example 4: Detailed mode - per-head plots, save to PNG files
    print("\n" + "=" * 80)
    print("Example 4: Detailed mode (per-head plots, save to PNG files)")
    print("=" * 80)
    
    save_dir = Path(latest_model) / "attention_detailed"
    
    cmd = [
        sys.executable, "-m", "src.translator.visualize_cross_attention",
        "--model_folder", str(latest_model),
        "--sample_idx", "0",
        "--mode", "detailed",
        "--save_dir", str(save_dir),
    ]
    
    subprocess.run(cmd)
    print(f"\nDetailed plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
