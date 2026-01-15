# Cross-Attention Visualization Guide

This tool visualizes the cross-attention patterns from all decoder blocks in the Transformer model, similar to what's rendered in the TensorBoard dashboard but with more granular control.

## What It Does

For a given sentence pair from the validation dataset:
1. **Encodes** the source (English) sentence
2. **Decodes** autoregressively with greedy decoding, capturing cross-attention weights at each decoder block
3. **Generates attention heatmaps** for all decoder blocks and all attention heads
4. **Displays or saves** the visualizations as PNG files

Each figure shows all attention heads for a single decoder block:
- **X-axis**: Source tokens (English)
- **Y-axis**: Target tokens (Dutch) 
- **Color intensity**: Attention weight (darker = higher attention)

## Usage

### Final Block Heads (Recommended - shows all heads for final layer)
Single plot showing all attention heads for the final decoder block:

```bash
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --mode final
```

### Overview Mode
Single plot showing all decoder blocks in a grid, with attention averaged across heads:

```bash
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --mode overview
```

### Comparison Mode
Side-by-side comparison of all decoder blocks:

```bash
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --mode comparison
```

### List available samples first

```bash
# List first 20 sentence pairs to find good examples
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --list_samples 20
```

### Visualize specific sample

```bash
# Visualize sample #5 (found from list above)
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --sample_idx 5
```

### Detailed Mode (Original behavior)
Per-head attention plots for each decoder block:

```bash
python -m src.translator.visualize_cross_attention \
  --model_folder models/transformer/20260113_135051 \
  --mode detailed \
  --sample_idx 42 \
  --save_dir output/attention_visualizations
```

### Via Python script

```python
from src.translator.visualize_cross_attention import visualize_sample

# Final block heads - most informative (default)
visualize_sample(
    model_folder="models/transformer/20260113_135051",
    sample_idx=5,
    mode="final"
)

# Overview mode - single plot with all blocks
visualize_sample(
    model_folder="models/transformer/20260113_135051",
    sample_idx=5,
    mode="overview"
)

# Detailed mode - per-head plots
visualize_sample(
    model_folder="models/transformer/20260113_135051",
    sample_idx=5,
    mode="detailed",
    save_dir="output/attention_visualizations"
)
```

## Output

### Console Output
```
Using device: cuda
Loading model from models/transformer/20260113_135051...
Model config: {'n_blocks': 6, 'd_model': 512, ...}
Loading dataset...

Using sample index: 0
Source: The cat sat on the mat
Target: De kat zat op de mat
Prediction: De kat zat op de mat

Source tokens (9): ['The', 'cat', 'sat', 'on', 'the', 'mat', '.', '<pad>', '<pad>']
Target tokens (8): ['<s>', 'De', 'kat', 'zat', 'op', 'de', 'mat', '.']

Generating attention visualizations for 6 decoder blocks...

Saved: output/attention_visualizations/block_00.png
Saved: output/attention_visualizations/block_01.png
Saved: output/attention_visualizations/block_02.png
Saved: output/attention_visualizations/block_03.png
Saved: output/attention_visualizations/block_04.png
Saved: output/attention_visualizations/block_05.png

Visualization complete!
```

### Visual Output

**Final Block Heads Mode:**
- Single figure showing all attention heads for the final decoder block
- Each subplot shows one attention head's cross-attention pattern
- Most informative for understanding what the model focuses on for final predictions

**Overview Mode:**
- Single figure with a grid of attention matrices (one per decoder block)
- Each block shows attention averaged across all heads
- Compact overview of how attention evolves through the model

**Comparison Mode:**
- Single figure with blocks arranged side-by-side
- Each block shows attention averaged across heads
- Easy to compare attention patterns between blocks

**Detailed Mode:**
- Multiple figures (one per decoder block)
- Each figure contains subplots for all attention heads
- Maximum detail for analyzing individual head behavior

Example attention patterns:
- **Block 0**: Broader, more scattered attention (learning general alignment)
- **Later blocks**: Sharper, more focused attention (refining specific alignments)
- **Diagonal-ish patterns**: Model aligns source and target in similar order
- **One-to-many**: One source token attending to multiple target positions (reordering)

## Key Implementation Details

### How Attention is Captured

The script modifies the decoding loop to:
1. Call `transformer.decode(..., return_attentions=True)` 
2. Extract `cross_attentions` from the returned dict
3. Store the final attention matrix for each block

### Attention Tensor Shape

- **Shape**: `(batch=1, n_heads, tgt_len, src_len)`
- **Values**: Normalized weights in [0, 1]
- **One matrix per decoding step**: As the model generates each token, it produces a new cross-attention matrix

### Token Handling

- **Special tokens included**: `<s>` (START), `</s>` (END), `<pad>` (PAD)
- **Trimmed to actual length**: Padding tokens are excluded from visualization
- **Thinned for readability**: If >24 tokens, every Nth token label is shown on axes

## Troubleshooting

### Model not found
```
FileNotFoundError: Model not found at models/transformer/20260113_135051/transformer_best.pt
```
→ Check that the model folder path is correct and contains `transformer_best.pt`

### Tokenizer not found
```
Tokenizer file not found at models/tokenizers/shared_tokenizer.json
```
→ Train the model first with `python -m src.translator.train`

### Out of memory on GPU
→ Try running on CPU: The script auto-detects available device, or set `device="cpu"` in code

### Matplotlib display issues (headless environment)
→ Use `--save_dir` to save as PNG files instead of displaying

## Integration with TensorBoard

This tool complements TensorBoard's visualization:
- **TensorBoard** shows samples logged during training (every N iterations)
- **This tool** lets you inspect any validation sample on-demand with full control

Both use the same underlying attention matrices from the model.

## Advanced: Batch Visualization

To visualize multiple samples programmatically:

```python
from src.translator.visualize_cross_attention import visualize_sample
from pathlib import Path

model_folder = "models/transformer/20260113_135051"
output_base = Path("output/all_samples")

for sample_idx in range(10):
    save_dir = output_base / f"sample_{sample_idx:03d}"
    visualize_sample(
        model_folder=model_folder,
        sample_idx=sample_idx,
        save_dir=str(save_dir),
    )
    print(f"Processed sample {sample_idx}")
```

## Interpreting Attention Patterns

### Good signs:
- **Block 0**: Broader, more scattered attention (learning general alignment)
- **Later blocks**: Sharper, more focused attention (refining specific alignments)
- **Diagonal-ish patterns**: Model aligns source and target in similar order
- **One-to-many**: One source token attending to multiple target positions (reordering)

### Red flags:
- **All uniform attention**: Model may not be learning meaningful alignments
- **Only attending to one source token**: Decoder may be ignoring input
- **No changes across blocks**: Model may have converged to trivial solution

## See Also

- [TensorBoard Attention Logs](../train.py): Run `tensorboard --logdir models/transformer/{timestamp}`
- [Transformer Architecture](../../transformer/Transformer.py): See `decode()` and `return_attentions` parameter
- [Training Script](./train.py): Where attention logs are originally created
