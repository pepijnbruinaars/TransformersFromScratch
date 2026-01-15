# Copilot Instructions: Transformers from Scratch

## Project Overview
This is a PyTorch implementation of the Transformer architecture (from "Attention Is All You Need" paper) being applied to English-Dutch neural machine translation. The codebase is structured to be educational with extensive type hints and modular components.

## Architecture Overview

### Core Components
- **Transformer** (`src/transformer/Transformer.py`): Encoder-decoder architecture with configurable blocks
  - **Encoder**: Processes source sequence (English)
  - **Decoder**: Generates target sequence (Dutch) with cross-attention to encoder
  - Built from reusable components in `src/transformer/components/`

- **Components** (`src/transformer/components/`):
  - `MultiHeadAttention.py`: Parallel attention heads for different representation subspaces
  - `FeedForward.py`: Position-wise FFN (d_model → d_ff → d_model)
  - `InputEmbedding.py`: Vocabulary embeddings
  - `PositionalEncoding.py`: Sine/cosine positional information
  - `ResidualConnection.py`: Skip connections around each sublayer
  - `LayerNormalization.py`: Pre-normalization of sublayer inputs
  - `EncoderDecoder.py`: Stacked encoder/decoder blocks

- **DecoderOnlyTransformer** (`src/transformer/DecoderOnlyTransformer.py`): Alternative GPT-style architecture (currently incomplete)

### Translation Pipeline (`src/translator/`)
- **Tokenizer** (`src/tokenizer.py`): BPE tokenizer, shared across source/target languages
  - Loads/trains from `models/tokenizers/`
  - Special tokens: `<s>` (START), `</s>` (END), `<pad>` (PAD), `<unk>` (UNK)
- **Dataset** (`dataset.py`): Converts raw pairs to padded tensors with attention masks
  - Produces `DatasetPair`: (source, target, src_mask, tgt_mask, label, source_text, target_text)
- **Training** (`train.py`): Main training loop with TensorBoard logging
  - Learning rate schedule: Warmup + cosine annealing
  - Greedy decoding for inference during validation
  - Model checkpoints in `models/transformer/<timestamp>/`

### Data Flow
1. Raw data: `data/wikimedia.en-nl.en` and `.nl` (parallel corpus)
2. Load via `load_data.py`: Splits into train/val/test
3. Tokenize with `CustomTokenizer`: IDs mapped by tokenizer JSON files
4. `CustomDataset`: Pads to max_length, adds special tokens, creates masks
5. `DataLoader`: Batches for training
6. Model forward pass → loss → backprop → checkpoint

## Key Patterns & Conventions

### Type Hints
All functions use type annotations (enforced by `src/constants.py` imports and codebase practice). This is critical for maintaining clarity in tensor operations.

### Device Handling
- Function `get_device()` in `train.py` returns 'cuda', 'mps', or 'cpu'
- Models moved to device via `.to(device)`
- TF32 enabled for stability: `torch.backends.cuda.matmul.allow_tf32 = True`

### Tensor Shapes
- Source/target tensors: `(batch_size, seq_length)`
- Embeddings: `(batch_size, seq_length, d_model)`
- Attention masks: `(batch_size, 1, seq_length, seq_length)` (causal for decoder)
- Logits: `(batch_size, seq_length, vocab_size)`

### Model Configuration
Config saved as JSON in checkpoint folder (e.g., `models/transformer/20260113_135051/model_config.json`):
```json
{"n_blocks": 6, "d_model": 512, "d_ff": 2048, "n_heads": 8, "dropout": 0.1, "source_length": 64, "target_length": 64}
```

## Critical Workflows

### Training
```bash
# Run from workspace root
python -m src.translator.train
```
- Creates timestamped folder in `models/transformer/`
- Saves best model, epoch checkpoints, loss histories, TensorBoard events
- Hyperparameters hardcoded in `train.py` (modify for experiments)

### Loading/Resuming Training
`train_utils.py` has `load_checkpoint()` and `save_checkpoint()` utilities that preserve RNG states (critical for reproducibility across CUDA/numpy/Python).

In addition, a convenience script `src/translator/resume.py` is provided to continue training from an existing run folder. The script:

- Loads the `model_config.json` from the run folder
- Reconstructs the `Transformer` and dataloaders using the shared tokenizer
- Loads a checkpoint (`transformer_epoch_N.pt`, `transformer_best.pt`, or `transformer_final.pt`) using `train_utils.load_checkpoint()`
- Restores model weights, optimizer state (when possible), and RNG states via `train_utils.restore_rng_states()`
- Continues training for a specified number of additional epochs

Example usage:

```bash
python -m src.translator.resume \
  --run-folder models/transformer/20260113_161242 \
  --checkpoint transformer_epoch_7.pt \
  --additional-epochs 5 \
  --device cuda
```

If `--run-folder` is omitted the script selects the latest run folder. If `--checkpoint` is omitted it will pick the latest checkpoint inside the run folder.

### Inference
`src/translator/inference.py` uses greedy decoding (`train_utils.greedy_decode_single`) to translate new sentences.

### Evaluation
`train_utils.py`: `calculate_bleu_chrf()` computes BLEU and CHRF scores on validation set.

## Important Implementation Details

### Attention Masks
- Encoder: No masking (processes full sequence)
- Decoder: Causal mask (triangular, prevents attending to future tokens)
- Function `attention_mask()` in `dataset.py` creates decoder mask

### Weight Initialization
`_initialize_weights()` in `Transformer.py`: 
- Linear layer weights: Normal(0, 0.02)
- Biases: zeros
- LayerNorm weights: ones (identity initialization)

### Special Tokens
Defined in `src/constants.py`. Used in `dataset.py`:
- Encoder input: source_tokens (no wrapping)
- Decoder input: START_TOKEN + target_tokens (shifted right)
- Labels: target_tokens + END_TOKEN (shifted right for next-token prediction)

### Checkpoint Structure
- `model_config.json`: Architecture parameters
- `transformer_best.pt`, `transformer_epoch_N.pt`: Model state dicts
- `loss_history.json`, `validation_loss_history.json`: Training curves
- `splits.json`: Train/val/test indices for reproducibility

## External Dependencies
- **PyTorch**: Core neural network framework (2.7.1)
- **HuggingFace tokenizers**: BPE tokenization (0.21.2)
- **datasets**: Loading Opus Books parallel corpus
- **sacrebleu**: CHRF metric computation
- **nltk**: BLEU score calculation
- **TensorBoard**: Training visualization (SummaryWriter)

## Common Debugging Notes
- **OOM errors**: Reduce batch size or sequence length in `train.py`
- **Tokenizer not found**: Ensure `train()` is called before `encode()` on `CustomTokenizer`
- **Shape mismatches**: Check `src_mask` dimensions in encoder vs decoder attention calls
- **Loss not decreasing**: Verify learning rate schedule; check if validation data overlaps train data
- **Greedy decoding stops early**: Ensure `end_id` parameter matches END_TOKEN ID from tokenizer
