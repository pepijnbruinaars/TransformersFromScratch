# Configuration Guide

This guide covers every field in the YAML config system used to define and train transformer models. Configs are passed to `train.py` via `--config path/to/config.yaml`.

**Architecture support:**
- `encoder_decoder` — seq2seq tasks (e.g. translation). Requires `multi_corpus` data section.
- `decoder_only` — generative/language modelling tasks (e.g. story generation). Requires `generative` data section.

---

## Table of Contents

1. [Top-Level Fields](#1-top-level-fields)
2. [model](#2-model)
3. [training](#3-training)
   - [scheduler](#31-scheduler)
   - [optimizer](#32-optimizer)
4. [data](#4-data)
5. [checkpoint](#5-checkpoint)
6. [generative](#6-generative-decoder-only) *(decoder-only)*
   - [generative.dataset](#61-generativedataset)
   - [generative.preprocessing](#62-generativepreprocessing)
   - [generative.split](#63-generativesplit)
   - [generative.train\_loader / val\_loader](#64-generativetrain_loader--val_loader)
7. [generation](#7-generation-decoder-only) *(decoder-only)*
8. [multi\_corpus](#8-multi_corpus-encoder-decoder) *(encoder-decoder)*
   - [multi\_corpus.normalization](#81-multi_corpusnormalization)
   - [multi\_corpus.categories](#82-multi_corpuscategories)
   - [multi\_corpus.split](#83-multi_corpussplit)
   - [multi\_corpus.preprocessing](#84-multi_corpuspreprocessing)
   - [multi\_corpus.train\_loader / val\_loader / test\_loader](#85-multi_corpustrain_loader--val_loader--test_loader)
9. [runpod](#9-runpod-optional) *(optional)*
10. [Complete Templates](#10-complete-templates)
11. [Validation Rules](#11-validation-rules)

---

## 1. Top-Level Fields

```yaml
experiment_name: "my_experiment"   # str, required
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `experiment_name` | `str` | Yes | Unique identifier for the run. Used in log and checkpoint paths. |
| `model` | section | Yes | Architecture definition. |
| `training` | section | Yes | Optimizer, scheduler, and training loop settings. |
| `data` | section | Yes | Tokenizer and batch size. |
| `checkpoint` | section | Yes | Where and how often to save checkpoints. |
| `generative` | section | Decoder-only | Dataset and dataloader config for generative tasks. |
| `generation` | section | No | Sampling config for in-training text generation. |
| `multi_corpus` | section | Encoder-decoder | Dataset and dataloader config for translation tasks. |
| `runpod` | section | No | Cloud spot-instance support. |

---

## 2. model

Defines the transformer architecture.

```yaml
model:
  architecture: "decoder_only"
  n_blocks: 6
  n_heads: 8
  d_model: 384
  d_ff: 1536
  dropout_rate: 0.05
  activation: "gelu"
  use_flash_attention: true
  use_rope: false
  sequence_length: 256
```

| Field | Type | Required | Valid Values | Description |
|-------|------|----------|--------------|-------------|
| `architecture` | `str` | Yes | `"encoder_decoder"`, `"decoder_only"` | Model type. Determines which data section is used. |
| `n_blocks` | `int` | Yes | > 0 | Number of transformer layers (encoder and decoder each get this many for enc-dec). |
| `n_heads` | `int` | Yes | > 0, must evenly divide `d_model` | Number of attention heads. More heads → finer-grained attention patterns. |
| `d_model` | `int` | Yes | > 0 | Hidden/embedding dimension. Larger = more capacity, more memory. |
| `d_ff` | `int` | Yes | > 0 | Feed-forward intermediate dimension. Typically `4 × d_model`. |
| `dropout_rate` | `float` | Yes | `0.0`–`1.0` | Dropout probability. |
| `activation` | `str` | Yes | `"relu"`, `"gelu"`, `"swish"`, `"swiglu"` | MLP activation function. |
| `use_flash_attention` | `bool` | Yes | `true`, `false` | Whether to use Flash Attention v2. |
| `use_rope` | `bool` | Yes | `true`, `false` | Use Rotary Position Embedding instead of sinusoidal. |
| `sequence_length` | `int` or `null` | Decoder-only | > 0 | Maximum context length. Required for decoder-only models. |

### Guidance

**`n_blocks`** — Depth of the network. More blocks = better representation power, but slower training and higher memory use. Start small (4–6) and scale up only after confirming the training pipeline works.

**`n_heads` vs `d_model`** — `d_model` must be divisible by `n_heads`. A good default is `d_model / n_heads = 64` (i.e. 64-dim per head). Going below 32-dim per head tends to underperform.

**`d_ff`** — The standard is `4 × d_model`. With `swiglu` you typically use `8/3 × d_model` (rounded to a multiple of 64) because SwiGLU applies two projections, making the effective parameter count already larger.

**`dropout_rate`** — Use `0.0`–`0.1` for small models or when data is abundant. Increase to `0.1`–`0.3` when overfitting. For very small datasets, dropout may hurt more than help.

**`activation`**

| Option | Notes |
|--------|-------|
| `"relu"` | Fast, simple. Good baseline. Can cause "dying ReLU" in very deep networks. |
| `"gelu"` | Smooth approximation of ReLU. Standard in BERT/GPT-style models. Generally preferred over `relu`. |
| `"swish"` | Similar to `gelu`, slightly smoother. Marginal difference in practice. |
| `"swiglu"` | Gated variant (combines Swish + gating). Used in LLaMA and modern LLMs. Often improves loss at the cost of slightly more parameters in the MLP. Reduce `d_ff` by ~25% when switching from `gelu` to `swiglu` to keep parameter count similar. |

**`use_flash_attention`** — Enable on any modern GPU (Ampere A100/RTX 30xx or newer). It computes attention in tiled blocks, avoiding materialising the full O(n²) attention matrix in HBM — often 2–4× faster for long sequences and lower memory. Disable only if you hit CUDA compatibility issues.

**`use_rope`** — Rotary Position Embedding encodes position information into the Q/K vectors directly rather than adding learned or sinusoidal embeddings to the token representation. RoPE generalises better to sequence lengths not seen during training and is the default in most modern decoder-only models (LLaMA, Mistral, etc.). Choose RoPE when:
- You expect to do inference at lengths > `sequence_length`
- Training a generative/decoder-only model
- You care about strong positional generalisation

Sinusoidal (default when `use_rope: false`) is simpler, works well for fixed-length translation, and is less likely to cause bugs if you're unsure.

---

## 3. training

Controls the training loop.

```yaml
training:
  n_epochs: 3
  label_smoothing: 0.1
  use_mixed_precision: true
  validate_every_n_steps: 500
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  tensorboard_log_dir: "logs/"
  tensorboard_flush_frequency: 100
  logging_verbosity: 1
  scheduler:
    ...
  optimizer:
    ...
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `n_epochs` | `int` | Yes | — | Number of full passes over the training data. |
| `label_smoothing` | `float` | No | `0.1` | Smoothing factor for cross-entropy loss. `0.0` = hard labels. |
| `use_mixed_precision` | `bool` | No | `false` | Enable fp16/bf16 mixed precision training. |
| `validate_every_n_steps` | `int` | No | `1000` | How often (in steps) to run validation. |
| `gradient_accumulation_steps` | `int` | No | `1` | Accumulate gradients over N steps before updating. Simulates larger batches. |
| `gradient_clipping` | `float` | No | `1.0` | Max global gradient norm. Prevents exploding gradients. |
| `tensorboard_log_dir` | `str` | No | `"logs/"` | Directory for TensorBoard event files. |
| `tensorboard_flush_frequency` | `int` | No | `100` | Steps between TensorBoard flushes. |
| `logging_verbosity` | `int` | No | `1` | `0` = silent, `1` = normal progress, `2` = verbose per-step output. |

### Guidance

**`label_smoothing`** — Prevents the model from becoming overconfident by distributing a small probability mass (`label_smoothing`) uniformly over all tokens instead of concentrating it on the target. Values of `0.05`–`0.1` are common. Use `0.0` for pure language modelling (perplexity benchmarking) or when you find smoothing hurts convergence.

**`use_mixed_precision`** — Trains parameters in fp32 but computes forward/backward passes in fp16 (or bf16 on Ampere+). Reduces memory by ~2× and speeds up matmuls on tensor cores. Enable whenever training on a GPU. bf16 is generally more stable than fp16 because it preserves the dynamic range of fp32; the code uses GradScaler only for fp16.

**`gradient_accumulation_steps`** — If your GPU can only fit a batch of 8 but you want an effective batch of 64, set `gradient_accumulation_steps: 8`. Gradients are summed across micro-steps and the optimizer is called once per effective batch. This is equivalent to a larger batch size at the cost of slower wall-clock throughput.

**`gradient_clipping`** — Clips the global L2 norm of all parameter gradients to this value. `1.0` is a robust default. Increase to `5.0` if you're seeing overly conservative updates early in training; decrease or keep at `1.0` if loss spikes.

---

### 3.1 scheduler

Controls how the learning rate changes over training.

```yaml
training:
  scheduler:
    type: "cosine"
    learning_rate: 3e-4
    warmup_ratio: 0.02
    min_lr_ratio: 0.1
    decay_factor: 0.1      # step-based schedules only
    decay_steps: null       # step-based schedules only
    decay_rate: 0.9         # exponential schedule only
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | `str` | Yes | — | Scheduler algorithm name (e.g. `"cosine"`, `"linear"`). |
| `learning_rate` | `float` | Yes | — | Peak learning rate after warmup. |
| `warmup_ratio` | `float` | No | `0.05` | Fraction of total steps used for linear warmup. |
| `min_lr_ratio` | `float` | No | `0.0` | Final LR as a fraction of `learning_rate`. `0.1` means LR decays to 10% of peak. |
| `decay_factor` | `float` | No | `0.1` | Multiplicative decay per event for step-based schedules. |
| `decay_steps` | `int` or `null` | No | `null` | Steps between decay events for step-based schedules. |
| `decay_rate` | `float` | No | `0.9` | Per-step exponential decay rate. |

### Guidance

**`learning_rate`** — The single most impactful hyperparameter. A reasonable starting range for AdamW training of transformers is `1e-4`–`3e-4`. Smaller models can handle higher LRs; larger models typically need lower LRs. If loss diverges early, reduce LR. If training is very slow to start, increase LR or extend warmup.

**`type`** — Common choices:

| Schedule | When to use |
|----------|-------------|
| `"cosine"` | Default choice. Smooth decay to `min_lr_ratio × learning_rate`. Works well for fixed-epoch training. |
| `"linear"` | Simple and predictable. Decays linearly from peak to `min_lr_ratio × learning_rate`. |
| Exponential | Use `decay_rate` close to 1.0 for slow decay. Useful when you want continuous decay without a hard endpoint. |

**`warmup_ratio`** — Linear warmup from near 0 to `learning_rate` over the first fraction of training. Warmup prevents large, destabilising gradient updates when parameters are randomly initialised. `0.02`–`0.05` (2–5% of total steps) is standard. Use a longer warmup for larger models or higher learning rates.

**`min_lr_ratio`** — `0.0` means the LR decays all the way to zero at the end of training. `0.1` (10% of peak) is a common choice because keeping a non-zero final LR can slightly improve convergence on short training runs.

---

### 3.2 optimizer

```yaml
training:
  optimizer:
    name: "adam"
    weight_decay: 0.1
    betas: [0.9, 0.95]
    epsilon: 1e-8
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | No | `"adam"` | Optimizer algorithm. Currently `"adam"` (AdamW). |
| `weight_decay` | `float` | No | `1e-5` | L2 regularisation on weight matrices. Bias/norm/embedding params are excluded automatically. |
| `betas` | `[float, float]` | No | `[0.9, 0.999]` | Adam momentum coefficients `(β₁, β₂)`. |
| `epsilon` | `float` | No | `1e-8` | Adam numerical stability constant. |

### Guidance

**`weight_decay`** — Acts as L2 regularisation on weights (but not biases, norms, or embeddings, which are excluded automatically). `0.1` is the GPT/LLaMA-style default for training from scratch. Use `0.01`–`0.1` for most runs. Lower to `1e-5` or `0.0` if the model underfits.

**`betas`** — `β₁` controls the momentum of the gradient estimate; `β₂` controls the momentum of the squared-gradient estimate.
- `[0.9, 0.999]`: default from the original Adam paper, widely used.
- `[0.9, 0.95]`: used in GPT-style pretraining. Faster to forget old squared gradients, which can help when the loss landscape changes rapidly in early training.
- Rarely need to change `β₁` from `0.9`. Consider lowering `β₂` to `0.95` for long runs or when you observe slow convergence.

**`epsilon`** — Leave at `1e-8` unless you're using fp16 without bf16, in which case you might increase to `1e-7` or `1e-6` to avoid numerical issues with very small gradient magnitudes.

---

## 4. data

Basic tokenizer and batch settings.

```yaml
data:
  batch_size: 32
  tokenizer_path: "models/tokenizers/tinystories_tokenizer.json"
  vocab_size: 16000
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `batch_size` | `int` | Yes | — | Per-GPU batch size for the main data section. For fine-grained dataloader control, set `batch_size` inside `generative.train_loader` or `multi_corpus.train_loader`. |
| `tokenizer_path` | `str` | Yes | — | Path to the BPE tokenizer JSON file (HuggingFace tokenizers format). |
| `vocab_size` | `int` | Yes | `16000` | Vocabulary size of the tokenizer. Must match the tokenizer file. |

### Guidance

**`vocab_size`** — Larger vocabularies reduce average sequence length (fewer tokens per word) but increase the size of the embedding matrix and projection layer. For English-only small models, `8000`–`16000` is common. For multilingual or large-scale models, `32000`–`50000` is typical. Must exactly match the tokenizer you trained.

**`tokenizer_path`** — Paths are resolved relative to the project root (or the RunPod base path if enabled). Use forward slashes.

---

## 5. checkpoint

```yaml
checkpoint:
  save_dir: "checkpoints/"
  save_frequency: 600
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `save_dir` | `str` | Yes | Directory where checkpoint `.pt` files are saved. |
| `save_frequency` | `int` | Yes | Time in **seconds** between checkpoint saves. `600` = every 10 minutes. |

### Guidance

**`save_frequency`** — Set based on how long you can afford to lose progress in case of a crash. On a stable local machine, `600`–`1800` s is fine. On a preemptible cloud instance, set lower (`120`–`300` s) or use `runpod.checkpoint_every_n_steps` for step-based saving.

---

## 6. generative *(decoder-only)*

Full data pipeline configuration for decoder-only (language model) training. Only used when `model.architecture: "decoder_only"`.

```yaml
generative:
  text_field: "text"
  dataset: ...
  preprocessing: ...
  split: ...
  train_loader: ...
  val_loader: ...
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text_field` | `str` | Yes | Name of the column/field in the dataset that contains the raw text. |
| `dataset` | section | Yes | Dataset source and provider. |
| `preprocessing` | section | Yes | Tokenisation, sequence length, caching. |
| `split` | section | Yes | Train / val / test fractions. |
| `train_loader` | section | Yes | Training dataloader settings. |
| `val_loader` | section | Yes | Validation dataloader settings. |

---

### 6.1 generative.dataset

```yaml
generative:
  dataset:
    provider_name: "huggingface"
    dataset_name: "roneneldan/TinyStories"
    dataset_config: null
    split: "train"
    cache_dir: null
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider_name` | `str` | Yes | Data provider identifier (e.g. `"huggingface"`). |
| `dataset_name` | `str` or `null` | No | HuggingFace dataset ID (e.g. `"roneneldan/TinyStories"`). |
| `dataset_config` | `str` or `null` | No | Sub-config of a HuggingFace dataset (e.g. `"en-nl"` for a language pair dataset). Leave `null` if the dataset has no sub-configs. |
| `split` | `str` | No | Which split of the source dataset to load (`"train"`, `"test"`, etc.). The loader then performs its own train/val/test split defined by `generative.split`. |
| `cache_dir` | `str` or `null` | No | Override the HuggingFace cache directory. |

---

### 6.2 generative.preprocessing

```yaml
generative:
  preprocessing:
    sequence:
      max_length: 256
      truncation: true
      padding: "max_length"
    add_special_tokens: true
    return_attention_mask: true
    return_causal_mask: true
    cache_dir: "cache/generative"
    use_preprocessing_cache: true
    use_sequence_packing: false
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sequence.max_length` | `int` | Yes | — | Truncate/pad all sequences to this length. Must match `model.sequence_length`. |
| `sequence.truncation` | `bool` | Yes | — | Truncate sequences longer than `max_length`. |
| `sequence.padding` | `str` | Yes | — | `"max_length"`: pad all sequences to `max_length`. `"do_not_pad"`: leave sequences at natural length (use with sequence packing). |
| `add_special_tokens` | `bool` | No | `true` | Add `[BOS]` and `[EOS]` tokens. Required for causal language modelling. |
| `return_attention_mask` | `bool` | No | `true` | Return a padding mask alongside token IDs. Set `false` only if using sequence packing with no padding. |
| `return_causal_mask` | `bool` | No | `true` | Return a causal (autoregressive) mask. Must be `true` for decoder-only training. |
| `cache_dir` | `str` or `null` | No | `null` | Directory to cache preprocessed tensors. |
| `use_preprocessing_cache` | `bool` | No | `true` | Cache tokenised tensors to disk. Dramatically speeds up epoch 2+ (3–4×). Disable only during debugging or if disk space is limited. |
| `use_sequence_packing` | `bool` | No | `false` | Pack multiple documents end-to-end into each sequence, eliminating padding waste. Pairs with `padding: "do_not_pad"`. |

### Guidance

**`use_sequence_packing`** — When documents are short relative to `max_length` (e.g. TinyStories stories averaging 200 tokens but `max_length: 512`), a lot of each batch is padding. Packing concatenates multiple documents (separated by `[EOS]`) to fill each sequence slot fully. This can improve effective throughput by 30–70% for short-document datasets. Trade-off: slightly more complex attention masking; not useful when documents are already close to `max_length`.

**`use_preprocessing_cache`** — After the first epoch, tokenised data is read from disk instead of being re-tokenised. Always enable for multi-epoch training. Disable if you are iterating on the preprocessing code and need to see fresh results.

---

### 6.3 generative.split

```yaml
generative:
  split:
    train: 0.9
    val: 0.05
    test: 0.05
    max_train_size: null
```

| Field | Type | Required | Constraint | Description |
|-------|------|----------|-----------|-------------|
| `train` | `float` | Yes | 0.0–1.0 | Fraction of data for training. |
| `val` | `float` | Yes | 0.0–1.0 | Fraction of data for validation. |
| `test` | `float` | Yes | 0.0–1.0 | Fraction of data for testing. |
| `max_train_size` | `int` or `null` | No | > 0 | Cap the training set at this many examples. Useful for quick ablations. |

`train + val + test` must sum to `1.0` (±0.01 tolerance).

---

### 6.4 generative.train\_loader / val\_loader

```yaml
generative:
  train_loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    persistent_workers: true
    prefetch_factor: 2

  val_loader:
    batch_size: 64
    shuffle: false
    num_workers: 2
    pin_memory: true
    drop_last: false
    persistent_workers: true
    prefetch_factor: 2
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `batch_size` | `int` | Yes | — | Sequences per batch. |
| `shuffle` | `bool` | No | `true` | Shuffle data each epoch. Always `true` for train, `false` for val/test. |
| `num_workers` | `int` | No | `0` | Parallel data-loading workers. `0` = load in the main process. |
| `pin_memory` | `bool` | No | `true` | Pin CPU memory for faster GPU transfers. Enable when using CUDA. |
| `drop_last` | `bool` | No | `false` | Discard the last batch if it's smaller than `batch_size`. Enable for training to avoid batch-size inconsistency with gradient accumulation. |
| `persistent_workers` | `bool` | No | `false` | Keep worker processes alive between epochs (avoids spawn overhead). Requires `num_workers > 0`. |
| `prefetch_factor` | `int` or `null` | No | `null` | Batches each worker prefetches. Requires `num_workers > 0`. Minimum value: `2`. |

### Guidance

**`num_workers`** — On Linux/macOS, set to 4–8 for most datasets. On Windows, worker processes use `spawn` rather than `fork`, which has higher overhead; values of 2–4 are usually sufficient. For very fast datasets (e.g. cached tensors on NVMe), diminishing returns set in around 4 workers.

**`pin_memory`** — Set `true` whenever training on GPU. Has no effect on CPU training.

**`drop_last`** — Set `true` for the training loader when using gradient accumulation, to ensure every accumulated step has the same batch size. Set `false` for validation so no examples are skipped.

**`prefetch_factor` + `persistent_workers`** — Set `persistent_workers: true` and `prefetch_factor: 2` together with `num_workers > 0` to reduce per-epoch worker startup latency. If you set `num_workers: 0`, both must be absent or `false`.

---

## 7. generation *(decoder-only)*

Optional: sample generated text at regular intervals during training. Useful for qualitative monitoring.

```yaml
generation:
  enabled: true
  sample_every_n_steps: 1000
  num_samples: 1
  temperatures: [0.0, 0.5, 0.8, 1.0, 1.2]
  max_new_tokens: 64
  prompts:
    - "Once upon a time"
    - "The little girl looked up and"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | `bool` | No | `false` | Toggle generation sampling on/off. |
| `sample_every_n_steps` | `int` | No | `100` | Steps between generation runs. |
| `num_samples` | `int` | No | `1` | Samples to generate per prompt per temperature. |
| `temperatures` | `list[float]` | No | `[0.8, 1.0]` | Sampling temperatures to try. `0.0` = greedy (deterministic). |
| `max_new_tokens` | `int` | No | `50` | Maximum tokens to generate per sample. |
| `prompts` | `list[str]` | No | `["Once upon a time"]` | Prompts to condition generation on. |

### Guidance

**`temperatures`** — Temperature scales the logits before softmax. Lower temperature → sharper distribution → more predictable but repetitive output. Higher temperature → flatter distribution → more creative but potentially incoherent. Sampling a range (e.g. `[0.0, 0.5, 1.0, 1.4]`) during training gives a quick qualitative sense of model confidence and diversity at different stages.

---

## 8. multi\_corpus *(encoder-decoder)*

Full data pipeline for translation/seq2seq training with support for multiple weighted datasets. Only used when `model.architecture: "encoder_decoder"`.

```yaml
multi_corpus:
  sampling_strategy: "interleaved"
  random_seed: 42
  normalization: ...
  categories: ...
  split: ...
  preprocessing: ...
  train_loader: ...
  val_loader: ...
  test_loader: ...
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sampling_strategy` | `str` | No | `"interleaved"` | How to draw from multiple categories. |
| `random_seed` | `int` | No | `42` | Seed for deterministic sampling. |
| `normalization` | section | No | (defaults) | Text normalisation rules. |
| `categories` | list | Yes | — | Dataset categories with proportions. |
| `split` | section | Yes | — | Train/val/test fractions. |
| `preprocessing` | section | Yes | — | Tokenisation and sequence settings. |
| `train_loader` | section | Yes | — | Training dataloader. |
| `val_loader` | section | Yes | — | Validation dataloader. |
| `test_loader` | section | No | — | Test dataloader (optional). |

**`sampling_strategy`**

| Value | Behaviour |
|-------|-----------|
| `"interleaved"` | Samples are drawn from all categories in proportion within each batch. Provides consistent exposure across categories throughout training. Recommended for multi-domain training. |
| `"sequential"` | Processes one category fully before moving to the next. Useful for curriculum learning or when categories have very different sizes and you want clean epoch boundaries. |

---

### 8.1 multi\_corpus.normalization

```yaml
multi_corpus:
  normalization:
    enabled: true
    unicode_normalization: "NFKC"
    standardize_whitespace: true
    standardize_quotes: true
    standardize_dashes: true
    lowercase: false
    remove_control_chars: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `true` | Master toggle for all normalisation. |
| `unicode_normalization` | `str` | `"NFKC"` | Unicode form: `"NFC"`, `"NFKC"`, `"NFD"`, `"NFKD"`, or `"none"`. |
| `standardize_whitespace` | `bool` | `true` | Collapse multiple spaces, strip leading/trailing whitespace. |
| `standardize_quotes` | `bool` | `true` | Convert curly quotes (`'`, `"`) to ASCII equivalents. |
| `standardize_dashes` | `bool` | `true` | Convert en-dash (–) and em-dash (—) to hyphen (-). |
| `lowercase` | `bool` | `false` | Convert all text to lowercase. |
| `remove_control_chars` | `bool` | `true` | Strip non-printable control characters. |

### Guidance

**`unicode_normalization`** — `NFKC` is the most aggressive: it decomposes and recomposes characters AND maps compatibility characters (e.g. `ﬁ` → `fi`, `²` → `2`). Use `NFKC` for mixed-source corpora where encoding inconsistencies are common. Use `NFC` if you want canonical composition without compatibility substitution (preserves ligatures and special characters). Use `"none"` only if your data is already consistent.

**`lowercase`** — Lowercasing reduces vocabulary size but loses case information. For translation tasks, keep `false` because case carries grammatical meaning (proper nouns, sentence boundaries). For classification or embedding tasks it can reduce noise.

---

### 8.2 multi\_corpus.categories

```yaml
multi_corpus:
  categories:
    - name: "legal-ish"
      proportion: 0.5
      datasets:
        - provider_name: "europarl"
          dataset_name: "Helsinki-NLP/europarl"
          dataset_config: "en-nl"
          split: "train"
          proportion: 1.0

    - name: "literary"
      proportion: 0.3
      datasets:
        - provider_name: "opus_books"
          dataset_name: "Helsinki-NLP/opus_books"
          dataset_config: "en-nl"
          split: "train"
          proportion: 1.0

    - name: "conversational"
      proportion: 0.2
      datasets:
        - provider_name: "open_subtitles"
          lang1: "en"
          lang2: "nl"
          proportion: 1.0
```

**Category fields:**

| Field | Type | Required | Constraint | Description |
|-------|------|----------|-----------|-------------|
| `name` | `str` | Yes | Non-empty | Human-readable category label. |
| `proportion` | `float` | Yes | 0 < x ≤ 1.0 | Weight of this category in the final dataset. All category proportions must sum to `1.0`. |
| `datasets` | list | Yes | ≥ 1 | Dataset specs within this category. |

**Dataset fields (per entry in `datasets`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider_name` | `str` | Yes | Provider identifier (e.g. `"europarl"`, `"opus_books"`, `"open_subtitles"`). |
| `dataset_name` | `str` or `null` | No | HuggingFace dataset ID. |
| `dataset_config` | `str` or `null` | No | Dataset sub-config (e.g. language pair `"en-nl"`). |
| `split` | `str` | No | Source split to load (`"train"`, etc.). |
| `proportion` | `float` | Yes | Weight of this dataset within its category. Must sum to `1.0` across datasets in the same category. |
| `lang1` / `lang2` | `str` or `null` | No | Language codes (used by the OpenSubtitles provider). |
| `cache_dir` | `str` or `null` | No | Cache directory override. |
| `deterministic` | `bool` | No | Use seeded sampling. |
| `seed` | `int` | No | Random seed for this dataset. |

### Guidance

**Category proportions** — Use proportions to control domain balance. If you have 1M legal sentences and 100K literary sentences but want a 50/50 mix, set both proportions to `0.5` regardless of raw dataset sizes. The loader samples to achieve the target distribution.

**Multiple datasets within a category** — Use when a single domain has multiple data sources. Their `proportion` fields control the mix *within* that category.

---

### 8.3 multi\_corpus.split

```yaml
multi_corpus:
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    max_train_size: null
```

Same fields and constraints as [generative.split](#63-generativesplit).

---

### 8.4 multi\_corpus.preprocessing

```yaml
multi_corpus:
  preprocessing:
    sequence:
      max_length: 256
      truncation: true
      padding: "max_length"
    translation:
      source_lang: "en"
      target_lang: "nl"
      source_field: "en"
      target_field: "nl"
      translation_key: "translation"
    add_special_tokens: true
    return_attention_mask: true
    return_causal_mask: false
    cache_dir: "cache/preprocessing"
    use_preprocessing_cache: true
```

**Sequence fields:** same as [generative.preprocessing](#62-generativepreprocessing).

**Translation-specific fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `translation.source_lang` | `str` | Yes | `"en"` | BCP-47 code of the source language. |
| `translation.target_lang` | `str` | Yes | `"nl"` | BCP-47 code of the target language. |
| `translation.source_field` | `str` | Yes | `"en"` | Key within the row dict for source text. |
| `translation.target_field` | `str` | Yes | `"nl"` | Key within the row dict for target text. |
| `translation.translation_key` | `str` | Yes | `"translation"` | Top-level row key that contains the source/target sub-dict (e.g. `row["translation"]["en"]`). |

### Guidance

**`return_causal_mask`** — Set `false` for encoder-decoder models. The encoder uses full bidirectional attention; only the decoder uses causal masking, which is handled internally by the model. Set `true` for decoder-only.

**`translation_key`** — HuggingFace translation datasets typically store pairs as `{"translation": {"en": "...", "nl": "..."}}`. Set `translation_key: "translation"` and `source_field`/`target_field` to the language codes.

---

### 8.5 multi\_corpus.train\_loader / val\_loader / test\_loader

Same fields as [generative.train\_loader / val\_loader](#64-generativetrain_loader--val_loader). `test_loader` is optional.

---

## 9. runpod *(optional)*

Support for training on RunPod spot instances, which can be preempted at any time.

```yaml
runpod:
  enabled: true
  base_path: "/workspace"
  emergency_checkpoint_on_signal: true
  checkpoint_every_n_steps: 100
  auto_resume: true
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enabled` | `bool` | No | `false` | Enable RunPod spot instance support. |
| `base_path` | `str` | No | `"/workspace"` | Root of persistent network storage. All relative paths are resolved under here. |
| `emergency_checkpoint_on_signal` | `bool` | No | `true` | Save a checkpoint immediately on receiving `SIGTERM` (the preemption signal). |
| `checkpoint_every_n_steps` | `int` | No | `100` | Save a checkpoint every N training steps (independent of `checkpoint.save_frequency`). |
| `auto_resume` | `bool` | No | `true` | Automatically find and resume from the latest checkpoint on startup. |

### Guidance

On RunPod spot instances, set `emergency_checkpoint_on_signal: true` and `checkpoint_every_n_steps` to a value corresponding to no more than a few minutes of training. On standard persistent pods, you can rely on `checkpoint.save_frequency` alone.

---

## 10. Complete Templates

### Decoder-Only (Generative)

```yaml
experiment_name: "tinystories_decoder_v1"

model:
  architecture: "decoder_only"
  n_blocks: 6
  n_heads: 8
  d_model: 384
  d_ff: 1536
  dropout_rate: 0.05
  activation: "gelu"
  use_flash_attention: true
  use_rope: false
  sequence_length: 256

training:
  n_epochs: 3
  label_smoothing: 0.0
  use_mixed_precision: true
  validate_every_n_steps: 500
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  tensorboard_log_dir: "logs/"
  tensorboard_flush_frequency: 100
  logging_verbosity: 1
  scheduler:
    type: "cosine"
    learning_rate: 3e-4
    warmup_ratio: 0.02
    min_lr_ratio: 0.1
  optimizer:
    name: "adam"
    weight_decay: 0.1
    betas: [0.9, 0.95]
    epsilon: 1e-8

data:
  batch_size: 32
  tokenizer_path: "models/tokenizers/tinystories_tokenizer.json"
  vocab_size: 16000

checkpoint:
  save_dir: "checkpoints/"
  save_frequency: 600

generative:
  text_field: "text"

  dataset:
    provider_name: "huggingface"
    dataset_name: "roneneldan/TinyStories"
    dataset_config: null
    split: "train"

  preprocessing:
    sequence:
      max_length: 256
      truncation: true
      padding: "max_length"
    add_special_tokens: true
    return_attention_mask: true
    return_causal_mask: true
    cache_dir: "cache/generative"
    use_preprocessing_cache: true
    use_sequence_packing: false

  split:
    train: 0.9
    val: 0.05
    test: 0.05
    max_train_size: null

  train_loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: true
    persistent_workers: true
    prefetch_factor: 2

  val_loader:
    batch_size: 64
    shuffle: false
    num_workers: 2
    pin_memory: true
    drop_last: false
    persistent_workers: true
    prefetch_factor: 2

generation:
  enabled: true
  sample_every_n_steps: 1000
  num_samples: 1
  temperatures: [0.0, 0.5, 0.8, 1.0, 1.4]
  max_new_tokens: 64
  prompts:
    - "Once upon a time"
    - "In a magical forest"
```

---

### Encoder-Decoder (Translation)

```yaml
experiment_name: "en_nl_translation_v1"

model:
  architecture: "encoder_decoder"
  n_blocks: 6
  n_heads: 8
  d_model: 512
  d_ff: 2048
  dropout_rate: 0.1
  activation: "gelu"
  use_flash_attention: true
  use_rope: false

training:
  n_epochs: 20
  label_smoothing: 0.1
  use_mixed_precision: true
  validate_every_n_steps: 1000
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  tensorboard_log_dir: "logs/"
  tensorboard_flush_frequency: 100
  logging_verbosity: 1
  scheduler:
    type: "cosine"
    learning_rate: 1e-4
    warmup_ratio: 0.05
    min_lr_ratio: 0.1
  optimizer:
    name: "adam"
    weight_decay: 0.1
    betas: [0.9, 0.98]
    epsilon: 1e-9

data:
  batch_size: 32
  tokenizer_path: "models/tokenizers/europarl_tokenizer.json"
  vocab_size: 32000

checkpoint:
  save_dir: "checkpoints/"
  save_frequency: 600

multi_corpus:
  sampling_strategy: "interleaved"
  random_seed: 42

  normalization:
    enabled: true
    unicode_normalization: "NFKC"
    standardize_whitespace: true
    standardize_quotes: true
    standardize_dashes: true
    lowercase: false
    remove_control_chars: true

  categories:
    - name: "legal"
      proportion: 0.6
      datasets:
        - provider_name: "europarl"
          dataset_name: "Helsinki-NLP/europarl"
          dataset_config: "en-nl"
          split: "train"
          proportion: 1.0

    - name: "literary"
      proportion: 0.4
      datasets:
        - provider_name: "opus_books"
          dataset_name: "Helsinki-NLP/opus_books"
          dataset_config: "en-nl"
          split: "train"
          proportion: 1.0

  split:
    train: 0.8
    val: 0.1
    test: 0.1
    max_train_size: null

  preprocessing:
    sequence:
      max_length: 256
      truncation: true
      padding: "max_length"
    translation:
      source_lang: "en"
      target_lang: "nl"
      source_field: "en"
      target_field: "nl"
      translation_key: "translation"
    add_special_tokens: true
    return_attention_mask: true
    return_causal_mask: false
    cache_dir: "cache/preprocessing"
    use_preprocessing_cache: true

  train_loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: false
    persistent_workers: true
    prefetch_factor: 2

  val_loader:
    batch_size: 32
    shuffle: false
    num_workers: 2
    pin_memory: true
    drop_last: false
    persistent_workers: true
    prefetch_factor: 2
```

---

## 11. Validation Rules

The loader and dataclasses enforce these constraints at startup. A misconfigured file will raise an error before any training begins.

| Rule | Where enforced |
|------|---------------|
| `train + val + test` must sum to `1.0` (±0.01) | `DatasetSplitConfig.__post_init__` |
| All category `proportion` fields must sum to `1.0` (±0.01) | `MultiCorpusConfig.__post_init__` |
| All dataset `proportion` fields within a category must sum to `1.0` (±0.01) | `CategoryConfig.__post_init__` |
| `sequence.max_length` must be > 0 | `SequenceConfig.__post_init__` |
| `batch_size` must be > 0 | `DataLoaderConfig.__post_init__` |
| `num_workers` must be ≥ 0 | `DataLoaderConfig.__post_init__` |
| `prefetch_factor` requires `num_workers > 0` | `DataLoaderConfig.__post_init__` |
| `persistent_workers` requires `num_workers > 0` | `DataLoaderConfig.__post_init__` |
| `unicode_normalization` must be one of `NFC`, `NFKC`, `NFD`, `NFKD`, `none` | `TextNormalizationConfig.__post_init__` |
| `model.n_heads` must evenly divide `model.d_model` | Implicit in attention layer construction |
| `model.sequence_length` required when `architecture: "decoder_only"` | `train.py` |
| One of `translation_config` or `generative_config` must be set in preprocessing | `PreprocessingConfig.__post_init__` |

---

## YAML Key Reference (YAML name → Python attribute)

Some YAML keys differ from their Python dataclass attribute names:

| YAML key | Python attribute | Section |
|----------|-----------------|---------|
| `n_blocks` | `n_block` | `ModelConfig` |
| `n_heads` | `n_head` | `ModelConfig` |
| `n_epochs` | `num_epochs` | `TrainingConfig` |
| `scheduler.type` | `scheduler_config.name` | `TrainingConfig` |
| `gradient_accumulation_steps` | `optimizer_config.accumulation_steps` | Can be at `training.*` level |
| `gradient_clipping` | `optimizer_config.gradient_clipping` | Can be at `training.*` level |
