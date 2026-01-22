# Data Pipeline Module

Production-ready data pipeline for training transformer models on translation tasks.

## Features

- ✅ **Fully type-safe**: Complete type hints with TypedDict and dataclasses
- ✅ **Configurable**: Declarative configuration for all pipeline components
- ✅ **Composable**: Transform pipeline using clean abstractions
- ✅ **Extensible**: Easy to add new data providers and transforms
- ✅ **Testable**: Separation of concerns enables unit testing
- ✅ **Efficient**: Lazy evaluation and optimized preprocessing

## Architecture

```
Data Pipeline Flow:
┌─────────────┐
│  Provider   │  Load raw data from source
└──────┬──────┘
       │
       v
┌─────────────┐
│   Dataset   │  Apply preprocessing transforms
└──────┬──────┘
       │
       v
┌─────────────┐
│ DataLoader  │  Batch and iterate
└─────────────┘
```

### Components

1. **Data Providers** (`providers/`)
   - Load raw data from different sources
   - Handle dataset splitting
   - Validate data schema

2. **Preprocessing** (`preprocessing/`)
   - Composable transforms (tokenization, padding, masking)
   - Clean separation of concerns
   - Reusable across different datasets

3. **Dataset** (`dataset.py`)
   - PyTorch Dataset interface
   - Applies transform pipeline on-the-fly
   - Lazy evaluation for memory efficiency

4. **DataLoader Factory** (`dataloader.py`)
   - Centralized DataLoader creation
   - Consistent configuration
   - Utility functions for optimal settings

5. **Configuration** (`config.py`)
   - Type-safe dataclass configurations
   - Validation and error handling
   - Composable config objects

## Quick Start

### Basic Usage

```python
from src.data import (
    DataPipelineConfig,
    WikimediaDataProvider,
    create_dataloaders_from_config,
)
from src.tokenization.tokenizer import CustomTokenizer

# 1. Create configuration
config = DataPipelineConfig.create_default(
    provider_name="wikimedia",
    max_sequence_length=512,
    batch_size=16,
)

# 2. Initialize provider
provider = WikimediaDataProvider(config.provider_config)

# 3. Load tokenizer
tokenizer = CustomTokenizer("path/to/tokenizer.json")

# 4. Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders_from_config(
    provider=provider,
    source_tokenizer=tokenizer,
    target_tokenizer=tokenizer,
    config=config,
)

# 5. Train!
for batch in train_loader:
    # batch contains: source, target, source_mask, target_mask, label
    # All tensors are properly shaped and ready for training
    pass
```

### Custom Configuration

```python
from src.data import (
    DataPipelineConfig,
    DataProviderConfig,
    DatasetSplitConfig,
    SequenceConfig,
    TranslationConfig,
    PreprocessingConfig,
    DataLoaderConfig,
)

config = DataPipelineConfig(
    provider_config=DataProviderConfig(
        name="custom_dataset",
        deterministic=True,
        seed=42,
    ),
    split_config=DatasetSplitConfig(
        train=0.8,
        val=0.1,
        test=0.1,
    ),
    preprocessing_config=PreprocessingConfig(
        sequence_config=SequenceConfig(
            max_length=256,
            truncation=True,
            padding="max_length",
        ),
        translation_config=TranslationConfig(
            source_lang="en",
            target_lang="nl",
        ),
    ),
    train_loader_config=DataLoaderConfig(
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ),
    val_loader_config=DataLoaderConfig(
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    ),
)
```

## Available Data Providers

### 1. WikimediaDataProvider
Loads data from local preprocessed Wikimedia files.

```python
from src.data import WikimediaDataProvider, DataProviderConfig

config = DataProviderConfig(name="wikimedia")
provider = WikimediaDataProvider(config)
```

### 2. EuroParlDataProvider
Loads Helsinki-NLP/europarl from HuggingFace.

```python
from src.data import EuroParlDataProvider, DataProviderConfig

config = DataProviderConfig(name="europarl")
provider = EuroParlDataProvider(config)
```

### 3. OpusBooksDataProvider
Loads opus_books from HuggingFace.

```python
from src.data import OpusBooksDataProvider, DataProviderConfig

config = DataProviderConfig(name="opus_books")
provider = OpusBooksDataProvider(config)
```

### Creating Custom Providers

Extend `HuggingFaceDataProvider` or `LocalFileDataProvider`:

```python
from src.data.providers.base import HuggingFaceDataProvider
from src.data import DataProviderConfig
from datasets import Dataset

class MyCustomProvider(HuggingFaceDataProvider):
    def __init__(self, config: DataProviderConfig) -> None:
        super().__init__(
            config=config,
            dataset_name="my-username/my-dataset",
            dataset_config="en-nl",
        )

    def get_all_sentences(self, dataset: Dataset) -> tuple[list[str], list[str]]:
        # Extract sentences for tokenizer training
        source_sentences = [item["src"] for item in dataset]
        target_sentences = [item["tgt"] for item in dataset]
        return source_sentences, target_sentences
```

## Preprocessing Pipeline

The preprocessing pipeline is composable and extensible:

```
Raw Data → Tokenization → Special Tokens → Padding → Masking → Output
```

### Transform Types

1. **TokenizationTransform**: Convert text to token IDs
2. **AddSpecialTokensTransform**: Add START/END tokens
3. **PaddingTransform**: Pad/truncate sequences
4. **MaskingTransform**: Create attention masks

### Custom Transforms

```python
from src.data.preprocessing.base import Transform
from typing import TypedDict

class MyCustomTransform(Transform):
    def __call__(self, item: dict) -> dict:
        # Your custom preprocessing logic
        return processed_item
```

## Configuration Reference

### DataPipelineConfig
Top-level configuration containing all sub-configs.

```python
@dataclass(frozen=True)
class DataPipelineConfig:
    provider_config: DataProviderConfig
    split_config: DatasetSplitConfig
    preprocessing_config: PreprocessingConfig
    train_loader_config: DataLoaderConfig
    val_loader_config: DataLoaderConfig
    test_loader_config: Optional[DataLoaderConfig] = None
```

### DataProviderConfig
Configuration for data providers.

```python
@dataclass(frozen=True)
class DataProviderConfig:
    name: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "train"
    cache_dir: Optional[Path] = None
    deterministic: bool = True
    seed: int = 42
```

### DatasetSplitConfig
Train/val/test split ratios (must sum to 1.0).

```python
@dataclass(frozen=True)
class DatasetSplitConfig:
    train: float
    val: float
    test: float
```

### SequenceConfig
Sequence length and padding settings.

```python
@dataclass(frozen=True)
class SequenceConfig:
    max_length: int
    truncation: bool = True
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
```

### TranslationConfig
Translation task configuration.

```python
@dataclass(frozen=True)
class TranslationConfig:
    source_lang: str
    target_lang: str
    source_field: str = "en"
    target_field: str = "nl"
    translation_key: str = "translation"
```

### DataLoaderConfig
PyTorch DataLoader settings.

```python
@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
```

## Utility Functions

### compute_optimal_max_length
Analyze dataset to find optimal sequence length.

```python
from src.data import compute_optimal_max_length

max_length = compute_optimal_max_length(
    provider=provider,
    source_tokenizer=tokenizer,
    target_tokenizer=tokenizer,
    max_cap=512,
)
```

### compute_max_sequence_length
Get exact max lengths from a dataset.

```python
from src.data import compute_max_sequence_length

max_src, max_tgt = compute_max_sequence_length(
    dataset=dataset,
    source_tokenizer=tokenizer,
    target_tokenizer=tokenizer,
)
```

## Best Practices

### 1. Use Configuration Objects
Don't pass individual parameters; use configuration objects for better maintainability.

✅ **Good**:
```python
config = DataPipelineConfig.create_default(...)
dataloaders = create_dataloaders_from_config(..., config=config)
```

❌ **Bad**:
```python
# Passing many individual parameters
```

### 2. Compute Optimal Sequence Length
Don't hardcode max_length; compute it from your data.

```python
max_length = compute_optimal_max_length(provider, tokenizer, tokenizer)
```

### 3. Use Type Hints
The module is fully typed; leverage this for better IDE support and fewer bugs.

### 4. Separate Concerns
Don't mix data loading, preprocessing, and model training. Keep them in separate modules.

### 5. Configure num_workers
On Windows, use `num_workers=0`. On Linux, experiment with `num_workers=4` or higher.

```python
DataLoaderConfig(
    batch_size=32,
    num_workers=4,  # Adjust based on your system
    prefetch_factor=2,
    persistent_workers=True,
)
```

## Testing

The modular design makes testing easy:

```python
# Test provider
provider = EuroParlDataProvider(config)
splits = provider.load(split_config)
assert len(splits['train']) > 0

# Test transform
transform = TokenizationTransform(tokenizer, tokenizer)
result = transform(raw_item)
assert 'source_ids' in result

# Test dataset
dataset = TranslationDataset(raw_dataset, tokenizer, tokenizer, config)
item = dataset[0]
assert item['source'].shape[0] == config.sequence_config.max_length
```

## Performance Tips

1. **Use pin_memory=True** on GPU machines
2. **Adjust num_workers** based on CPU cores
3. **Use prefetch_factor** for async data loading
4. **Consider persistent_workers** for repeated epochs
5. **Profile your pipeline** to find bottlenecks

## Troubleshooting

### "splits must sum to 1.0"
Ensure train + val + test = 1.0:
```python
DatasetSplitConfig(train=0.7, val=0.15, test=0.15)  # ✅ sums to 1.0
```

### "prefetch_factor requires num_workers > 0"
Don't use prefetch_factor with num_workers=0:
```python
DataLoaderConfig(num_workers=0, prefetch_factor=None)  # ✅
```

### "File not found"
For LocalFileDataProvider, ensure files exist at the specified paths.

## Examples

See [examples/data_pipeline_example.py](../../examples/data_pipeline_example.py) for comprehensive examples.

## Migration

See [MIGRATION_GUIDE.md](../../MIGRATION_GUIDE.md) for migrating from the old data loading approach.
