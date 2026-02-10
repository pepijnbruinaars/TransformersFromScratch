from typing import Dict, Any, Optional
import yaml
from pathlib import Path
from .base import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    SchedulerConfig,
    OptimizerConfig,
    CheckpointConfig,
    RunPodConfig,
    DataConfig,
)
from .environment import resolve_env_vars
from ..data.config import (
    MultiCorpusConfig,
    CategoryConfig,
    DatasetSpec,
    TextNormalizationConfig,
    DataProviderConfig,
    DatasetSplitConfig,
    PreprocessingConfig,
    SequenceConfig,
    TranslationConfig,
    DataLoaderConfig,
)


class ConfigLoader:
    """Loads YAML configuration files and maps them to ExperimentConfig dataclass."""

    def from_yaml(self, file_path: str) -> ExperimentConfig:
        """
        Load YAML configuration file and deserialize to ExperimentConfig.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            ExperimentConfig: Populated config dataclass

        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML is invalid
            KeyError: If required fields are missing
            TypeError: If field types don't match expectations
        """
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)

        if config_dict is None:
            raise ValueError("YAML file is empty or contains only comments")

        return self._dict_to_experiment_config(config_dict)

    def to_yaml(self, config: Any, file_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration object (dict or ExperimentConfig)
            file_path: Path where YAML will be saved
        """
        # Convert dataclass to dict if necessary
        if hasattr(config, "__dataclass_fields__"):
            config = self._dataclass_to_dict(config)

        with open(file_path, "w") as file:
            yaml.safe_dump(config, file, default_flow_style=False)

    @staticmethod
    def _dict_to_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
        """
        Convert dictionary to ExperimentConfig with nested dataclasses.

        Args:
            data: Dictionary from YAML

        Returns:
            ExperimentConfig: Fully populated config

        Raises:
            KeyError: If required fields are missing
            TypeError: If field types are incorrect
        """
        # Extract and build ModelConfig
        model_data = data.get("model", {})
        model_config = ModelConfig(
            n_block=model_data.get("n_blocks"),
            n_head=model_data.get("n_heads"),
            d_model=model_data.get("d_model"),
            d_ff=model_data.get("d_ff"),
            dropout_rate=model_data.get("dropout_rate"),
        )

        # Extract and build SchedulerConfig
        scheduler_data = data.get("training", {}).get("scheduler", {})
        scheduler_config = SchedulerConfig(
            name=scheduler_data.get("type"),
            learning_rate=float(scheduler_data.get("learning_rate", 1e-4)),
            warmup_ratio=float(scheduler_data.get("warmup_ratio", 0.05)),
            min_lr_ratio=float(scheduler_data.get("min_lr_ratio", 0.0)),
            decay_factor=float(scheduler_data.get("decay_factor", 0.1)),
            decay_steps=scheduler_data.get("decay_steps", None),
            decay_rate=float(scheduler_data.get("decay_rate", 0.9)),
        )

        # Extract and build OptimizerConfig
        optimizer_data = data.get("training", {}).get("optimizer", {})
        optimizer_config = OptimizerConfig(
            name=optimizer_data.get("name", "adam"),
            weight_decay=optimizer_data.get("weight_decay", 1e-5),
            betas=tuple(optimizer_data.get("betas", [0.9, 0.999])),
            epsilon=optimizer_data.get("epsilon", 1e-8),
            accumulation_steps=optimizer_data.get("accumulation_steps", 1),
        )

        # Extract and build TrainingConfig
        training_data = data.get("training", {})
        training_config = TrainingConfig(
            num_epochs=training_data.get("n_epochs"),
            scheduler_config=scheduler_config,
            optimizer_config=optimizer_config,
            logging_verbosity=training_data.get("logging_verbosity", 1),
        )

        # Extract and build CheckpointConfig
        checkpoint_data = data.get("checkpoint", {})
        checkpoint_config = CheckpointConfig(
            save_dir=checkpoint_data.get("save_dir", "./checkpoints"),
            save_frequency=checkpoint_data.get("save_frequency", 1000),
        )

        # Extract and build DataConfig
        data_config_data = data.get("data", {})
        data_config = DataConfig(
            batch_size=data_config_data.get("batch_size", 8),
            tokenizer_path=data_config_data.get(
                "tokenizer_path", "models/tokenizers/europarl_tokenizer.json"
            ),
        )

        # Extract and build RunPodConfig (optional)
        runpod_data = data.get("runpod", {})
        runpod_config = None
        if runpod_data:
            runpod_config = RunPodConfig(
                enabled=runpod_data.get("enabled", False),
                base_path=resolve_env_vars(
                    runpod_data.get("base_path", "/workspace")
                ),
                emergency_checkpoint_on_signal=runpod_data.get(
                    "emergency_checkpoint_on_signal", True
                ),
                checkpoint_every_n_steps=runpod_data.get("checkpoint_every_n_steps", 100),
                auto_resume=runpod_data.get("auto_resume", True),
            )

        # Extract and build MultiCorpusConfig (optional)
        multi_corpus_config = None
        if "multi_corpus" in data:
            multi_corpus_config = ConfigLoader._parse_multi_corpus_config(
                data["multi_corpus"]
            )

        # Build and return ExperimentConfig
        experiment_name = data.get("experiment_name", "default_experiment")
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            model_config=model_config,
            training_config=training_config,
            checkpoint_config=checkpoint_config,
            data_config=data_config,
            runpod_config=runpod_config,
            multi_corpus_config=multi_corpus_config,
        )

        return experiment_config

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary recursively with YAML field name mapping.

        Args:
            obj: Dataclass instance

        Returns:
            Dict: Dictionary representation with YAML-compatible structure
        """
        if not hasattr(obj, "__dataclass_fields__"):
            return obj

        result = {}
        for field_name, field in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)

            if hasattr(value, "__dataclass_fields__"):
                result[field_name] = ConfigLoader._dataclass_to_dict(value)
            elif isinstance(value, (list, tuple)):
                result[field_name] = list(value)
            else:
                result[field_name] = value

        # Restructure to match YAML format
        if "model_config" in result:
            model = result.pop("model_config")
            # Map Python field names to YAML names
            result["model"] = {
                "n_blocks": model.get("n_block"),
                "n_heads": model.get("n_head"),
                "d_model": model.get("d_model"),
                "d_ff": model.get("d_ff"),
                "dropout_rate": model.get("dropout_rate"),
            }

        if "training_config" in result:
            training = result.pop("training_config")
            training_out = {
                "n_epochs": training.get("num_epochs"),
                "logging_verbosity": training.get("logging_verbosity"),
            }
            if "scheduler_config" in training:
                scheduler = training.pop("scheduler_config")
                # Map scheduler field name to YAML convention
                training_out["scheduler"] = {
                    "type": scheduler.get("name"),
                    "warmup_ratio": scheduler.get("warmup_ratio"),
                }
            if "optimizer_config" in training:
                training_out["optimizer"] = training.pop("optimizer_config")
            result["training"] = training_out

        if "checkpoint_config" in result:
            result["checkpoint"] = result.pop("checkpoint_config")

        return result

    @staticmethod
    def _parse_multi_corpus_config(data: Dict[str, Any]) -> MultiCorpusConfig:
        """Parse multi-corpus configuration from YAML dictionary.

        Args:
            data: Dictionary from YAML multi_corpus section

        Returns:
            MultiCorpusConfig: Parsed multi-corpus configuration
        """
        # Parse normalization config
        norm_data = data.get("normalization", {})
        normalization_config = TextNormalizationConfig(
            enabled=norm_data.get("enabled", True),
            unicode_normalization=norm_data.get("unicode_normalization", "NFKC"),
            standardize_whitespace=norm_data.get("standardize_whitespace", True),
            standardize_quotes=norm_data.get("standardize_quotes", True),
            standardize_dashes=norm_data.get("standardize_dashes", True),
            lowercase=norm_data.get("lowercase", False),
            remove_control_chars=norm_data.get("remove_control_chars", True),
        )

        # Parse categories
        categories = []
        for cat_data in data.get("categories", []):
            # Parse dataset specs
            datasets = []
            for ds_data in cat_data.get("datasets", []):
                # Build provider config
                provider_config = DataProviderConfig(
                    name=ds_data.get("provider_name"),
                    dataset_name=ds_data.get("dataset_name"),
                    dataset_config=ds_data.get("dataset_config"),
                    split=ds_data.get("split", "train"),
                    cache_dir=Path(ds_data["cache_dir"]) if "cache_dir" in ds_data else None,
                    local_files={
                        k: Path(v) for k, v in ds_data["local_files"].items()
                    } if "local_files" in ds_data else None,
                    deterministic=True,
                    seed=data.get("random_seed", 42),
                    lang1=ds_data.get("lang1"),
                    lang2=ds_data.get("lang2"),
                )

                dataset_spec = DatasetSpec(
                    provider_name=ds_data.get("provider_name"),
                    provider_config=provider_config,
                    proportion=ds_data.get("proportion", 1.0),
                )
                datasets.append(dataset_spec)

            category = CategoryConfig(
                name=cat_data.get("name"),
                proportion=cat_data.get("proportion", 1.0),
                datasets=tuple(datasets),
            )
            categories.append(category)

        # Parse split config
        split_data = data.get("split", {})
        split_config = DatasetSplitConfig(
            train=split_data.get("train", 0.8),
            val=split_data.get("val", 0.1),
            test=split_data.get("test", 0.1),
            max_train_size=split_data.get("max_train_size"),
        )

        # Parse preprocessing config
        prep_data = data.get("preprocessing", {})
        seq_data = prep_data.get("sequence", {})
        trans_data = prep_data.get("translation", {})

        sequence_config = SequenceConfig(
            max_length=seq_data.get("max_length", 512),
            truncation=seq_data.get("truncation", True),
            padding=seq_data.get("padding", "max_length"),
        )

        translation_config = TranslationConfig(
            source_lang=trans_data.get("source_lang", "en"),
            target_lang=trans_data.get("target_lang", "nl"),
            source_field=trans_data.get("source_field", "en"),
            target_field=trans_data.get("target_field", "nl"),
            translation_key=trans_data.get("translation_key", "translation"),
        )

        preprocessing_config = PreprocessingConfig(
            sequence_config=sequence_config,
            translation_config=translation_config,
            add_special_tokens=prep_data.get("add_special_tokens", True),
            return_attention_mask=prep_data.get("return_attention_mask", True),
            return_causal_mask=prep_data.get("return_causal_mask", True),
            cache_dir=Path(prep_data["cache_dir"]) if "cache_dir" in prep_data else None,
        )

        # Parse dataloader configs
        train_loader_data = data.get("train_loader", {})
        train_loader_config = DataLoaderConfig(
            batch_size=train_loader_data.get("batch_size", 32),
            shuffle=train_loader_data.get("shuffle", True),
            num_workers=train_loader_data.get("num_workers", 0),
            pin_memory=train_loader_data.get("pin_memory", True),
            drop_last=train_loader_data.get("drop_last", False),
        )

        val_loader_data = data.get("val_loader", {})
        val_loader_config = DataLoaderConfig(
            batch_size=val_loader_data.get("batch_size", 32),
            shuffle=val_loader_data.get("shuffle", False),
            num_workers=val_loader_data.get("num_workers", 0),
            pin_memory=val_loader_data.get("pin_memory", True),
            drop_last=val_loader_data.get("drop_last", False),
        )

        test_loader_config = None
        if "test_loader" in data:
            test_loader_data = data["test_loader"]
            test_loader_config = DataLoaderConfig(
                batch_size=test_loader_data.get("batch_size", 32),
                shuffle=test_loader_data.get("shuffle", False),
                num_workers=test_loader_data.get("num_workers", 0),
                pin_memory=test_loader_data.get("pin_memory", True),
                drop_last=test_loader_data.get("drop_last", False),
            )

        return MultiCorpusConfig(
            categories=tuple(categories),
            split_config=split_config,
            preprocessing_config=preprocessing_config,
            train_loader_config=train_loader_config,
            val_loader_config=val_loader_config,
            test_loader_config=test_loader_config,
            normalization_config=normalization_config,
            sampling_strategy=data.get("sampling_strategy", "interleaved"),
            random_seed=data.get("random_seed", 42),
        )