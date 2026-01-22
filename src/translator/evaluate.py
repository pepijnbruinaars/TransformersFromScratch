"""
Evaluation script for the transformer model (translator package).
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from ..constants import PAD_TOKEN, START_TOKEN, END_TOKEN
from .load_data import load_opus_data, get_max_sequence_length, get_sentences_from_data
from ..tokenization.tokenizer import CustomTokenizer
from ..models import Transformer
from ..utils import get_device
from .dataset import CustomDataset

def find_model_folders() -> list[Path]:
    model_dir = Path("models/transformer")
    if not model_dir.exists():
        return []
    folders = [f for f in model_dir.iterdir() if f.is_dir()]
    return sorted(folders)


def load_model_config(model_folder: Path) -> dict:
    config_path = model_folder / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Model configuration loaded from: {config_path}")
    return config


def load_model_checkpoint(
    model_folder: Path,
    epoch: int,
    transformer: Transformer,
    device: str
) -> dict:
    checkpoint_paths = [
        model_folder / f"transformer_epoch_{epoch}.pt",
        model_folder / "transformer_best.pt" if epoch == -1 else None,
        model_folder / "transformer_final.pt" if epoch == -2 else None,
    ]
    
    for path in checkpoint_paths:
        if path and path.exists():
            print(f"Loading checkpoint from: {path}")
            checkpoint = torch.load(path, map_location=device)
            transformer.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint
    
    raise FileNotFoundError(f"No checkpoint found for epoch {epoch} in {model_folder}")


def greedy_decode(
    transformer: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    max_length: int,
    device: str,
) -> list[int]:
    transformer.eval()
    with torch.no_grad():
        encoder_output = transformer.encode(source, source_mask)

    decoder_input = torch.tensor(
        [[target_tokenizer.token_to_id(START_TOKEN)]],
        dtype=torch.int64,
        device=device,
    )

    generated_tokens = []
    for _ in range(max_length):
        with torch.no_grad():
            decoder_mask = torch.triu(
                torch.ones(1, decoder_input.size(1), decoder_input.size(1), device=device),
                diagonal=1,
            ).type(torch.bool)
            decoder_mask = ~decoder_mask
            decoder_mask = decoder_mask.unsqueeze(0)

            decoder_output = transformer.decode(
                decoder_input,
                encoder_output,
                source_mask,
                decoder_mask,
            )

            projection = transformer.project(decoder_output)
            next_token = projection[:, -1, :].argmax(dim=-1)
            next_token_id = next_token.item()

            if next_token_id == target_tokenizer.token_to_id(END_TOKEN):
                break

            generated_tokens.append(next_token_id)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

    return generated_tokens


def evaluate_model(
    transformer: Transformer,
    test_dataloader: DataLoader,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    device: str,
    num_examples: int = 10,
    max_decode_length: int = 512,
) -> dict:
    transformer.eval()
    transformer.to(device)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    total_loss = 0.0
    total_batches = 0
    
    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=target_tokenizer.token_to_id(PAD_TOKEN),
        label_smoothing=0.0,
    ).to(device)
    
    examples_shown = 0
    all_examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            source_mask = batch["source_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            label = batch["label"].to(device)
            source_text = batch["source_text"]
            target_text = batch["target_text"]
            
            encoder_output = transformer.encode(source, source_mask)
            decoder_output = transformer.decode(
                target, encoder_output, source_mask, target_mask
            )
            projection = transformer.project(decoder_output)
            
            loss = loss_function(
                projection.view(-1, target_tokenizer.vocabulary_size),
                label.view(-1),
            )
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                total_batches += 1
            
            if examples_shown < num_examples:
                batch_size = source.size(0)
                for i in range(min(batch_size, num_examples - examples_shown)):
                    source_i = source[i:i+1]
                    source_mask_i = source_mask[i:i+1]
                    generated_ids = greedy_decode(
                        transformer,
                        source_i,
                        source_mask_i,
                        source_tokenizer,
                        target_tokenizer,
                        max_decode_length,
                        device,
                    )
                    predicted_text = target_tokenizer.decode(generated_ids)
                    example = {
                        "source": source_text[i],
                        "target": target_text[i],
                        "predicted": predicted_text,
                    }
                    all_examples.append(example)
                    print(f"\n{'-'*80}")
                    print(f"Example {examples_shown + 1}:")
                    print(f"  Source (EN): {source_text[i]}")
                    print(f"  Target (NL): {target_text[i]}")
                    print(f"  Predicted:   {predicted_text}")
                    print(f"{'-'*80}")
                    examples_shown += 1
                    if examples_shown >= num_examples:
                        break
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    results = {
        "avg_test_loss": avg_loss,
        "num_batches": total_batches,
        "examples": all_examples,
    }
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Number of Batches: {total_batches}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained transformer model")
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Epoch number to load (use -1 for best model, -2 for final model)",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Model folder path (e.g., models/transformer/20260106_165929). If not provided, uses the most recent folder.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of translation examples to show (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["validation", "test"],
        default="test",
        help="Which dataset to evaluate on (default: test)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save evaluation results to JSON file",
    )
    
    args = parser.parse_args()
    
    if args.model_folder:
        model_folder = Path(args.model_folder)
        if not model_folder.exists():
            print(f"Error: Model folder '{model_folder}' does not exist.")
            return
    else:
        folders = find_model_folders()
        if not folders:
            print("Error: No model folders found in models/transformer/")
            return
        model_folder = folders[-1]
        print(f"Using most recent model folder: {model_folder}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    print("\nLoading tokenizers...")
    dutch_tokenizer = CustomTokenizer("models/tokenizers/dutch_tokenizer.json")
    english_tokenizer = CustomTokenizer("models/tokenizers/english_tokenizer.json")
    
    if not dutch_tokenizer.trained or not english_tokenizer.trained:
        print("Error: Tokenizers are not trained. Please train the model first.")
        return
    
    print(f"Dutch vocabulary size: {dutch_tokenizer.vocabulary_size}")
    print(f"English vocabulary size: {english_tokenizer.vocabulary_size}")
    
    print("\nLoading model configuration...")
    model_config = load_model_config(model_folder)
    print(f"Model config: {json.dumps(model_config, indent=2)}")
    
    print("\nLoading data...")
    full_dataset = load_opus_data(1.0, 0.0, 0.0)
    train_raw, validation_raw, test_raw = load_opus_data(0.7, 0.15, 0.15)
    
    source_length, target_length = get_max_sequence_length(
        full_dataset[0], english_tokenizer, dutch_tokenizer
    )
    max_sequence_length = min(max(source_length, target_length), 512)
    
    if "source_length" in model_config:
        max_sequence_length = model_config["source_length"]
    
    print(f"Maximum sequence length: {max_sequence_length}")
    
    if args.dataset == "test":
        eval_raw = test_raw
        print(f"Test set size: {len(test_raw)}")
    else:
        eval_raw = validation_raw
        print(f"Validation set size: {len(validation_raw)}")
    
    eval_dataset = CustomDataset(
        eval_raw,
        source_tokenizer=english_tokenizer,
        target_tokenizer=dutch_tokenizer,
        sequence_length=max_sequence_length,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print("\nInitializing model...")
    transformer = Transformer(
        n_blocks=model_config["n_blocks"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        n_heads=model_config["n_heads"],
        dropout=model_config["dropout"],
        source_length=model_config["source_length"],
        target_length=model_config["target_length"],
        source_vocabulary_size=model_config["source_vocabulary_size"],
        target_vocabulary_size=model_config["target_vocabulary_size"],
    )
    
    print(f"\nLoading model from epoch {args.epoch}...")
    checkpoint = load_model_checkpoint(model_folder, args.epoch, transformer, device)
    print(f"Checkpoint loaded successfully!")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
    
    results = evaluate_model(
        transformer,
        eval_dataloader,
        english_tokenizer,
        dutch_tokenizer,
        device,
        num_examples=args.num_examples,
        max_decode_length=max_sequence_length,
    )
    
    if args.save_results:
        results_path = model_folder / f"evaluation_epoch_{args.epoch}_{args.dataset}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
