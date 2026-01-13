"""
Interactive inference script for English-to-Dutch translation using a trained transformer model.

Usage:
    python -m src.translator.inference <model_folder_path>

Example:
    python -m src.translator.inference models/transformer/20260108_001540
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from ..constants import END_TOKEN, PAD_TOKEN, START_TOKEN
from ..tokenizer import CustomTokenizer
from ..transformer import Transformer


def get_device() -> str:
    """Returns the device to be used for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config(model_folder: Path) -> dict:
    """Load model configuration from JSON file."""
    config_path = model_folder / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"✓ Model configuration loaded from: {config_path}")
    return config


def load_model_checkpoint(
    model_folder: Path,
    transformer: Transformer,
    device: str,
) -> None:
    """Load the best model checkpoint."""
    checkpoint_paths = [
        model_folder / "transformer_best.pt",
        model_folder / "transformer_final.pt",
    ]
    
    for path in checkpoint_paths:
        if path.exists():
            print(f"✓ Loading checkpoint from: {path}")
            checkpoint = torch.load(path, map_location=device)
            transformer.load_state_dict(checkpoint["model_state_dict"])
            return
    
    raise FileNotFoundError(f"No checkpoint found (best.pt or final.pt) in {model_folder}")


def create_source_mask(source: torch.Tensor, pad_id: int, device: str) -> torch.Tensor:
    """Create attention mask for source sequence."""
    batch_size, seq_len = source.shape
    mask = (source != pad_id).unsqueeze(1).unsqueeze(1)
    return mask.to(device)


def create_decoder_mask(seq_len: int, device: str) -> torch.Tensor:
    """Create causal mask for decoder (autoregressive)."""
    mask = torch.triu(
        torch.ones(1, seq_len, seq_len, device=device),
        diagonal=1,
    ).type(torch.bool)
    mask = ~mask
    return mask.unsqueeze(0)


def greedy_decode(
    transformer: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    source_tokenizer: CustomTokenizer,
    target_tokenizer: CustomTokenizer,
    device: str,
    max_length: int = 512,
) -> str:
    """Perform greedy decoding to generate translation."""
    transformer.eval()
    
    with torch.no_grad():
        # Encode source sequence
        encoder_output = transformer.encode(source, source_mask)
        
        # Initialize decoder input with START token
        start_id = target_tokenizer.token_to_id(START_TOKEN)
        decoder_input = torch.tensor(
            [[start_id]],
            dtype=torch.int64,
            device=device,
        )
        
        # Greedily generate tokens
        generated_tokens = []
        end_id = target_tokenizer.token_to_id(END_TOKEN)
        
        for _ in range(max_length):
            # Create decoder mask
            decoder_mask = create_decoder_mask(decoder_input.size(1), device)
            
            # Decode
            decoder_output = transformer.decode(
                decoder_input,
                encoder_output,
                source_mask,
                decoder_mask,
            )
            
            # Project to vocabulary
            projection = transformer.project(decoder_output)
            
            # Get next token (argmax)
            next_token = projection[:, -1, :].argmax(dim=-1)
            next_token_id = next_token.item()
            
            # Stop if END token is generated
            if next_token_id == end_id:
                break
            
            generated_tokens.append(next_token_id)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        
        # Decode generated token IDs to text
        translation = target_tokenizer.decode(generated_tokens)
        return translation


def translate(
    english_text: str,
    transformer: Transformer,
    english_tokenizer: CustomTokenizer,
    dutch_tokenizer: CustomTokenizer,
    device: str,
) -> str:
    """Translate English text to Dutch."""
    # Tokenize input
    source_tokens = english_tokenizer.encode(english_text)
    source_tensor = torch.tensor([source_tokens], dtype=torch.int64, device=device)
    
    # Create mask
    pad_id = english_tokenizer.token_to_id(PAD_TOKEN)
    source_mask = create_source_mask(source_tensor, pad_id, device)
    
    # Generate translation
    translation = greedy_decode(
        transformer,
        source_tensor,
        source_mask,
        english_tokenizer,
        dutch_tokenizer,
        device,
    )
    
    return translation


def main():
    """Main inference loop."""
    parser = argparse.ArgumentParser(
        description="Interactive English-to-Dutch translator using a trained transformer model."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model folder (should contain model_config.json and transformer_best.pt)",
    )
    
    args = parser.parse_args()
    model_folder = Path(args.model_path)
    
    # Validate model folder
    if not model_folder.exists():
        print(f"❌ Model folder not found: {model_folder}")
        sys.exit(1)
    
    # Determine device
    device = get_device()
    print(f"✓ Using device: {device}")
    
    # Load configuration
    config = load_model_config(model_folder)
    
    # Load tokenizers
    print("✓ Loading tokenizers...")
    english_tokenizer = CustomTokenizer("models/tokenizers/english_tokenizer.json")
    dutch_tokenizer = CustomTokenizer("models/tokenizers/dutch_tokenizer.json")
    
    if not english_tokenizer.trained:
        print("❌ English tokenizer not trained!")
        sys.exit(1)
    if not dutch_tokenizer.trained:
        print("❌ Dutch tokenizer not trained!")
        sys.exit(1)
    
    # Create and load model
    print("✓ Building transformer model...")
    transformer = Transformer(
        n_blocks=config["n_blocks"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        source_length=config["source_length"],
        target_length=config["target_length"],
        source_vocabulary_size=english_tokenizer.vocabulary_size,
        target_vocabulary_size=dutch_tokenizer.vocabulary_size,
    )
    transformer.to(device)
    
    load_model_checkpoint(model_folder, transformer, device)
    transformer.eval()
    
    print("\n" + "="*80)
    print("English-to-Dutch Translator (Interactive Mode)")
    print("="*80)
    print("Type your English text and press Enter to translate.")
    print("Type 'exit' or 'quit' to exit.\n")
    
    # Interactive loop
    try:
        while True:
            try:
                user_input = input("English: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                # Translate
                translation = translate(
                    user_input,
                    transformer,
                    english_tokenizer,
                    dutch_tokenizer,
                    device,
                )
                
                print(f"Dutch:  {translation}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted.")
                break
            except Exception as e:
                print(f"Error during translation: {e}")
                print("Please try again.\n")
    
    except EOFError:
        print("\n\nEnd of input.")


if __name__ == "__main__":
    main()
