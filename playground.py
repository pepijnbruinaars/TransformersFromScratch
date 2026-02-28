"""Gradio playground for interactive text generation with the decoder-only transformer."""

import argparse
import os
import torch
import torch.nn.functional as F
import gradio as gr

from src.config.loader import ConfigLoader
from src.models.DecoderOnlyTransformer import DecoderOnlyTransformer
from src.tokenization.tokenizer import CustomTokenizer
from src.utils.device import get_device


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint directory and return path to last_state.pt."""
    subdirs = sorted(
        [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))],
        reverse=True,
    )
    for subdir in subdirs:
        candidate = os.path.join(checkpoint_dir, subdir, "last_state.pt")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No last_state.pt found in any subdirectory of {checkpoint_dir}")


def load_model(config_path: str, checkpoint_path: str):
    """Load the model and tokenizer from config and checkpoint."""
    device = get_device()
    config = ConfigLoader().from_yaml(config_path)
    mc = config.model_config

    model = DecoderOnlyTransformer(
        n_blocks=mc.n_block,
        d_model=mc.d_model,
        d_ff=mc.d_ff,
        n_heads=mc.n_head,
        dropout=mc.dropout_rate,
        vocab_size=config.data_config.vocab_size,
        sequence_length=mc.sequence_length,
        use_flash_attention=mc.use_flash_attention,
        activation=mc.activation,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = CustomTokenizer(path=config.data_config.tokenizer_path)

    print(f"Model loaded from {checkpoint_path} on {device}")
    return model, tokenizer, device


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    top_k_logits, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_logits[..., -1]
    return torch.where(logits >= min_top_k.unsqueeze(-1), logits, torch.full_like(logits, float("-inf")))


def nucleus_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumsum_probs > p
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float("-inf")
    return logits


def generate_text(
    model: torch.nn.Module,
    tokenizer: CustomTokenizer,
    device: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

            logits = model(input_ids, mask=causal_mask)
            next_token_logits = logits[0, -1, :]

            if temperature == 0:
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    next_token_logits = top_k_filter(next_token_logits, top_k)

                next_token_logits = nucleus_filter(next_token_logits, top_p)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            if next_token_id.item() == tokenizer.token_to_id("</s>"):
                break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Transformer Playground")
    parser.add_argument("--config", default="configs/tinystories_decoder_only.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="checkpoints/20260227_002835/epoch_2.pt", help="Path to checkpoint .pt file")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("checkpoints")

    model, tokenizer, device = load_model(args.config, checkpoint_path)

    def generate(prompt, temperature, top_p, top_k, max_new_tokens):
        if not prompt.strip():
            return ""
        return generate_text(model, tokenizer, device, prompt, temperature, top_p, int(top_k), int(max_new_tokens))

    with gr.Blocks(title="Transformer Playground") as demo:
        gr.Markdown("# Transformer Playground")
        gr.Markdown("Generate text with your trained decoder-only transformer.")

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", placeholder="Once upon a time", lines=3)
                output = gr.Textbox(label="Generated Text", lines=10, interactive=False)
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature", info="0 = greedy, higher = more random")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p (nucleus)", info="1.0 = disabled")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k", info="0 = disabled")
                max_new_tokens = gr.Slider(1, 256, value=64, step=1, label="Max new tokens")

        generate_btn.click(fn=generate, inputs=[prompt, temperature, top_p, top_k, max_new_tokens], outputs=output)
        prompt.submit(fn=generate, inputs=[prompt, temperature, top_p, top_k, max_new_tokens], outputs=output)

    demo.launch()


if __name__ == "__main__":
    main()
