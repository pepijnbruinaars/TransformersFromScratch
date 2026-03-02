"""Gradio playground for interactive text generation with the decoder-only transformer."""

import argparse
import os
import time
import math
import torch
import torch.nn.functional as F
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config.loader import ConfigLoader
from src.models.DecoderOnlyTransformer import DecoderOnlyTransformer
from src.tokenization.tokenizer import CustomTokenizer
from src.utils.device import get_device

PROMPT_PRESETS = {
    "Custom": "",
    "Once upon a time": "Once upon a time",
    "In a magical forest": "In a magical forest",
    "A young wizard discovered": "A young wizard discovered",
    "The dragon flew over the mountains": "The dragon flew over the mountains",
    "One cold winter morning": "One cold winter morning",
}


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
        use_rope=mc.use_rope,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = CustomTokenizer(path=config.data_config.tokenizer_path)

    print(f"Model loaded from {checkpoint_path} on {device}")
    return model, tokenizer, device, mc.n_head, mc.n_block


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


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    """Penalize tokens that have already been generated."""
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def generate_streaming(
    model: torch.nn.Module,
    tokenizer: CustomTokenizer,
    device: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
):
    """Generator that yields partial results for streaming, plus final metadata."""
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prompt_len = len(token_ids)

    # Tracking for stats and visualization
    token_probs = []  # (token_str, chosen_prob, top10) per generated token
    log_probs_sum = 0.0
    generated_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

            logits = model(input_ids, mask=causal_mask)
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            generated_so_far = input_ids[0].tolist()
            next_token_logits = apply_repetition_penalty(next_token_logits, generated_so_far, repetition_penalty)

            if temperature == 0:
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                probs_all = F.softmax(next_token_logits, dim=-1)
                chosen_prob = probs_all[next_token_id.item()].item()
            else:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Compute full probs before filtering for top-10 display
                probs_all = F.softmax(next_token_logits, dim=-1)

                if top_k > 0:
                    next_token_logits = top_k_filter(next_token_logits, top_k)

                next_token_logits = nucleus_filter(next_token_logits, top_p)

                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                chosen_prob = probs[next_token_id.item()].item()

            # Collect top-10 candidates from pre-filter probs
            top10_probs, top10_ids = torch.topk(probs_all, 10, dim=-1)
            top10 = [
                (tokenizer.decode([tid.item()]), tp.item())
                for tid, tp in zip(top10_ids, top10_probs)
            ]

            token_str = tokenizer.decode([next_token_id.item()])
            token_probs.append((token_str, chosen_prob, top10))
            if chosen_prob > 0:
                log_probs_sum += math.log(chosen_prob)
            generated_count += 1

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            if next_token_id.item() == tokenizer.token_to_id("</s>"):
                break

            # Yield streaming text
            generated_ids = input_ids[0].tolist()
            yield tokenizer.decode(generated_ids), None, None, None

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_count / elapsed if elapsed > 0 else 0
    perplexity = math.exp(-log_probs_sum / generated_count) if generated_count > 0 else float("inf")

    # Final attention pass for heatmap (only on the last step to avoid slowing down generation)
    last_attentions = None
    with torch.no_grad():
        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        _, last_attentions = model(input_ids, mask=causal_mask, return_attentions=True)

    generated_ids = input_ids[0].tolist()
    final_text = tokenizer.decode(generated_ids)

    stats = (
        f"**Tokens generated:** {generated_count}\n\n"
        f"**Time:** {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/sec)\n\n"
        f"**Perplexity:** {perplexity:.2f}"
    )

    yield final_text, token_probs, last_attentions, stats


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _prob_color(prob: float) -> str:
    """Smooth green-yellow-red gradient based on probability via HSL."""
    hue = int(120 * prob)
    return f"hsl({hue}, 70%, 35%)"


def build_token_prob_html(token_probs: list[tuple[str, float, list]]) -> str:
    """Build colored HTML with pure-CSS hover popups showing top-10 candidates."""
    if not token_probs:
        return ""

    # All CSS inline in the HTML output so it works inside Gradio's sandboxed component
    style = (
        '<style>'
        '.tp-wrap{font-family:Consolas,Monaco,Courier New,monospace;font-size:13px;'
        'line-height:2.2;padding:12px 14px;background:#0f0f1a;border-radius:10px;'
        'color:#e0e0e0;word-wrap:break-word;overflow:visible;}'
        '.tp{position:relative;display:inline-block;padding:2px 5px;margin:2px 1px;'
        'border-radius:4px;color:#fff;cursor:default;transition:filter .15s;'
        'font-size:13px;text-shadow:0 1px 2px rgba(0,0,0,.5);}'
        '.tp:hover{filter:brightness(1.3);z-index:100;}'
        '.tp:hover .tt{display:block;}'
        '.tt{display:none;position:absolute;top:calc(100% + 6px);left:50%;'
        'transform:translateX(-50%);width:280px;z-index:1000;'
        'background:#1e1e2e;border:1px solid #3a3a5c;border-radius:8px;'
        'box-shadow:0 8px 24px rgba(0,0,0,.5);padding:10px 12px;'
        'font-size:11px;color:#ccc;text-align:left;pointer-events:none;'
        'line-height:1.4;}'
        '.tt::before{content:"";position:absolute;bottom:100%;left:50%;'
        'transform:translateX(-50%);border:6px solid transparent;'
        'border-bottom-color:#3a3a5c;}'
        '.tt-h{font-size:12px;font-weight:600;color:#fff;margin-bottom:8px;'
        'padding-bottom:6px;border-bottom:1px solid #3a3a5c;}'
        '.tt-r{display:flex;align-items:center;gap:6px;margin:3px 0;'
        'font-size:11px;line-height:1.5;}'
        '.tt-r.ch{color:#7dd3fc;font-weight:600;}'
        '.tt-rk{width:18px;text-align:right;color:#888;flex-shrink:0;}'
        '.tt-lb{width:80px;overflow:hidden;text-overflow:ellipsis;'
        'white-space:nowrap;flex-shrink:0;}'
        '.tt-bg{flex:1;height:10px;background:#2a2a3e;border-radius:3px;'
        'overflow:hidden;min-width:60px;}'
        '.tt-br{height:100%;border-radius:3px;}'
        '.tt-pv{width:48px;text-align:right;color:#aaa;flex-shrink:0;'
        'font-variant-numeric:tabular-nums;}'
        '</style>'
    )

    parts = [style, '<div class="tp-wrap">']

    for token_str, prob, top10 in token_probs:
        bg = _prob_color(prob)
        display = _escape_html(token_str).replace(" ", "&nbsp;")

        max_p = top10[0][1] if top10 else 1.0

        rows = []
        for rank, (t10_str, t10_prob) in enumerate(top10, 1):
            is_chosen = (t10_str == token_str)
            bar_w = (t10_prob / max_p * 100) if max_p > 0 else 0
            bar_c = "hsl(200,80%,55%)" if is_chosen else "hsl(260,40%,50%)"
            cls = " ch" if is_chosen else ""
            label = _escape_html(t10_str) or "&nbsp;"
            rows.append(
                f'<div class="tt-r{cls}">'
                f'<span class="tt-rk">{rank}.</span>'
                f'<span class="tt-lb">{label}</span>'
                f'<span class="tt-bg"><span class="tt-br" '
                f'style="width:{bar_w:.1f}%;background:{bar_c}"></span></span>'
                f'<span class="tt-pv">{t10_prob:.3f}</span>'
                f'</div>'
            )

        tooltip = (
            f'<div class="tt">'
            f'<div class="tt-h">p = {prob:.4f}</div>'
            f'{"".join(rows)}'
            f'</div>'
        )

        parts.append(f'<span class="tp" style="background:{bg}">{display}{tooltip}</span>')

    parts.append("</div>")
    return "".join(parts)


def _get_tokens_and_attn(attentions: list, tokenizer: CustomTokenizer, input_ids: torch.Tensor,
                         layer: int, head: int, max_display: int = 40):
    """Extract attention matrix and token labels, truncated to max_display."""
    attn = attentions[layer][0, head].cpu().numpy()
    if attn.shape[0] > max_display:
        attn = attn[-max_display:, -max_display:]
        ids = input_ids[0, -max_display:].tolist()
    else:
        ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]).strip() or f"[{tid}]" for tid in ids]
    tokens = [t[:8] for t in tokens]
    return attn, tokens


def build_attention_grid(attentions: list, tokenizer: CustomTokenizer, input_ids: torch.Tensor,
                         n_blocks: int, n_heads: int) -> plt.Figure:
    """Build a grid of attention heatmaps: rows = layers, columns = heads."""
    fig, axes = plt.subplots(n_blocks, n_heads, figsize=(n_heads * 2.5, n_blocks * 2.5))

    for layer in range(n_blocks):
        for head in range(n_heads):
            ax = axes[layer, head] if n_blocks > 1 else axes[head]
            attn, tokens = _get_tokens_and_attn(attentions, tokenizer, input_ids, layer, head)
            ax.imshow(attn, cmap="Blues", aspect="auto")
            ax.set_title(f"L{layer} H{head}", fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Attention Patterns (all layers x heads)", fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def build_attention_detail(attentions: list, tokenizer: CustomTokenizer, input_ids: torch.Tensor,
                           layer: int, head: int) -> plt.Figure:
    """Build a detailed attention heatmap for a single layer/head with token labels."""
    attn, tokens = _get_tokens_and_attn(attentions, tokenizer, input_ids, layer, head)

    fig, ax = plt.subplots(figsize=(min(12, len(tokens) * 0.4 + 2), min(10, len(tokens) * 0.35 + 2)))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(f"Layer {layer}, Head {head}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Transformer Playground")
    parser.add_argument("--config", default="configs/tinystories_decoder_only.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="checkpoints/20260227_002835/epoch_2.pt", help="Path to checkpoint .pt file")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("checkpoints")

    model, tokenizer, device, n_heads, n_blocks = load_model(args.config, checkpoint_path)

    # State to hold last generation results for attention visualization
    last_state = {"attentions": None, "input_ids": None}

    def on_preset_change(preset_name):
        if preset_name == "Custom":
            return gr.update()
        return PROMPT_PRESETS.get(preset_name, "")

    def generate(prompt, temperature, top_p, top_k, max_new_tokens, repetition_penalty):
        if not prompt.strip():
            yield "", "", ""
            return

        for text, token_probs, attentions, stats in generate_streaming(
            model, tokenizer, device, prompt, temperature, top_p,
            int(top_k), int(max_new_tokens), repetition_penalty,
        ):
            if attentions is not None:
                # Final yield — store state and build visualizations
                last_state["attentions"] = attentions
                last_state["input_ids"] = torch.tensor(
                    [tokenizer.encode(text)], dtype=torch.long, device=device
                )
                prob_html = build_token_prob_html(token_probs)
                yield text, prob_html, stats
            else:
                # Streaming yield — text only
                yield text, "", ""

    def update_attention_grid():
        if last_state["attentions"] is None:
            return None
        return build_attention_grid(
            last_state["attentions"], tokenizer, last_state["input_ids"],
            n_blocks, n_heads,
        )

    def update_attention_detail(layer, head):
        if last_state["attentions"] is None:
            return None
        return build_attention_detail(
            last_state["attentions"], tokenizer, last_state["input_ids"],
            int(layer), int(head),
        )

    with gr.Blocks(title="Transformer Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Transformer Playground")
        gr.Markdown("Generate text with your trained decoder-only transformer.")

        with gr.Row():
            # Left column: prompt and output
            with gr.Column(scale=2):
                with gr.Row():
                    preset = gr.Dropdown(
                        choices=list(PROMPT_PRESETS.keys()),
                        value="Custom",
                        label="Prompt presets",
                        scale=1,
                    )
                prompt = gr.Textbox(label="Prompt", placeholder="Once upon a time", lines=3)
                generate_btn = gr.Button("Generate", variant="primary")
                output = gr.Textbox(label="Generated Text", lines=8, interactive=False)

            # Right column: controls
            with gr.Column(scale=1):
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature",
                                        info="0 = greedy, higher = more random")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p (nucleus)",
                                  info="1.0 = disabled")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k",
                                  info="0 = disabled")
                max_new_tokens = gr.Slider(1, 256, value=64, step=1, label="Max new tokens")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.05,
                                               label="Repetition penalty",
                                               info="1.0 = disabled, higher = less repetition")

        # Visualization tabs
        with gr.Tabs():
            with gr.TabItem("Token Probabilities"):
                token_prob_display = gr.HTML(label="Token Probabilities")

            with gr.TabItem("Attention Grid"):
                grid_btn = gr.Button("Show All Layers x Heads")
                attn_grid_plot = gr.Plot(label="Attention Grid")

            with gr.TabItem("Attention Detail"):
                with gr.Row():
                    attn_layer = gr.Slider(0, n_blocks - 1, value=0, step=1, label="Layer")
                    attn_head = gr.Slider(0, n_heads - 1, value=0, step=1, label="Head")
                detail_btn = gr.Button("Show Detail")
                attn_detail_plot = gr.Plot(label="Attention Detail")

            with gr.TabItem("Generation Stats"):
                stats_display = gr.Markdown()

        # Events
        preset.change(fn=on_preset_change, inputs=[preset], outputs=[prompt])

        gen_inputs = [prompt, temperature, top_p, top_k, max_new_tokens, repetition_penalty]
        gen_outputs = [output, token_prob_display, stats_display]
        generate_btn.click(fn=generate, inputs=gen_inputs, outputs=gen_outputs)
        prompt.submit(fn=generate, inputs=gen_inputs, outputs=gen_outputs)

        grid_btn.click(fn=update_attention_grid, inputs=[], outputs=[attn_grid_plot])
        detail_btn.click(fn=update_attention_detail, inputs=[attn_layer, attn_head], outputs=[attn_detail_plot])

    demo.launch()


if __name__ == "__main__":
    main()
