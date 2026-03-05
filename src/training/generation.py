"""Generation sampling utilities for monitoring model quality during training."""

import logging
import re
from typing import List, Optional
import torch
import torch.nn.functional as F

from ..models.components.DecoderOnly.KVCache import KVCache
from ..tokenization.tokenizer import CustomTokenizer
from ..utils.device import get_device

logger = logging.getLogger(__name__)


def _clean_decoded_text(text: str) -> str:
    """Fix spacing artifacts from BPE tokenization with a whitespace pre-tokenizer.

    Whitespace-based BPE tokenizers split on every space boundary, so punctuation
    and apostrophes become separate tokens. The decoder then joins all tokens with
    spaces, producing artefacts like "time , there", "didn ' t", "hide - and - seek".
    This function repairs those patterns for display purposes.
    """
    # Space before punctuation: "time , there" → "time, there"
    text = re.sub(r' ([.,!?;:])', r'\1', text)
    # Contractions and possessives: "didn ' t" → "didn't", "Lily ' s" → "Lily's"
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)
    # Hyphens between word characters: "hide - and - seek" → "hide-and-seek"
    text = re.sub(r'(\w) - (\w)', r'\1-\2', text)
    # Space after opening double quote: '" Hello' → '"Hello'
    text = re.sub(r'" (\w)', r'"\1', text)
    # Space before closing double quote: 'welcome "' → 'welcome"'
    text = re.sub(r'(\w) "', r'\1"', text)
    return text


def _word_distinct_ngrams(text: str) -> tuple[float, float]:
    """Compute word-level distinct-1 and distinct-2 for a single text.

    Distinct-n = |unique n-grams| / |total n-grams|.
    Values near 1.0 indicate high lexical diversity; low values signal
    repetition or mode collapse.

    Args:
        text: Decoded model output (post-processed display text).

    Returns:
        (distinct_1, distinct_2) floats in [0, 1].
    """
    words = text.lower().split()
    if not words:
        return 0.0, 0.0
    unigrams = words
    bigrams = list(zip(words[:-1], words[1:])) if len(words) > 1 else []
    d1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return d1, d2


class GenerationSampler:
    """Samples text from a generative model during training.

    Features:
    - Fixed prompts with multiple temperatures
    - Greedy and top-k/nucleus sampling
    - Logging of generated samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: CustomTokenizer,
        prompts: List[str],
        temperatures: List[float],
        max_new_tokens: int = 50,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
    ):
        """Initialize generation sampler.

        Args:
            model: The generative model
            tokenizer: CustomTokenizer instance
            prompts: List of prompts to generate from
            temperatures: List of temperatures to sample at
            max_new_tokens: Maximum new tokens to generate
            top_k: For top-k sampling (None = disabled)
            top_p: For nucleus sampling
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.temperatures = temperatures
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.device = get_device()

    def sample(self) -> str:
        """Generate samples for all prompts and temperatures.

        Returns:
            Formatted string with all generated samples
        """
        self.model.eval()
        output = []

        with torch.no_grad():
            for prompt in self.prompts:
                output.append(f"\n{'='*60}")
                output.append(f"Prompt: {prompt}")
                output.append(f"{'='*60}")

                for temp in self.temperatures:
                    generated = self._generate(prompt, temp)
                    output.append(f"Temperature {temp:.1f}: {generated}")

        self.model.train()
        return "\n".join(output)

    def sample_with_stats(self) -> tuple[str, dict[float, dict[str, float]]]:
        """Generate samples and compute distinct-n diversity metrics.

        Returns:
            A 2-tuple of:
            - Formatted display string (same as ``sample()``)
            - Stats dict: {temperature: {"distinct_1": float, "distinct_2": float}}
              Averaged across all prompts for that temperature.
        """
        self.model.eval()
        output = []
        # Accumulate per-temperature n-gram counts across all prompts
        temp_d1: dict[float, list[float]] = {t: [] for t in self.temperatures}
        temp_d2: dict[float, list[float]] = {t: [] for t in self.temperatures}

        with torch.no_grad():
            for prompt in self.prompts:
                output.append(f"\n{'='*60}")
                output.append(f"Prompt: {prompt}")
                output.append(f"{'='*60}")

                for temp in self.temperatures:
                    generated = self._generate(prompt, temp)
                    output.append(f"Temperature {temp:.1f}: {generated}")
                    d1, d2 = _word_distinct_ngrams(generated)
                    temp_d1[temp].append(d1)
                    temp_d2[temp].append(d2)

        self.model.train()

        stats: dict[float, dict[str, float]] = {}
        for temp in self.temperatures:
            d1_vals = temp_d1[temp]
            d2_vals = temp_d2[temp]
            stats[temp] = {
                "distinct_1": sum(d1_vals) / len(d1_vals) if d1_vals else 0.0,
                "distinct_2": sum(d2_vals) / len(d2_vals) if d2_vals else 0.0,
            }

        return "\n".join(output), stats

    def _generate(self, prompt: str, temperature: float) -> str:
        """Generate text from a prompt using a two-phase prefill + decode loop.

        When the model uses RoPE (no absolute positional encoding), a KV cache
        is used: the prompt is processed once (prefill), then each new token is
        passed individually (decode). This avoids recomputing attention over the
        full growing sequence at every step.

        For sinusoidal-PE models the full accumulated sequence is passed each
        step (original behaviour), because single-token decode would apply the
        wrong positional encoding to the new token.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated text continuation (post-processed for display)
        """
        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.token_to_id("</s>")

        # RoPE models: positional_encoding is None (positions live in Q/K via RoPE)
        use_kv_cache = (
            hasattr(self.model, "positional_encoding")
            and self.model.positional_encoding is None
        )
        kv_cache = KVCache(len(self.model.decoder_stack.blocks)) if use_kv_cache else None

        # --- Prefill: full prompt in one forward pass, populates KV cache ---
        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
        )
        logits = self.model(input_ids, mask=causal_mask, cache=kv_cache)
        next_token_id = self._sample_next_token(logits[0, -1, :], temperature)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        # --- Decode: one new token per step ---
        for _ in range(self.max_new_tokens - 1):
            if next_token_id.item() == eos_id:
                break

            if use_kv_cache:
                # Pass only the single new token; KV cache supplies all past context
                logits = self.model(input_ids[:, -1:], mask=None, cache=kv_cache)
                next_token_id = self._sample_next_token(logits[0, 0, :], temperature)
            else:
                # Sinusoidal PE: must pass full sequence for correct position encodings
                seq_len = input_ids.shape[1]
                causal_mask = torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
                )
                logits = self.model(input_ids, mask=causal_mask)
                next_token_id = self._sample_next_token(logits[0, -1, :], temperature)

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        generated_text = self.tokenizer.decode(input_ids[0].tolist())
        return _clean_decoded_text(generated_text)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample the next token from logits.

        Args:
            logits: (vocab_size,) unnormalised logits for the next position
            temperature: 0 = greedy argmax; > 0 = stochastic sampling

        Returns:
            Scalar tensor containing the chosen token ID
        """
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        if temperature != 1.0:
            logits = logits / temperature

        if self.top_k is not None:
            logits = self._top_k_filter(logits, self.top_k)

        logits = self._nucleus_filter(logits, self.top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filter logits to top-k values."""
        if k <= 0:
            return logits

        top_k_logits, _ = torch.topk(logits, k, dim=-1)
        min_top_k = top_k_logits[..., -1]

        logits = torch.where(
            logits >= min_top_k.unsqueeze(-1),
            logits,
            torch.full_like(logits, float("-inf")),
        )
        return logits

    @staticmethod
    def _nucleus_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""
        if p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumsum_probs > p
        # Keep at least one token
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

        return logits
