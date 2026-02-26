"""Generation sampling utilities for monitoring model quality during training."""

import logging
from typing import List, Optional
import torch
import torch.nn.functional as F

from ..tokenization.tokenizer import CustomTokenizer
from ..utils.device import get_device

logger = logging.getLogger(__name__)


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

    def _generate(self, prompt: str, temperature: float) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature

        Returns:
            Generated text continuation
        """
        # Tokenize prompt
        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Generate tokens one by one
        for _ in range(self.max_new_tokens):
            # Forward pass
            logits = self.model(input_ids)
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k/nucleus sampling
            if self.top_k is not None:
                next_token_logits = self._top_k_filter(next_token_logits, self.top_k)

            next_token_logits = self._nucleus_filter(next_token_logits, self.top_p)

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append to input
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            # Stop if we generate end token
            if next_token_id.item() == self.tokenizer.token_to_id("</s>"):
                break

        # Decode and return
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text

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
