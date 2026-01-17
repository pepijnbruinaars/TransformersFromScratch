from pathlib import Path
from typing import Optional
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # type: ignore

from .constants import END_TOKEN, PAD_TOKEN, START_TOKEN, UNKNOWN_TOKEN


class CustomTokenizer:
    def __init__(
        self,
        path: Optional[str] = None,
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        """Initializes the CustomTokenizer.

        Args:
            path (Optional[str], optional): Path to the tokenizer file. If None, a new tokenizer will be created. Defaults to None.
            special_tokens (Optional[list[str]], optional): List of special tokens to use in the tokenizer. Defaults to [START_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN].
        """
        self._trained = False
        # Try to load the tokenizer from the specified path
        if path:
            try:
                self.load(path)
                self._trained = True
                return
            # If there are any issues loading the tokenizer, initialize a new one
            except (FileNotFoundError, ValueError, Exception):
                print(
                    f"Tokenizer file not found at {path}. Initializing a new tokenizer."
                )

        if special_tokens is None:
            special_tokens = [START_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]

        # Initialize the tokenizer with BPE model and pre-tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token=UNKNOWN_TOKEN))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore
        self.trainer = trainers.BpeTrainer(
            special_tokens=special_tokens,  # type: ignore
            min_frequency=2,  # type: ignore
        )

        self.path = path

    @property
    def trained(self) -> bool:
        """Whether the tokenizer has been trained already."""
        return self._trained

    def train(self, dataset: list[str]) -> None:
        """Trains the tokenizer on the provided dataset.

        Args:
            dataset (list[str]): A list of strings to train the tokenizer on.
        """
        self.tokenizer.train_from_iterator(dataset, trainer=self.trainer)
        self._trained = True

    def save(self, path: Optional[str]) -> None:
        """Saves the tokenizer to the specified path.

        Args:
            path (Optional[str]): The path to save the tokenizer file. If None, uses the existing path.
        Raises:
            ValueError: If no path is specified to save the tokenizer.
        """
        if path is None:
            path = self.path
        if path is None:
            raise ValueError("No path specified to save the tokenizer.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path: Optional[str]) -> None:
        """Loads the tokenizer from the specified path.

        Args:
            path (Optional[str]): The path to the tokenizer file. If None, uses the existing path.
        Raises:
            ValueError: If no path is specified to load the tokenizer.
            FileNotFoundError: If the tokenizer file does not exist at the specified path.
        """
        if path is None:
            path = self.path
        if path is None:
            raise ValueError("No path specified to load the tokenizer.")

        # Load the tokenizer from the specified path
        self.tokenizer = Tokenizer.from_file(path)

        # Update the path and set trained status
        self.path = path
        self._trained = True

        print(f"Tokenizer loaded from {path}")

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a list of token IDs back into a string.

        Args:
            token_ids (list[int]): A list of token IDs to decode.

        Returns:
            str: The decoded string corresponding to the token IDs.
        """
        decoded = self.tokenizer.decode(token_ids)
        return decoded

    def token_to_id(self, token: str) -> int:
        """Returns the ID of a token.

        Args:
            token (str): The token to convert to an ID.

        Returns:
            int: The ID of the token.
        """
        if not self._trained:
            raise ValueError("Tokenizer is not trained yet.")
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Returns the token corresponding to a given ID.

        Args:
            token_id (int): The ID of the token to convert to a string.

        Raises:
            ValueError: If the tokenizer has not been trained yet.

        Returns:
            str: The token corresponding to the given ID.
        """
        if not self._trained:
            raise ValueError("Tokenizer is not trained yet.")
        return self.tokenizer.id_to_token(token_id)

    def print_tokens(self, text: str) -> None:
        """Prints the tokens and their IDs for a given text.

        Args:
            text (str): The input text to tokenize and print.
        """
        encoded = self.tokenizer.encode(text)
        print(f"Original text: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Token IDs: {encoded.ids}")

    @property
    def vocabulary_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        if not self._trained:
            raise ValueError("Tokenizer is not trained yet.")
        return self.tokenizer.get_vocab_size()
    
    @property
    def special_tokens(self) -> dict[str, int]:
        """Returns a dictionary of special tokens and their IDs."""
        if not self._trained:
            raise ValueError("Tokenizer is not trained yet.")
        specials = {
            "start_token": START_TOKEN,
            "end_token": END_TOKEN,
            "pad_token": PAD_TOKEN,
            "unknown_token": UNKNOWN_TOKEN,
        }
        return {name: self.token_to_id(token) for name, token in specials.items()}
        
    @property
    def tokenizer_model(self) -> Tokenizer:
        """Returns the underlying Tokenizer model."""
        return self.tokenizer
