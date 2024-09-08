from .model_wrapper import ModelWrapper
import numpy as np
from mec.utilities import log

class ModelMarginal:
    def __init__(
        self,
        prompt: str,
        max_len: int,
        temperature: float,
        k: int,
        model_name,
    ):
        """Autoregressive Marginal.

        Args:
            prompt: Prompt to condition on.
            max_len: Maximum length of the generated text.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.

        Attributes:
            max_len: Maximum length of the generated text.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.
            branching_factor: Attribute from `AutoRegressiveMarginal` protocol.
            lm_model: model.
            prompt: Prompt to condition on.
            mapping: Mapping from prefixes to feasible tokens.

        Raises:
            ValueError: If prompt is empty.
        """
        if len(prompt) == 0:
            raise ValueError
        self.max_len = max_len
        self.temperature = temperature
        self.k = k
        self.branching_factor = k
        self.lm_model = ModelWrapper(model_name)
        # Add newline to prompt to separate it from the query text.
        self.prompt = prompt + "\n"
        self.mapping: dict[tuple[int, ...], np.ndarray] = {}

    def conditional(self, prefix: list[int]) -> np.ndarray:
        """Implement method from `AutoRegressiveMarginal` protocol."""
        decoded_text = self.decode(prefix)
        conditional = self.lm_model.top_k_conditional(
            self.prompt + decoded_text, self.temperature, self.k
        )
        mask = conditional > 0
        self.mapping[tuple(prefix)] = np.arange(self.lm_model.vocab_size)[mask]
        return conditional[mask]

    def evaluate(self, prefix: list[int]) -> float:
        """Implement method from `SupportsEvaluate` protocol."""
        ll = 0
        for upper in range(len(prefix)):
            conditional = self.conditional(prefix[:upper])
            ll += log(conditional[prefix[upper]])
        return ll

    def is_terminal(self, prefix: list[int]) -> bool:
        """Implement method from `AutoRegressiveMarginal` protocol."""
        if len(prefix) > self.max_len:
            raise ValueError
        return len(prefix) == self.max_len

    def encode(self, text: str) -> list[int] | None:
        """Encode given into coupling representation.

        Args:
            text: Text to encode.

        Returns:
            Encoded prefix, None if encoding fails.
        """
        return self.lm_model.reduced_ids(self.prompt, text, self.k)

    def decode(self, prefix: list[int]) -> str:
        """Decode given prefix.

        Args:
            prefix: Prefix to decode.

        Returns:
            Decoded text.
        """
        decoded_tokens = []
        for k, z_i in enumerate(prefix):
            if tuple(prefix[:k]) not in self.mapping:
                self.conditional(prefix[:k])
            decoded_tokens.append(self.mapping[tuple(prefix[:k])][z_i])
        decoded_text = self.lm_model.tokenizer.decode(decoded_tokens)
        return decoded_text

    def sample(self) -> tuple[list[int], float]:
        """Implement method from `SupportsSample` protocol."""
        prefix: list[int] = []
        likelihoods: list[float] = []
        while len(prefix) < self.max_len:
            conditional = self.conditional(prefix)
            z_k = np.random.choice(range(len(conditional)), p=conditional)
            prefix.append(z_k)
            likelihoods.append(conditional[z_k])
        return prefix, np.log(likelihoods).sum()