from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from mec.utilities import log
from .model_wrapper import ModelWrapper

class ModelMarginal:
    """
    A class representing an autoregressive marginal model that conditions on a prompt 
    and generates sequences based on a given language model.
    """

    def __init__(
        self,
        prompt: str,
        max_len: int,
        temperature: float,
        k: int,
        model_name: str,
    ) -> None:
        """
        Initializes the ModelMarginal class.

        Args:
            prompt (str): Prompt to condition the generation on.
            max_len (int): Maximum length of the generated text.
            temperature (float): Temperature parameter for sampling.
            k (int): Number of top elements to consider during sampling.
            model_name (str): Name of the language model to use.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        self.max_len = max_len
        self.temperature = temperature
        self.k = k
        self.branching_factor = k
        self.lm_model = ModelWrapper(model_name)
        self.prompt = f"{prompt}\n"  # Add newline to separate prompt from query text
        self.mapping: Dict[Tuple[int, ...], np.ndarray] = {}

    def conditional(self, prefix: List[int]) -> np.ndarray:
        """
        Generates a conditional probability distribution over the next token given a prefix.

        Args:
            prefix (List[int]): The sequence of token IDs that represents the prefix.

        Returns:
            np.ndarray: An array of probabilities for the next token, conditioned on the prefix.
        """
        decoded_text = self.decode(prefix)
        conditional = self.lm_model.top_k_conditional(
            self.prompt + decoded_text, self.temperature, self.k
        )
        mask = conditional > 0
        self.mapping[tuple(prefix)] = np.arange(self.lm_model.vocab_size)[mask]
        return conditional[mask]

    def evaluate(self, prefix: List[int]) -> float:
        """
        Evaluates the log-likelihood of a given prefix under the model.

        Args:
            prefix (List[int]): The sequence of token IDs to evaluate.

        Returns:
            float: The log-likelihood of the prefix.
        """
        log_likelihood = 0.0
        for upper in range(len(prefix)):
            conditional = self.conditional(prefix[:upper])
            log_likelihood += log(conditional[prefix[upper]])
        return log_likelihood

    def is_terminal(self, prefix: List[int]) -> bool:
        """
        Checks if a given prefix has reached the maximum length and thus is terminal.

        Args:
            prefix (List[int]): The sequence of token IDs to check.

        Returns:
            bool: True if the prefix is terminal, False otherwise.

        Raises:
            ValueError: If the prefix length exceeds the maximum allowed length.
        """
        if len(prefix) > self.max_len:
            raise ValueError("Prefix length exceeds the maximum allowed length.")
        return len(prefix) == self.max_len

    def encode(self, text: str) -> Optional[List[int]]:
        """
        Encodes the given text into a sequence of token IDs.

        Args:
            text (str): The text to encode.

        Returns:
            Optional[List[int]]: The encoded sequence of token IDs, or None if encoding fails.
        """
        return self.lm_model.reduced_ids(self.prompt, text, self.k)

    def decode(self, prefix: List[int]) -> str:
        """
        Decodes a sequence of token IDs into a human-readable text string.

        Args:
            prefix (List[int]): The sequence of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        decoded_tokens = []
        for k, z_i in enumerate(prefix):
            if tuple(prefix[:k]) not in self.mapping:
                self.conditional(prefix[:k])
            decoded_tokens.append(self.mapping[tuple(prefix[:k])][z_i])
        decoded_text = self.lm_model.tokenizer.decode(decoded_tokens)
        return decoded_text

    def sample(self) -> Tuple[List[int], float]:
        """
        Samples a sequence of token IDs from the model.

        Returns:
            Tuple[List[int], float]: The sampled sequence of token IDs and its log-likelihood.
        """
        prefix: List[int] = []
        likelihoods: List[float] = []

        while len(prefix) < self.max_len:
            conditional = self.conditional(prefix)
            z_k = np.random.choice(len(conditional), p=conditional)
            prefix.append(z_k)
            likelihoods.append(conditional[z_k])

        total_log_likelihood = np.log(likelihoods).sum()
        return prefix, total_log_likelihood
