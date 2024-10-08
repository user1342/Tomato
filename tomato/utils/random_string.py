import numpy as np
from typing import List, Tuple

class RandomString:
    """
    A class that generates uniform random strings following the FactoredMarginal protocol.
    """

    def __init__(self, num_chars: int, string_len: int) -> None:
        """
        Initializes the RandomString class.

        Args:
            num_chars (int): Number of characters in the alphabet.
            string_len (int): Length of the generated strings.

        Attributes:
            num_chars (int): Number of characters in the alphabet.
            string_len (int): Length of the generated strings.
            component_distributions (List[np.ndarray]): List of uniform distributions over the alphabet.
            ll (float): Log-likelihood of any generated string.
        """
        self.num_chars = num_chars
        self.string_len = string_len
        self.component_distributions = [
            np.ones(num_chars) / num_chars for _ in range(string_len)
        ]
        self.ll = self.string_len * np.log(1 / self.num_chars)

    def evaluate(self, x: List[int]) -> float:
        """
        Evaluates the log-likelihood of a given sequence.

        Args:
            x (List[int]): The sequence to evaluate.

        Returns:
            float: The log-likelihood of the sequence.
        """
        return self.ll

    def sample(self) -> Tuple[List[int], float]:
        """
        Samples a random sequence according to the uniform distribution.

        Returns:
            Tuple[List[int], float]: The sampled sequence and its log-likelihood.
        """
        seq = np.random.choice(range(self.num_chars), size=self.string_len).tolist()
        return seq, self.ll
