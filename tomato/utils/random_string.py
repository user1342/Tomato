import numpy as np

class RandomString:
    def __init__(self, num_chars: int, string_len: int):
        """A FactoredMarginal that generates uniform random strings.

        Args:
            num_chars: Number of characters in the alphabet.
            string_len: Length of the generated strings.

        Attributes:
            num_chars: Number of characters in the alphabet.
            string_len: Length of the generated strings.
            component_distributions: Attribute from FactoredMarginal protocol.
            ll: Log-likelihood of states.
        """
        self.num_chars = num_chars
        self.string_len = string_len
        self.component_distributions = [
            np.ones(num_chars) / num_chars for _ in range(string_len)
        ]
        self.ll = self.string_len * np.log(1 / self.num_chars)

    def evaluate(self, x: list[int]) -> float:
        """Implements method from `SupportsEvaluate` protocol."""
        return self.ll

    def sample(self) -> tuple[list[int], float]:
        """Implements method from `SupportsSample` protocol."""
        seq = np.random.choice(range(self.num_chars), size=self.string_len).tolist()
        return seq, self.ll