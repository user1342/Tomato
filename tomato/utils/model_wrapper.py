import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from .easy_llm import EasyLLM
from mec.utilities import log

os.environ["HF_ENDPOINT"] = "https://huggingface.co"

class ModelWrapper:
    def __init__(self, model_name):
        """Initialize model and tokenizer.

        Attributes:
            tokenizer: Tokenizer for model.
            model: model.
            vocab_size: Size of the vocabulary.
            vocab: Vocabulary of the model.
        """

        llm = EasyLLM(model_name=model_name,max_new_tokens=100)

        self.tokenizer = llm.tokenizer
        self.model = llm.model
        self.vocab_size = self.model.config.vocab_size
        self.vocab = np.array(
            [self.tokenizer.decode([i]) for i in range(self.vocab_size)]
        )

    def conditional(self, text: str, temperature: float) -> torch.Tensor:
        """Compute conditional distribution given text.

        Args:
            text: Text to condition on.
            temperature: Temperature for sampling.

        Returns:
            Conditional distribution.
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text,
                return_tensors="pt",
            )
            outputs = self.model(encoded_input["input_ids"])
            return nn.Softmax(dim=-1)(outputs[0][:, -1, :] / temperature)

    def top_k_conditional(self, text: str, temperature: float, k: int) -> np.ndarray:
        """Compute top-k conditional distribution given text.

        Args:
            text: Text to condition on.
            temperature: Temperature for sampling.
            k: Number of top elements to consider.

        Returns:
            Top-k conditional distribution.
        """
        conditional = self.conditional(text, temperature)
        kth = torch.topk(conditional, k).values.flatten()[-1]
        conditional[conditional < kth] = 0
        conditional /= conditional.sum()
        return conditional.numpy().reshape(-1)

    def reduced_ids(self, prompt: str, text: str, k: int) -> list[int] | None:
        """Token IDs using indexing post top-k reduction.

        Args:
            prompt: Prompt to condition on.
            text: Text to encode.
            k: Number of top elements to consider.

        Returns:
            Token IDs, None if encoding fails.
        """
        prompt_tokens = self.tokenizer(prompt)["input_ids"]
        text_tokens = self.tokenizer(text)["input_ids"]
        reduced_tokens = []
        for i, token in enumerate(text_tokens):
            inputs = torch.tensor([prompt_tokens + text_tokens[:i]])
            outputs = self.model(inputs)[0][0, -1, :]
            top_k_idx = outputs >= torch.topk(outputs, k).values.flatten()[-1]
            if top_k_idx[token]:
                correct_idx = torch.arange(outputs.shape[-1])[top_k_idx] == token
                assert correct_idx.sum() == 1
                reduced_tokens.append(correct_idx.nonzero(as_tuple=True)[0].item())
            else:
                return None
        return reduced_tokens