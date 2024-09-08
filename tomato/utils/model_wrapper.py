import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tomato.utils.easy_llm import EasyLLM
from mec.utilities import log

os.environ["HF_ENDPOINT"] = "https://huggingface.co"

class ModelWrapper:
    """
    A wrapper class for a language model that provides methods for computing 
    conditional distributions and token ID reductions based on top-k filtering.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the ModelWrapper with a given model name.

        Args:
            model_name (str): The name of the pretrained model to use.

        Attributes:
            tokenizer (AutoTokenizer): Tokenizer for the model.
            model (AutoModelForCausalLM): The language model.
            vocab_size (int): The size of the model's vocabulary.
            vocab (np.ndarray): Decoded vocabulary tokens.
        """
        llm = EasyLLM(model_name=model_name, max_new_tokens=100)

        self.tokenizer = llm.tokenizer
        self.model = llm.model
        self.vocab_size = self.model.config.vocab_size
        self.vocab = np.array([self.tokenizer.decode([i]) for i in range(self.vocab_size)])

    def conditional(self, text: str, temperature: float) -> torch.Tensor:
        """
        Computes the conditional distribution given the input text.

        Args:
            text (str): The input text to condition on.
            temperature (float): Temperature for sampling.

        Returns:
            torch.Tensor: The conditional distribution over the vocabulary.
        """
        with torch.no_grad():
            encoded_input = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(encoded_input["input_ids"])
            return nn.Softmax(dim=-1)(outputs[0][:, -1, :] / temperature)

    def top_k_conditional(self, text: str, temperature: float, k: int) -> np.ndarray:
        """
        Computes the top-k conditional distribution given the input text.

        Args:
            text (str): The input text to condition on.
            temperature (float): Temperature for sampling.
            k (int): The number of top elements to consider.

        Returns:
            np.ndarray: The top-k conditional distribution.
        """
        conditional = self.conditional(text, temperature)
        kth_value = torch.topk(conditional, k).values.flatten()[-1]
        conditional[conditional < kth_value] = 0
        conditional /= conditional.sum()
        return conditional.numpy().reshape(-1)

    def reduced_ids(self, prompt: str, text: str, k: int) -> Optional[List[int]]:
        """
        Computes token IDs using indexing post top-k reduction.

        Args:
            prompt (str): The prompt to condition on.
            text (str): The text to encode.
            k (int): The number of top elements to consider.

        Returns:
            Optional[List[int]]: The list of token IDs, or None if encoding fails.
        """
        prompt_tokens = self.tokenizer(prompt)["input_ids"]
        text_tokens = self.tokenizer(text)["input_ids"]
        reduced_tokens: List[int] = []

        for i, token in enumerate(text_tokens):
            inputs = torch.tensor([prompt_tokens + text_tokens[:i]])
            outputs = self.model(inputs)[0][0, -1, :]
            top_k_idx = outputs >= torch.topk(outputs, k).values.flatten()[-1]
            if top_k_idx[token]:
                correct_idx = (torch.arange(outputs.shape[-1])[top_k_idx] == token).nonzero(as_tuple=True)[0]
                assert correct_idx.numel() == 1
                reduced_tokens.append(correct_idx.item())
            else:
                return None

        return reduced_tokens
