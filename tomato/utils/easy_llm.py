import re
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class EasyLLM:
    """
    A simple class for interacting with a pretrained language model to generate dialogue responses.
    """

    def __init__(
        self,
        max_new_tokens: int = 1000,
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    ) -> None:
        """
        Initializes the EasyLLM class with a specified model and token generation limit.

        Args:
            max_new_tokens (int): Maximum number of new tokens to generate in a response.
            model_name (str): Name of the pretrained language model to use.
        """
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.dialogue: List[dict] = []

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model(self.model_name)

    def set_to_eval(self) -> None:
        """
        Sets the model to evaluation mode, which is necessary for generating responses.
        """
        self.model.eval()

    def _load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the pretrained language model and tokenizer.

        Args:
            model_name (str): Name of the pretrained model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded language model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            temperature=1.0,
            do_sample=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        return model, tokenizer

    def _generate_dialogue_response(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        messages: List[dict],
    ) -> str:
        """
        Generates a response from the language model based on the input messages.

        Args:
            model (AutoModelForCausalLM): Loaded language model.
            tokenizer (AutoTokenizer): Loaded tokenizer.
            device (str): Device to run the model on, e.g., "cuda" or "cpu".
            messages (List[dict]): List of input messages.

        Returns:
            str: Generated response from the language model.
        """
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=50256,
        )
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return decoded[0]

    def _remove_inst_tags(self, text: str) -> str:
        """
        Removes instruction tags and other unwanted tokens from the generated text.

        Args:
            text (str): Input text containing instruction tags.

        Returns:
            str: Cleaned text with instruction tags removed.
        """
        cleaned_text = re.sub(r"^.*\[\\INST\](?!.*\[\\INST\])", "", text)

        pattern = r"\[INST\].*?\[/INST\]"
        clean_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL)
        clean_text = re.sub(
            r"<\|user\|>.*?<\|end\|>.*?<\|endoftext\|>", "", clean_text, flags=re.DOTALL
        )

        return (
            clean_text.replace("<s>", "")
            .replace("</s>", "")
            .replace("Explanation:", "")
            .strip()
        )

    def reset_dialogue(self) -> None:
        """
        Resets the dialogue history, clearing all previous messages.
        """
        self.dialogue = []

    def ask_question(self, question: str, reset_dialogue: bool = False) -> str:
        """
        Generates a response for the given question using the loaded model.

        Args:
            question (str): The question or prompt provided by the user.
            reset_dialogue (bool): Whether to reset the dialogue history after generating a response.

        Returns:
            str: Generated response to the question.
        """
        self.dialogue.append({"role": "user", "content": question})

        result = self._generate_dialogue_response(
            self.model, self.tokenizer, self._device, self.dialogue
        )

        # Extract the response after the last "[/INST]" tag and clean it
        data = result.split("[/INST]")
        result = data[-1]
        result = self._remove_inst_tags(result)

        self.dialogue.append({"role": "assistant", "content": result})

        if reset_dialogue:
            self.dialogue = []

        return result
