import re
import secrets
from tomato.utils.random_string import RandomString
from tomato.utils.model_marginal import ModelMarginal
from mec import FIMEC
import numpy as np
from typing import Tuple, Optional

class Encoder:
    """
    This class implements encrypted steganography using FIMEC (a mechanism
    for generating covertext that is statistically indistinguishable from
    innocuous content).
    """

    def __init__(
        self,
        cipher_len: int = 15,
        shared_private_key: Optional[bytes] = None,
        prompt: str = "Good evening.",
        max_len: int = 100,
        temperature: float = 1.0,
        k: int = 50,
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    ) -> None:
        """
        Initializes the Encoder with the necessary parameters for encrypted steganography.
        
        Args:
            cipher_len (int): Length of the cipher in bytes. Default is 15.
            shared_private_key (bytes, optional): Shared private key for encryption. 
                If None, a random key is generated. Default is None.
            prompt (str): Prompt for the covertext model. Default is "Good evening."
            max_len (int): Maximum length of the covertext. Default is 100.
            temperature (float): Sampling temperature for the covertext model. Default is 1.0.
            k (int): The top-k sampling parameter for the covertext model. Default is 50.
            model_name (str): Name of the language model used for generating covertext. 
                Default is "unsloth/mistral-7b-instruct-v0.3-bnb-4bit".
        """

        # For encrypted steganography, the sender and receiver share a private key.
        self._cipher_len = cipher_len
        self._shared_private_key = shared_private_key or secrets.token_bytes(cipher_len)
        self._prompt = prompt
        self._max_len = max_len
        self._temperature = temperature
        self._model_name = model_name
        self._k = k

        # The covertext distribution is a distribution over innocuous content.
        self._covertext_dist = ModelMarginal(
            prompt=self._prompt,
            max_len=self._max_len,
            temperature=self._temperature,
            k=self._k,
            model_name=self._model_name
        )

        # Ciphertext distribution (uniform random string)
        ciphertext_dist = RandomString(num_chars=2**8, string_len=self._cipher_len)
        
        # FIMEC defines the communication protocol between the sender and receiver.
        self._imec = FIMEC(ciphertext_dist, self._covertext_dist)

    def encode(self, plaintext: str = "Attack at dawn!") -> Tuple[str, np.ndarray]:
        """
        Encodes the plaintext into stegotext using encrypted steganography.
        
        Args:
            plaintext (str): The message to encode. Default is "Attack at dawn!".
        
        Returns:
            Tuple[str, np.ndarray]: The formatted stegotext and the original stegotext.
        """
        # Convert plaintext to a sequence of bytes.
        bytetext = plaintext.encode("utf-8")

        # Pad bytetext if necessary to match the cipher length.
        if len(bytetext) < self._cipher_len:
            bytetext += b'A' * (self._cipher_len - len(bytetext))

        if len(bytetext) != self._cipher_len:
            raise ValueError("The length of the bytetext representation of the plaintext should be less than or equal to the cipher length provided.")

        # Encrypt the plaintext with the shared private key to generate ciphertext.
        ciphertext = [a ^ b for a, b in zip(bytetext, self._shared_private_key)]

        # Generate stegotext with the ciphertext hidden inside.
        stegotext, _ = self._imec.sample_y_given_x(ciphertext)

        # Format the stegotext by replacing multiple spaces with newlines.
        formatted_stegotext = re.sub(" {2,}", "\n", self._covertext_dist.decode(stegotext).replace("\n", " ")).strip()

        return formatted_stegotext, stegotext

    def decode(self, stegotext: np.ndarray) -> Tuple[str, bytes]:
        """
        Decodes the stegotext back into plaintext.
        
        Args:
            stegotext (np.ndarray): The stegotext to decode.
        
        Returns:
            Tuple[str, bytes]: The estimated plaintext and its byte representation.
        """
        # Estimate the ciphertext from the stegotext.
        estimated_ciphertext, _ = self._imec.estimate_x_given_y(stegotext)

        # Decrypt the estimated ciphertext to retrieve the original bytetext.
        estimated_bytetext = bytes(
            [a ^ b for a, b in zip(estimated_ciphertext, self._shared_private_key)]
        )

        # Decode the bytetext back into a string.
        try:
            estimated_plaintext = estimated_bytetext.decode("utf-8")
        except UnicodeDecodeError:
            estimated_plaintext = "Estimated bytetext is not valid UTF-8."

        return estimated_plaintext, estimated_bytetext
