"""In this example, we perform encrypted steganography using FIMEC.
Doing encrypted steganography in this way is "perfectly secure" in the sense
that the stegotext and covertext are statistically indistinguishable.
"""

import re
import secrets
from tomato.utils.random_string import RandomString
from tomato.utils.model_marginal import ModelMarginal
from mec import FIMEC
import numpy as np

class Encoder():
    def __init__(self, cipher_len = 15, shared_private_key=None,prompt="Good evening.",max_len=100,temperature=1.0,k=50, model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit") -> None:

        # For encrypted steganography, the sender and receiver share a private key.
        if shared_private_key == None:
            shared_private_key = secrets.token_bytes(cipher_len)

        self._cipher_len = cipher_len
        self._shared_private_key = shared_private_key
        self._prompt = prompt
        self._max_len = max_len
        self._temperature = temperature
        self._model_name = model_name
        self._k = k

        # The covertext distribution is a distribution over innocuous content.
        # Use a better language model for real applications.
        covertext_dist = ModelMarginal(
            prompt=self._prompt,
            max_len=self._max_len,
            temperature=self._temperature,
            k=self._k,
            model_name=self._model_name
        )

        ciphertext_dist = RandomString(num_chars=2**8, string_len=self._cipher_len)
        # FIMEC defines the communication protocol between the sender and receiver.
        self._imec = FIMEC(ciphertext_dist, covertext_dist)


    def encode(self, plaintext = "Attack at dawn!"):
        # The plaintext is the message the sender wants to communicate.

        # This is a representation of the plaintext as a sequence of bytes.
        bytetext = plaintext.encode("utf-8")

        if len(bytetext) < self._cipher_len:
            # Pad the bytetext with 'A's (which is 65 in ASCII) until it reaches the desired length.
            bytetext += b'A' * (self._cipher_len - len(bytetext))

        if len(bytetext) != self._cipher_len:
            raise Exception("The length of the bytetext representation of the plaintext should less or equal to the cipher length provided when creating the class.")

        # The ciphertext is the plaintext encrypted with the shared private key.
        # It is always distributed uniformly, since the private key is random.
        ciphertext = [a ^ b for a, b in zip(bytetext, self._shared_private_key)]
        
        # The stegotext is some innocuous content with the ciphertext hidden inside.
        # The sender communicates the stegotext to the receiver over a public channel.
        stegotext, _ = self._imec.sample_y_given_x(ciphertext)

        formatted_stegotext = re.sub(" {2,}", "\n", covertext_dist.decode(stegotext).replace("\n", " ")).strip()

        return formatted_stegotext, stegotext

    def decode(self, stegotext):
        # The estimated ciphertext is the receiver's "best guess" of the ciphertext,
        # given the stegotext.
        estimated_ciphertext, _ = self._imec.estimate_x_given_y(stegotext)

        # The estimated bytetext is a decryption of the estimated ciphertext using the
        # shared private key.
        estimated_bytetext = bytes(
            [a ^ b for a, b in zip(estimated_ciphertext, self._shared_private_key)]
        )

        # The estimated plaintext is the estimated bytetext decoded as a string.
        try:
            estimated_plaintext = estimated_bytetext.decode("utf-8")
        except UnicodeDecodeError:
            estimated_plaintext = "Estimated bytetext is not valid UTF-8."
        
        return estimated_plaintext, estimated_bytetext
