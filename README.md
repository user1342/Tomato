<p align="center">
    <img width=100% src="tomato.png">
  </a>
</p>
<p align="center"> 🤖 LLM steganography with minimum-entropy coupling 🍅 </p>

**Tomato is a proof of concept steganography tool that utilises minimum-entropy coupling code provided by [ssokota](https://github.com/ssokota/mec/tree/master)! ⭐**

# 🧠 How It Works: Minimum Entropy Coupling and Language Models
To understand how the Tomato Encoder/Decoder Tool works, we need to dive a bit deeper into two key concepts: Minimum Entropy Coupling (MEC) and how it relates to Large Language Models (LLMs).

## Minimum Entropy Coupling (MEC)
At the core of this tool is a mathematical technique called Minimum Entropy Coupling. Let’s break down what this means and why it’s crucial.

## What is Entropy in This Context?
In information theory, entropy is a measure of uncertainty or randomness. For example, a perfectly balanced coin flip has high entropy because there's equal uncertainty about whether it will land on heads or tails. When we talk about coupling two distributions (like two different sets of possible outcomes), we’re often interested in how much uncertainty (entropy) is present when these two distributions are combined.

## Coupling Distributions
Coupling refers to the process of linking two different probability distributions together in a way that they remain true to their original forms but are now interdependent. Imagine you have two bags, one filled with different colored marbles (representing distribution X) and another with different shaped objects (representing distribution Y). Coupling would be like finding a way to always pair a specific marble color with a specific shape in such a way that respects the original randomness of each bag.

## Minimum Entropy Coupling
Now, Minimum Entropy Coupling (MEC) takes this a step further. MEC tries to pair the elements from two distributions (like our marbles and shapes) in a way that minimizes the overall randomness or uncertainty in their combined distribution. This is not just about pairing them in any random way but finding the pairings that make the joint distribution as "orderly" or predictable as possible, without losing the original characteristics of each distribution.

Why does this matter? In the context of steganography and encoding, reducing entropy while maintaining a natural appearance in the encoded message (stegotext) ensures that the hidden message is both secure and indistinguishable from normal content.

## Large Language Models (LLMs)
Large Language Models (LLMs) like GPT-2 or the "mistral-7b-instruct-v0.3-bnb-4bit" used in this tool are incredibly advanced neural networks trained on vast amounts of text data. They can generate human-like text, predict the next word in a sentence, or even answer questions in a coherent manner.

## How LLMs Are Used in This Tool
In this tool, LLMs serve a dual purpose:
```
Generating Covertext: When encoding a message, the tool uses an LLM to generate natural-sounding text that will serve as the covertext. This ensures that the stegotext appears indistinguishable from any regular text that the model might generate, making it very hard for anyone to suspect that it contains hidden information.
```
```
Decoding Stegotext: The LLM is also used in the decoding process. By understanding the patterns and structures it was trained on, the model helps in accurately extracting the hidden message from the stegotext.
```
## The Role of MEC in LLMs
When LLMs generate text, they do so by sampling from a probability distribution of possible next words or tokens. MEC plays a crucial role in this process:

During Encoding: MEC is used to couple the distribution of the hidden message (ciphertext) with the distribution of possible covertext generated by the LLM. The goal is to ensure that the final stegotext is not only meaningful but also has the least possible entropy, making it look perfectly normal while still embedding the secret message.

During Decoding: The same principles are applied in reverse. The tool decodes the stegotext by leveraging the same MEC principles to disentangle the hidden message from the natural-sounding covertext, ensuring that the message can be accurately retrieved.


# ⚙️ Setup
Tomato required Nvidia CUDA. Follow the steps below:
- Ensure your Nvidia drivers are up to date: https://www.nvidia.com/en-us/geforce/drivers/
- Install the appropriate dependancies from here: https://pytorch.org/get-started/locally/
- Validate CUDA is installed correctly by running the following and being returned a prompt ```python -c "import torch; print(torch.rand(2,3).cuda())"```
  
Install the dependencies using:

```bash
pip install git+https://github.com/user1342/mec
```
```bash
git clone https://github.com/user1342/Tomato.git
cd tomato
pip install -r requirements.txt
pip install -e .
```

# 🏃 Running
You can use the Tomato Encoder/Decoder Tool directly from the command line. Here are the available commands:

## Encode a Message
To encode a plaintext message into stegotext:

```bash
tomato-encode.exe "Your secret message here" --cipher_len 20 --shared_private_key 123abc... --prompt "Good evening."
```

Example:
```bash
tomato-encode.exe "Your plaintext here" --cipher_len 15 --shared_private_key 123abc... --max_len 100 --temperature 1.0 --k 50 --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

This will output something like:
```
Stegotext: [Your encoded message here]
```

## Decode a Message
To decode a stegotext back into its original plaintext:

```bash
tomato-decode.exe "Your stegotext here" --cipher_len 20 --shared_private_key 123abc... --prompt "Good evening."
```

Example:
```bash
tomato-decode.exe "Your stegotext here" --cipher_len 15 --shared_private_key 123abc... --max_len 100 --temperature 1.0 --k 50 --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

This will output something like:

```
Estimated Plaintext: [Your decoded plaintext]
Estimated Bytetext: [Your decoded bytetext]
```

## Programatic Example
Checkout the [example playbook](https://github.com/user1342/Tomato/blob/main/example.ipynb)! For a quick demonstration, you can try encoding and decoding a simple message using the following code snippet:

```python
from tomato import Encoder

encoder = Encoder()

plaintext = "I am a hidden code"
formatted_stegotext, stegotext = encoder.encode(plaintext)
estimated_plaintext, estimated_bytetext = encoder.decode(stegotext)

print(formatted_stegotext)
print("------")
print(estimated_plaintext)
```

# 🛡️ Customization Options
The Tomato Encoder/Decoder Tool offers several customizable parameters:

* cipher_len: Length of the cipher (default is 15).
* shared_private_key: Shared private key in hex format. If not provided, a random key will be generated.
* prompt: Prompt for the language model (default is "Good evening.").
* max_len: Maximum length of the covertext (default is 100).
* temperature: Sampling temperature for the language model (default is 1.0).
* k: The k parameter for the language model (default is 50).
* model_name: Name of the language model to be used (default is "unsloth/mistral-7b-instruct-v0.3-bnb-4bit").

# 🙏 Contributions
Tomoto is an open-source project and welcomes contributions from the community. If you would like to contribute to Tomoto, please follow these guidelines:

- Fork the repository to your own GitHub account.
- Create a new branch with a descriptive name for your contribution.
- Make your changes and test them thoroughly.
- Submit a pull request to the main repository, including a detailed description of your changes and any relevant documentation.
- Wait for feedback from the maintainers and address any comments or suggestions (if any).
- Once your changes have been reviewed and approved, they will be merged into the main repository.

# ⚖️ Code of Conduct
Tomoto follows the Contributor Covenant Code of Conduct. Please make sure to review and adhere to this code of conduct when contributing to Monocle.

# 🐛 Bug Reports and Feature Requests
If you encounter a bug or have a suggestion for a new feature, please open an issue in the GitHub repository. Please provide as much detail as possible, including steps to reproduce the issue or a clear description of the proposed feature. Your feedback is valuable and will help improve Monocle for everyone.

# 📜 License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
