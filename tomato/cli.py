import argparse
from tomato.encoder import Encoder  # Import Encoder class from the right module

def encode_command(args):
    encoder = Encoder(
        cipher_len=args.cipher_len,
        shared_private_key=args.shared_private_key,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        k=args.k,
        model_name=args.model_name
    )
    stegotext = encoder.encode(args.plaintext)
    print(f"Stegotext: {stegotext}")

def decode_command(args):
    encoder = Encoder(
        cipher_len=args.cipher_len,
        shared_private_key=args.shared_private_key,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        k=args.k,
        model_name=args.model_name
    )
    estimated_plaintext, estimated_bytetext = encoder.decode(args.stegotext)
    print(f"Estimated Plaintext: {estimated_plaintext}")
    print(f"Estimated Bytetext: {estimated_bytetext}")

def main():
    parser = argparse.ArgumentParser(description="Encode and decode messages using encrypted steganography.")
    subparsers = parser.add_subparsers(help="Commands: encode or decode")
    
    encode_parser = subparsers.add_parser("encode", help="Encode a message into stegotext.")
    encode_parser.add_argument("plaintext", type=str, help="The plaintext message to encode.")
    encode_parser.add_argument("--cipher_len", type=int, default=15, help="Length of the cipher.")
    encode_parser.add_argument("--shared_private_key", type=lambda x: bytes.fromhex(x) if x else None, default=None, help="Shared private key in hex format. If not provided, a random key will be generated.")
    encode_parser.add_argument("--prompt", type=str, default="Good evening.", help="Prompt for the language model.")
    encode_parser.add_argument("--max_len", type=int, default=100, help="Maximum length of the covertext.")
    encode_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for language model sampling.")
    encode_parser.add_argument("--k", type=int, default=50, help="The k parameter for the model.")
    encode_parser.add_argument("--model_name", type=str, default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", help="Model name for the language model.")
    encode_parser.set_defaults(func=encode_command)
    
    decode_parser = subparsers.add_parser("decode", help="Decode a stegotext message.")
    decode_parser.add_argument("stegotext", type=str, help="The stegotext message to decode.")
    decode_parser.add_argument("--cipher_len", type=int, default=15, help="Length of the cipher.")
    decode_parser.add_argument("--shared_private_key", type=lambda x: bytes.fromhex(x) if x else None, default=None, help="Shared private key in hex format. Must match the key used for encoding.")
    decode_parser.add_argument("--prompt", type=str, default="Good evening.", help="Prompt for the language model.")
    decode_parser.add_argument("--max_len", type=int, default=100, help="Maximum length of the covertext.")
    decode_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for language model sampling.")
    decode_parser.add_argument("--k", type=int, default=50, help="The k parameter for the model.")
    decode_parser.add_argument("--model_name", type=str, default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", help="Model name for the language model.")
    decode_parser.set_defaults(func=decode_command)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
