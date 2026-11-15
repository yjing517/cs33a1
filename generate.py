import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from preprocess_data import DEFAULT_TOKENIZER, END_OF_TEXT, load_tokenizer
from train_lm import create_model


@torch.inference_mode()
def generate(model, prompt, eos_id, context_length, max_new_tokens, temperature, top_p):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1 or max_new_tokens < 0:
        raise ValueError("Expected 0 < top_p <= 1 and max_new_tokens >= 0")
    ids = list(prompt) or [eos_id]
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-context_length:]], device=device)
        logits = model(x)[0, -1].float()
        if temperature == 0:
            token = logits.argmax().item()
        else:
            scores, indices = logits.sort(descending=True)
            probabilities = ((scores - scores[0]) / temperature).softmax(dim=-1)
            if top_p < 1:
                cumulative = probabilities.cumsum(dim=-1)
                remove = torch.cat((torch.zeros_like(cumulative[:1], dtype=torch.bool), cumulative[:-1] >= top_p))
                probabilities.masked_fill_(remove, 0)
            token = indices[torch.multinomial(probabilities, 1)].item()
        if token == eos_id:
            break
        ids.append(token)
        generated.append(token)
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = args.config or args.checkpoint.with_name("config.json")
    with config.open(encoding="utf-8") as source:
        model_args = SimpleNamespace(**json.load(source))
    model_args.device = args.device
    tokenizer = load_tokenizer(args.tokenizer)
    if set(tokenizer.vocab) != set(range(model_args.vocab_size)):
        parser.error("Tokenizer IDs do not match the model vocabulary size")

    model = create_model(model_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = generate(
        model, tokenizer.encode(args.prompt), tokenizer.byte_to_token_id[END_OF_TEXT.encode()],
        model_args.context_length, args.max_new_tokens, args.temperature, args.top_p,
    )
    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
