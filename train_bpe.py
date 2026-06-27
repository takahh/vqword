#!/usr/bin/env python3
import argparse, os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or f"bpe_{args.vocab_size}"
    corpus_file = "bpe_corpus.txt"
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split="train")

    with open(corpus_file, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.max_samples:
                break
            text = ex[args.text_col].replace("\n", " ")
            f.write(text + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    tokenizer_json = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    hf_tok.save_pretrained(out_dir)

    print("saved:", out_dir)
    print("vocab size:", hf_tok.vocab_size)
    print("test encode:", hf_tok.encode("Once upon a time, there was a little girl."))

if __name__ == "__main__":
    main()