#!/usr/bin/env python3
import argparse, random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--tokenizer", required=True)   # e.g. bpe_32768
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--out", default="tinystories_bpe_ids.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(args.dataset, split="train")

    samples = []
    all_vq = []

    for i, ex in enumerate(tqdm(ds, desc="[tokenize]")):
        if i >= args.max_samples:
            break

        text = ex[args.text_col]
        ids = tok.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.seq_len,
        )

        if len(ids) < 4:
            continue

        token_ids = torch.tensor(ids, dtype=torch.long)

        # dummy VQ IDs, because ar.py dataset expects vqword_ids
        vqword_ids = torch.zeros_like(token_ids)

        samples.append({
            "token_ids": token_ids,
            "vqword_ids": vqword_ids,
        })
        all_vq.append(vqword_ids)

    vq_ids_flat = torch.cat(all_vq)

    torch.save({
        "samples": samples,
        "tokenizer": args.tokenizer,
        "vq_ids_flat": vq_ids_flat,
        "args": vars(args),
    }, args.out)

    print("saved:", args.out)
    print("samples:", len(samples))
    print("tokenizer vocab:", tok.vocab_size)

if __name__ == "__main__":
    main()