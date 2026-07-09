#!/usr/bin/env python3
import argparse
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--vocab_file", default="bpe_wikitext103_50k/vocab.json")
    ap.add_argument("--merges_file", default="bpe_wikitext103_50k/merges.txt")
    ap.add_argument("--tokenizer_out", default="bpe_wikitext103_50k_hf")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.tokenizer_out, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer(
        args.vocab_file,
        args.merges_file,
    )

    pad_id = tokenizer.token_to_id("<pad>")
    unk_id = tokenizer.token_to_id("<unk>")
    vocab_size = tokenizer.get_vocab_size()

    print("[test encode]", tokenizer.encode("Once upon a time").ids[:20])
    print("[vocab_size]", vocab_size)
    print("[pad_token_id]", pad_id)
    print("[unk_token_id]", unk_id)

    tokenizer.save_model(args.tokenizer_out)

    ds = load_dataset(args.dataset, split=args.split)

    samples = []
    token_ids_flat = []
    offsets = []

    for i, ex in enumerate(tqdm(ds.select(range(min(args.max_samples, len(ds)))))):
        ids = tokenizer.encode(ex[args.text_col]).ids

        if len(ids) < 4:
            continue

        ids = torch.tensor(ids, dtype=torch.long)

        start = sum(len(x) for x in token_ids_flat)
        end = start + len(ids)

        samples.append({
            "sample_idx": i,
            "token_ids": ids,
            "vqword_ids": ids.clone(),
            "length": len(ids),
        })

        token_ids_flat.append(ids)
        offsets.append((i, start, end, len(ids)))

    if len(token_ids_flat) == 0:
        raise ValueError("No tokenized samples. tokenizer.encode() returned empty ids.")

    token_ids_flat = torch.cat(token_ids_flat, dim=0)

    torch.save({
        "samples": samples,
        "token_ids_flat": token_ids_flat,
        "vq_ids_flat": token_ids_flat.clone(),
        "offsets": offsets,
        "word2id": None,
        "id2word": None,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "vocab_type": f"bytelevel_bpe:{args.tokenizer_out}",
        "hop": None,
        "ckpt": None,
        "tokenizer": args.tokenizer_out,
        "vq_vocab_size": vocab_size,
        "vq_pad_id": pad_id,
    }, args.out)

    print("[save]", args.out)


if __name__ == "__main__":
    main()