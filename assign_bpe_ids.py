#!/usr/bin/env python3
import argparse
import os

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        default="Salesforce/wikitext",
    )
    ap.add_argument(
        "--dataset_config",
        default="wikitext-103-raw-v1",
    )
    ap.add_argument(
        "--split",
        default="train",
    )
    ap.add_argument(
        "--text_col",
        default="text",
    )

    ap.add_argument(
        "--vocab_file",
        default="/vqword/bpe_wikitext103_50257/vocab.json",
    )
    ap.add_argument(
        "--merges_file",
        default="/vqword/bpe_wikitext103_50257/merges.txt",
    )
    ap.add_argument(
        "--tokenizer_out",
        default="/vqword/bpe_wikitext103_50257_hf",
    )

    # 0以下なら全件使用
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    if not os.path.isfile(args.vocab_file):
        raise FileNotFoundError(args.vocab_file)

    if not os.path.isfile(args.merges_file):
        raise FileNotFoundError(args.merges_file)

    os.makedirs(args.tokenizer_out, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer(
        args.vocab_file,
        args.merges_file,
    )

    pad_id = tokenizer.token_to_id("<pad>")
    unk_id = tokenizer.token_to_id("<unk>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()

    print("[test encode]", tokenizer.encode("Once upon a time").ids[:20])
    print("[vocab_size]", vocab_size)
    print("[pad_token_id]", pad_id)
    print("[unk_token_id]", unk_id)
    print("[bos_token_id]", bos_id)
    print("[eos_token_id]", eos_id)

    tokenizer.save_model(args.tokenizer_out)

    if args.dataset_config:
        ds = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
        )
    else:
        ds = load_dataset(
            args.dataset,
            split=args.split,
        )

    if args.max_samples > 0:
        n_samples = min(args.max_samples, len(ds))
        ds = ds.select(range(n_samples))

    print("[dataset]", args.dataset)
    print("[dataset_config]", args.dataset_config)
    print("[split]", args.split)
    print("[num_rows]", len(ds))

    samples = []
    token_chunks = []
    offsets = []

    current_offset = 0
    kept_samples = 0

    for dataset_idx, ex in enumerate(tqdm(ds, desc="[tokenize]")):
        text = ex[args.text_col]

        if not isinstance(text, str):
            continue

        # WikiTextには空行が多い
        if not text.strip():
            continue

        ids_list = tokenizer.encode(text).ids

        if len(ids_list) < 4:
            continue

        ids = torch.tensor(ids_list, dtype=torch.long)

        start = current_offset
        end = start + ids.numel()

        samples.append({
            "sample_idx": dataset_idx,
            "token_ids": ids,
            # Step 2ではVQWord未作成なので仮にBPE IDsを入れる
            "vqword_ids": ids.clone(),
            "length": ids.numel(),
        })

        token_chunks.append(ids)

        offsets.append({
            "sample_idx": dataset_idx,
            "start": start,
            "end": end,
            "length": ids.numel(),
        })

        current_offset = end
        kept_samples += 1

    if not token_chunks:
        raise ValueError(
            "No tokenized samples. "
            "Check dataset, text column and tokenizer files."
        )

    token_ids_flat = torch.cat(token_chunks, dim=0)

    payload = {
        "samples": samples,
        "token_ids_flat": token_ids_flat,

        # Step 2時点ではVQ IDsはまだ存在しないため仮コピー
        "vq_ids_flat": token_ids_flat.clone(),

        "offsets": offsets,

        "word2id": None,
        "id2word": None,

        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,

        "vocab_type": "bytelevel_bpe:wikitext103_50257",
        "hop": None,
        "ckpt": None,

        "tokenizer": args.tokenizer_out,

        # 既存ar.pyとの互換性維持
        "vq_vocab_size": vocab_size,
        "vq_pad_id": pad_id,

        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,

        "num_samples": kept_samples,
        "num_tokens": token_ids_flat.numel(),
        "token_vocab_size": vocab_size,
    }

    torch.save(payload, args.out)

    print("[save]", args.out)
    print("[kept samples]", kept_samples)
    print("[total tokens]", token_ids_flat.numel())
    print("[file size bytes]", os.path.getsize(args.out))


if __name__ == "__main__":
    main()