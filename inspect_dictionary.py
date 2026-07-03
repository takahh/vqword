#!/usr/bin/env python3
import torch
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default="/Users/taka/Downloads/wikitext103_bpe_vqword_32k_bpe_self03_dictionary.pt")
ap.add_argument("--top_words", type=int, default=20)
ap.add_argument("--top_clusters", type=int, default=100)
args = ap.parse_args()

d = torch.load(args.dict, map_location="cpu")

# クラスタサイズ順
items = sorted(
    d.items(),
    key=lambda kv: sum(c for _, _, c in kv[1]),
    reverse=True,
)

for cid, words in items[:args.top_clusters]:
    total = sum(c for _, _, c in words)
    print(f"\n=== Cluster {cid} total={total} ===")

    for wid, word, count in words[:args.top_words]:
        print(f"{count:6d} {word}")