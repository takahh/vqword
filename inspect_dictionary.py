#!/usr/bin/env python3
import torch
import argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default="/Users/taka/Downloads/tinystories_from_wikitext103_bpe_vqword_bpe_self03_pertok_f03_dictionary.pt")
args = ap.parse_args()

d = torch.load(args.dict, map_location="cpu")

sizes = [len(words) for words in d.values()]

print("===== Dictionary Statistics =====")
print("VQ entries         :", len(sizes))
print("Average candidates :", sum(sizes) / len(sizes))
print("Max candidates     :", max(sizes))

cnt = Counter(sizes)

print("\nDistribution")
for k in sorted(cnt):
    print(f"{k:3d} candidates : {cnt[k]}")

print("\nTop ambiguous clusters")

items = sorted(
    d.items(),
    key=lambda kv: len(kv[1]),
    reverse=True,
)

for cid, words in items[:20]:
    print(f"\nCluster {cid} ({len(words)} candidates)")
    for wid, word, count in words[:10]:
        print(f"  {count:6d}  {word}")