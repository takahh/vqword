#!/usr/bin/env python3

import torch
import pandas as pd
import matplotlib.pyplot as plt

ckpt = torch.load(
    "/Users/taka/Downloads/ar_bpe_vqw2bpe_finetune_32k_bpe_self03_20260703_185923.pt",
    map_location="cpu",
    weights_only=False,
)

hist = ckpt["history"]

df = pd.DataFrame(hist)

print(df)

# -----------------------
# Loss
# -----------------------
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["valid_loss"], label="Valid")
plt.plot(df["epoch"], df["test_loss"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Main Loss")
plt.title("Main Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------
# Main PPL
# -----------------------
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["valid_ppl"], label="Valid")
plt.plot(df["epoch"], df["test_ppl"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Main PPL")
plt.title("Main PPL")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------
# Token PPL
# -----------------------
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["valid_tok_ppl"], label="Valid")
plt.plot(df["epoch"], df["test_tok_ppl"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Token PPL")
plt.title("BPE Token PPL")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------
# VQ PPL
# -----------------------
plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["valid_vq_ppl"], label="Valid")
plt.plot(df["epoch"], df["test_vq_ppl"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("VQ PPL")
plt.title("VQWord PPL")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()