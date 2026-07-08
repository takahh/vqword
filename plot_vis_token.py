#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/Users/taka/Documents/vis_token_284.pt")
    ap.add_argument("--out", default="vis_token.png")
    args = ap.parse_args()

    d = torch.load(args.input, map_location="cpu")

    z = d["z"].float().numpy()
    centers = d["centers"].float().numpy()
    cluster = d["cluster"].numpy()

    X = np.concatenate([z, centers], axis=0)

    xy = PCA(n_components=2).fit_transform(X)

    z_xy = xy[:len(z)]
    c_xy = xy[len(z):]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        z_xy[:, 0],
        z_xy[:, 1],
        c=cluster,
        s=6,
        alpha=0.5,
    )

    plt.scatter(
        c_xy[:, 0],
        c_xy[:, 1],
        marker="x",
        s=200,
        linewidths=3,
        c="black",
    )

    for i, (x, y) in enumerate(c_xy):
        plt.text(x, y, f"C{i}", fontsize=12)

    plt.title(f"token wid={d['wid']} embeddings and centers")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()