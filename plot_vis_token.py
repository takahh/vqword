#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/Users/taka/Documents/vis_token_284.pt")
    ap.add_argument("--out", default="/Users/taka/Documents/vis_token_umap.png")
    ap.add_argument("--n_neighbors", type=int, default=50)
    ap.add_argument("--min_dist", type=float, default=0.05)
    ap.add_argument("--metric", default="cosine")
    args = ap.parse_args()

    d = torch.load(args.input, map_location="cpu")
    z = d["z"].float().numpy()

    centers = d["centers"].float().numpy()
    cluster = d["cluster"].numpy()

    n_vis = 50000

    idx = np.random.RandomState(0).choice(len(z), n_vis, replace=False)

    z_vis = z[idx]
    cluster_vis = cluster[idx]
    X = np.concatenate([z_vis, centers], axis=0)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=0,
    )

    xy = reducer.fit_transform(X)

    z_xy = xy[:len(z_vis)]
    c_xy = xy[len(z_vis):]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        z_xy[:, 0],
        z_xy[:, 1],
        c=cluster_vis,
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

    plt.title(f"token wid={d['wid']} UMAP embeddings and centers")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()