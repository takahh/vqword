#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_vis", type=int, default=50000)
    ap.add_argument("--n_neighbors", type=int, default=50)
    ap.add_argument("--min_dist", type=float, default=0.05)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    input_path = Path(args.input)

    if args.out is None:
        out_path = input_path.with_name(
            f"{input_path.stem}_umap.png"
        )
    else:
        out_path = Path(args.out)

    d = torch.load(input_path, map_location="cpu")

    z = d["z"].float().numpy()
    centers = d["centers"].float().numpy()
    cluster = d["cluster"].cpu().numpy()

    n_total = len(z)
    n_vis = min(args.n_vis, n_total)

    rng = np.random.RandomState(args.seed)

    if n_vis < n_total:
        idx = rng.choice(
            n_total,
            n_vis,
            replace=False,
        )
    else:
        idx = np.arange(n_total)

    z_vis = z[idx]
    cluster_vis = cluster[idx]

    X = np.concatenate(
        [z_vis, centers],
        axis=0,
    )

    print(
        f"[UMAP] wid={d.get('wid')} "
        f"total={n_total:,} "
        f"sampled={n_vis:,} "
        f"centers={len(centers)}"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(
            args.n_neighbors,
            max(2, len(X) - 1),
        ),
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        low_memory=True,
        verbose=True,
    )

    xy = reducer.fit_transform(X)

    z_xy = xy[:n_vis]
    c_xy = xy[n_vis:]

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
        plt.text(
            x,
            y,
            f"C{i}",
            fontsize=12,
        )

    plt.title(
        f"token wid={d.get('wid')} "
        f"UMAP embeddings and centers"
    )
    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
