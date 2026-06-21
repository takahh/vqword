#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from train_vqword import VQWordGNN, make_windows


@torch.no_grad()
def assign_ids(model, centroids, ctx, batch_size, device):
    model.eval()
    centroids = F.normalize(centroids.to(device), dim=-1)

    all_ids = []
    for s in tqdm(range(0, len(ctx), batch_size), desc="[assign]"):
        xb = ctx[s:s+batch_size].to(device)
        z = model.encode_context(xb)
        z = F.normalize(z, dim=-1)

        # cosine nearest centroid
        sim = z @ centroids.T
        ids = sim.argmax(dim=-1)
        all_ids.append(ids.cpu())

    return torch.cat(all_ids, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="vqword_gnn_kmeans.pt")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--out", default="tiny_vqword_ids.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cargs = ckpt["args"]

    tok = AutoTokenizer.from_pretrained(ckpt["tokenizer"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = VQWordGNN(
        vocab_size=tok.vocab_size,
        d_model=cargs["d_model"],
        hop=cargs["hop"],
        n_layers=cargs["n_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    centroids = ckpt["centroids"]

    ds = load_dataset(args.dataset, split=args.split)

    rows = []
    all_ctx = []
    all_tgt = []
    offsets = []

    print("[data] tokenizing")
    for i, ex in enumerate(tqdm(ds.select(range(min(args.max_samples, len(ds)))))):
        text = ex[args.text_col]
        ids = tok.encode(text, add_special_tokens=False)[:args.seq_len]

        if len(ids) < 2 * cargs["hop"] + 2:
            continue

        ctx, tgt = make_windows(ids, cargs["hop"], ckpt["pad_token_id"])

        start = sum(len(x) for x in all_tgt)
        end = start + len(tgt)

        offsets.append((i, start, end, len(ids)))
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print(f"[data] windows={len(tgt):,}")

    vq_ids = assign_ids(model, centroids, ctx, args.batch_size, device)

    # sampleごとに戻す
    samples = []
    for sample_idx, start, end, n_tok in offsets:
        samples.append({
            "sample_idx": sample_idx,
            "token_ids": tgt[start:end],
            "vqword_ids": vq_ids[start:end],
            "length": n_tok,
        })

    torch.save({
        "samples": samples,
        "vq_ids_flat": vq_ids,
        "token_ids_flat": tgt,
        "offsets": offsets,
        "tokenizer": ckpt["tokenizer"],
        "hop": cargs["hop"],
        "ckpt": args.ckpt,
    }, args.out)

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()