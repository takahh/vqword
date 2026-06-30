#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from train_vqword import VQWordGNN, make_windows
import re

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

def word_tokenize(text):
    return WORD_RE.findall(text)

@torch.no_grad()
def assign_ids(model, centroids, ctx, batch_size, device):
    model.eval()
    centroids = F.normalize(centroids.to(device), dim=-1)

    all_ids = []
    for s in tqdm(range(0, len(ctx), batch_size), desc="[assign]"):
        xb = ctx[s:s+batch_size].to(device)
        z = model.encode_context(xb)
        z = F.normalize(z, dim=-1)

        sim = z @ centroids.T
        pred = sim.argmax(dim=1)

        all_ids.append(pred.cpu())

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
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]

    pad_id = ckpt.get("pad_token_id", 0)
    unk_id = ckpt.get("unk_token_id", 1)

    vocab_size = len(word2id)
    cargs = ckpt["args"]

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=ckpt["args"]["d_model"],
        hop=ckpt["args"]["hop"],
        n_layers=ckpt["args"]["n_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    centroids = ckpt["centroids"].to(device)

    ds = load_dataset(args.dataset, split=args.split)

    all_ctx = []
    all_tgt = []
    offsets = []

    print("[data] tokenizing")
    for i, ex in enumerate(tqdm(ds.select(range(min(args.max_samples, len(ds)))))):
        text = ex[args.text_col]
        words = word_tokenize(text)
        ids = [word2id.get(w, unk_id) for w in words[:args.seq_len]]

        if len(ids) < 2 * cargs["hop"] + 2:
            continue

        ctx = torch.cat(all_ctx, dim=0)
        tgt = torch.cat(all_tgt, dim=0)

        start = sum(len(x) for x in all_tgt)
        end = start + len(tgt)

        offsets.append((i, start, end, len(ids)))
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    ctx, tgt = make_windows(ids, cargs["hop"], pad_id)

    print(f"[data] windows={len(tgt):,}")
    vq_ids = assign_ids(model, centroids, ctx, args.batch_size, device)

    # sampleごとに戻す
    samples = []
    for sample_idx, start, end, n_tok in offsets:
        samples.append({
            "sample_idx": sample_idx,
            "token_ids": tgt[start:end].cpu(),
            "vqword_ids": vq_ids[start:end].cpu(),
            "length": n_tok,
        })

    torch.save({
        "samples": samples,
        "vq_ids_flat": vq_ids,
        "token_ids_flat": tgt,
        "offsets": offsets,
        "word2id": word2id,
        "id2word": id2word,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "vocab_type": "word",
        "hop": cargs["hop"],
        "ckpt": args.ckpt,
    }, args.out)

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()