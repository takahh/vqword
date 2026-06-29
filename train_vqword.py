#!/usr/bin/env python3
import argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

def word_tokenize(text):
    return WORD_RE.findall(text)

def build_word_vocab(ds, text_col, max_samples, min_freq=1):
    cnt = Counter()

    for ex in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="[vocab]"):
        words = word_tokenize(ex[text_col])
        cnt.update(words)

    word2id = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.most_common():
        if c >= min_freq and w not in word2id:
            word2id[w] = len(word2id)

    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def make_adj(seq_len, hop, device):
    pos = torch.arange(seq_len, device=device)
    dist = (pos[:, None] - pos[None, :]).abs()

    adj = (dist <= hop).float()
    adj.fill_diagonal_(1.0)

    # degree normalize
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adj = adj / deg
    return adj

from sklearn.cluster import MiniBatchKMeans
import numpy as np

@torch.no_grad()
def fit_kmeans_streaming(model, ctx, batch_size, device, args):
    model.eval()

    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size,
        batch_size=args.kmeans_batch_size,
        init="random",
        n_init=1,
        max_iter=1,
        reassignment_ratio=0.0,
        random_state=0,
        verbose=1,
    )

    print("[kmeans] partial_fit streaming")

    init_buf = []
    init_n = 0
    initialized = False

    for start in tqdm(range(0, len(ctx), batch_size), desc="[kmeans fit]"):
        xb = ctx[start:start + batch_size].to(device)
        z = model.encode_context(xb).cpu().numpy().astype(np.float32)

        if not initialized:
            init_buf.append(z)
            init_n += len(z)

            if init_n < args.codebook_size:
                continue

            z0 = np.concatenate(init_buf, axis=0)
            kmeans.partial_fit(z0)

            initialized = True
            init_buf = []
        else:
            kmeans.partial_fit(z)

    if not initialized:
        raise ValueError(
            f"Not enough samples for codebook_size={args.codebook_size}. "
            f"Only got {init_n} samples."
        )

    return kmeans

@torch.no_grad()
def predict_kmeans_streaming(model, kmeans, ctx, batch_size, device):
    model.eval()
    ids = []

    print("[kmeans] predict streaming")
    for start in tqdm(range(0, len(ctx), batch_size), desc="[kmeans predict]"):
        xb = ctx[start:start + batch_size].to(device)
        z = model.encode_context(xb).cpu().numpy().astype(np.float32)
        pred = kmeans.predict(z)
        ids.append(torch.from_numpy(pred).long())

    return torch.cat(ids, dim=0)

class AdjGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.nei_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        # h: [B, L, D]
        # adj: [L, L]
        m = torch.einsum("ij,bjd->bid", adj, h)
        out = self.self_lin(h) + self.nei_lin(m)
        out = F.gelu(out)
        return self.norm(h + out)


class VQWordGNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, hop=3, n_layers=3, dropout=0.1):
        super().__init__()
        self.hop = hop
        self.seq_len = 2 * hop + 1

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        self.layers = nn.ModuleList([
            AdjGNNLayer(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

    def encode_context(self, ctx_ids):
        # ctx_ids: [B, 2hop+1]
        B, L = ctx_ids.shape
        pos = torch.arange(L, device=ctx_ids.device).unsqueeze(0).expand(B, L)

        h = self.tok_emb(ctx_ids) + self.pos_emb(pos)
        h = self.dropout(h)

        adj = make_adj(L, self.hop, ctx_ids.device)

        for layer in self.layers:
            h = layer(h, adj)

        z = h[:, self.hop]
        return F.normalize(z, dim=-1)

    def forward(self, ctx_ids, target_ids):
        z = self.encode_context(ctx_ids)
        logits = self.decoder(z)
        loss = F.cross_entropy(logits, target_ids)
        return loss, logits, z


def make_windows(token_ids, hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (hop, hop), value=pad_id)

    ctx, tgt = [], []
    for i in range(len(ids)):
        ctx.append(padded[i:i + 2 * hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt, dtype=torch.long)


@torch.no_grad()
def collect_embeddings(model, ctx, batch_size, device):
    model.eval()
    zs = []

    for start in tqdm(range(0, len(ctx), batch_size), desc="[embed]"):
        xb = ctx[start:start + batch_size].to(device)
        z = model.encode_context(xb)
        zs.append(z.cpu())

    return torch.cat(zs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--codebook_size", type=int, default=8192)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kmeans_batch_size", type=int, default=8192)
    ap.add_argument("--out", default="vqword_gnn_kmeans.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pad_id = 0
    unk_id = 1

    ds = load_dataset(args.dataset, split="train")

    word2id, id2word = build_word_vocab(
        ds=ds,
        text_col=args.text_col,
        max_samples=args.max_samples,
        min_freq=args.min_freq,
    )

    vocab_size = len(word2id)
    print(f"[word_vocab_size] {vocab_size}")

    all_ctx, all_tgt = [], []

    print("[data] tokenizing")
    for ex in tqdm(ds.select(range(min(args.max_samples, len(ds))))):
        text = ex[args.text_col]
        words = word_tokenize(text)
        ids = [word2id.get(w, unk_id) for w in words[:args.seq_len]]
        if len(ids) < 2 * args.hop + 2:
            continue

        ctx, tgt = make_windows(ids, args.hop, pad_id)
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print(f"[data] windows={len(tgt):,} vocab={tok.vocab_size}")
    model = VQWordGNN(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        hop=args.hop,
        n_layers=args.n_layers,
    ).to(device)

    # No pretraining.
    # Directly encode local token windows and run KMeans.

    # print("[kmeans] collecting embeddings")
    # z = collect_embeddings(model, ctx, args.batch_size, device)
    # z_np = z.numpy()
    #
    # print("[kmeans] fitting")
    # # kmeans = MiniBatchKMeans(
    # #     n_clusters=args.codebook_size,
    # #     batch_size=args.kmeans_batch_size,
    # #     random_state=0,
    # #     verbose=1,
    # #     n_init="auto",
    # # )
    # kmeans = MiniBatchKMeans(
    #     n_clusters=args.codebook_size,
    #     batch_size=args.kmeans_batch_size,
    #     random_state=0,
    #     verbose=1,
    #     n_init=3,
    #     max_iter=20,
    #     max_no_improvement=20,
    #     reassignment_ratio=0.0,
    # )
    # vq_ids = kmeans.fit_predict(z_np)

    kmeans = fit_kmeans_streaming(
        model=model,
        ctx=ctx,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    vq_ids = predict_kmeans_streaming(
        model=model,
        kmeans=kmeans,
        ctx=ctx,
        batch_size=args.batch_size,
        device=device,
    )

    used = len(set(vq_ids.tolist()))
    print(f"[kmeans] used={used}/{args.codebook_size}")

    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    vq_ids = torch.tensor(vq_ids, dtype=torch.long)

    torch.save(
        {
            "model": model.state_dict(),
            "centroids": centroids,
            "args": vars(args),
            "word2id": word2id,
            "id2word": id2word,
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": "word",
        },
        args.out,
    )

    id_out = args.out.replace(".pt", "_ids.pt")
    torch.save(
        {
            "vq_ids": vq_ids,
            "tgt": tgt,
            "word2id": word2id,
            "id2word": id2word,
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": "word",
        },
        id_out,
    )

    print(f"[save model] {args.out}")
    print(f"[save ids] {id_out}")

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()