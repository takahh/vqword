#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")

import numpy as np

@torch.no_grad()
def compute_cluster_metrics(y, K_req, topk=5):
    y = y.long().view(-1)
    N = y.numel()
    bc = torch.bincount(y, minlength=K_req)
    nz = bc[bc > 0]
    p = nz.float() / max(1, N)

    return {
        "N": int(N),
        "K_req": int(K_req),
        "K_eff": int(nz.numel()),
        "max_frac": float(p.max().item()) if nz.numel() else 0.0,
        "top5_frac": float(torch.topk(p, min(topk, p.numel())).values.sum().item()) if nz.numel() else 0.0,
        "entropy": float(-(p * torch.log(p.clamp_min(1e-12))).sum().item()) if nz.numel() else 0.0,
        "perplexity": float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()) if nz.numel() else 1.0,
        "singleton_ratio": float((nz == 1).float().mean().item()) if nz.numel() else 0.0,
    }


@torch.no_grad()
def init_centers_random(model, ctx, K, batch_size, device):
    """
    Randomly sample K encoded windows without storing all embeddings.
    """
    N = len(ctx)
    if N < K:
        raise ValueError(f"Not enough windows: N={N}, K={K}")

    idx = torch.randperm(N)[:K]
    idx, _ = torch.sort(idx)

    centers = []
    for s in tqdm(range(0, K, batch_size), desc="[kmeans init]"):
        batch_idx = idx[s:s + batch_size]
        xb = ctx[batch_idx].to(device)
        z = model.encode_context(xb)
        centers.append(z.float())

    centers = torch.cat(centers, dim=0)
    centers = F.normalize(centers, dim=-1)
    return centers


@torch.no_grad()
def assign_blockwise(z, centers, k_block=4096):
    """
    z:       [B, D]
    centers:[K, D]
    returns [B]
    """
    z = F.normalize(z.float(), dim=-1)
    centers = F.normalize(centers.float(), dim=-1)

    B = z.size(0)
    best_val = torch.full((B,), -float("inf"), device=z.device)
    best_idx = torch.zeros((B,), dtype=torch.long, device=z.device)

    K = centers.size(0)
    for k0 in range(0, K, k_block):
        k1 = min(k0 + k_block, K)
        sim = z @ centers[k0:k1].T
        val, idx = sim.max(dim=1)

        update = val > best_val
        best_val = torch.where(update, val, best_val)
        best_idx = torch.where(update, idx + k0, best_idx)

    return best_idx


@torch.no_grad()
def fit_kmeans_torch_streaming(model, ctx, batch_size, device, args):
    model.eval()

    K = args.codebook_size
    D = args.d_model

    print("[kmeans] torch blockwise init")
    centers = init_centers_random(
        model=model,
        ctx=ctx,
        K=K,
        batch_size=batch_size,
        device=device,
    )

    for it in range(args.kmeans_iters):
        print(f"[kmeans] iter {it + 1}/{args.kmeans_iters}")

        sums = torch.zeros(K, D, device=device, dtype=torch.float32)
        counts = torch.zeros(K, device=device, dtype=torch.long)

        for start in tqdm(range(0, len(ctx), batch_size), desc="[kmeans assign/update]"):
            xb = ctx[start:start + batch_size].to(device)
            z = model.encode_context(xb).float()
            z = F.normalize(z, dim=-1)

            y = assign_blockwise(z, centers, k_block=args.k_block)

            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.long))

        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = sums[nonempty] / counts[nonempty].float().unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=-1)

        shift = (new_centers - centers).pow(2).sum(dim=1).sqrt().mean()
        centers = new_centers

        used = int(nonempty.sum().item())
        print(f"[kmeans] used={used}/{K} shift={shift.item():.6f}")

    return centers

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

@torch.no_grad()
def predict_kmeans_torch_streaming(model, centers, ctx, batch_size, device, args):
    model.eval()
    ids = []

    centers = centers.to(device)

    print("[kmeans] predict torch blockwise")
    for start in tqdm(range(0, len(ctx), batch_size), desc="[kmeans predict]"):
        xb = ctx[start:start + batch_size].to(device)
        z = model.encode_context(xb).float()
        pred = assign_blockwise(z, centers, k_block=args.k_block)
        ids.append(pred.cpu())

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

@torch.no_grad()
def ema_update_codebook(centroids, z, ids, decay=0.99):
    K, D = centroids.shape

    sums = torch.zeros_like(centroids)
    counts = torch.zeros(K, device=z.device)

    sums.index_add_(0, ids, z)
    counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))

    nonempty = counts > 0
    batch_means = centroids.clone()
    batch_means[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)

    centroids.mul_(decay).add_(batch_means, alpha=1 - decay)
    centroids.copy_(F.normalize(centroids, dim=-1))


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
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--codebook_size", type=int, default=8192)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="vqword_gnn_kmeans.pt")
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--k_block", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vq_beta", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.99)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pad_id = 0
    unk_id = 1

    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split="train")

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

    print(f"[data] windows={len(tgt):,} vocab={vocab_size}")
    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=args.d_model,
        hop=args.hop,
        n_layers=args.n_layers,
    ).to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    centroids = fit_kmeans_torch_streaming(
        model=model,
        ctx=ctx,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    vq_ids = predict_kmeans_torch_streaming(
        model=model,
        centers=centroids,
        ctx=ctx,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    metrics = compute_cluster_metrics(vq_ids, K_req=args.codebook_size)
    print(
        f"[CLST] N={metrics['N']} "
        f"K_eff={metrics['K_eff']}/{metrics['K_req']} "
        f"max_frac={metrics['max_frac']:.4f} "
        f"top5_frac={metrics['top5_frac']:.4f} "
        f"H={metrics['entropy']:.4f} "
        f"ppl={metrics['perplexity']:.2f} "
        f"singleton_ratio={metrics['singleton_ratio']:.4f}"
    )

    centroids = centroids.cpu().float()
    vq_ids = vq_ids.long()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    centroids = centroids.to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for start in tqdm(range(0, len(ctx), args.batch_size), desc=f"[train {epoch + 1}]"):
            xb = ctx[start:start + args.batch_size].to(device)
            yb = tgt[start:start + args.batch_size].to(device)

            z = model.encode_context(xb)
            z = F.normalize(z, dim=-1)

            with torch.no_grad():
                ids = assign_blockwise(z, centroids, k_block=args.k_block)

            q = centroids[ids]

            # VQ-style commitment loss
            vq_loss = F.mse_loss(z, q.detach())

            # optional decoder CE loss
            logits = model.decoder(z)
            ce_loss = F.cross_entropy(logits, yb)

            loss = ce_loss + args.vq_beta * vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                z2 = model.encode_context(xb)
                z2 = F.normalize(z2, dim=-1)
                ids2 = assign_blockwise(z2, centroids, k_block=args.k_block)
                ema_update_codebook(centroids, z2, ids2, decay=args.ema_decay)

            total_loss += loss.item()

        print(f"[epoch {epoch + 1}] loss={total_loss:.4f}")

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