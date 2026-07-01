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
def fit_kmeans_partitioned_streaming(model, ctx, tgt, batch_size, device, args):
    model.eval()

    P = args.n_partitions
    Kp = args.codebook_size // P
    D = args.d_model

    centers = init_centers_partitioned_random(
        model, ctx, tgt, P, Kp, batch_size, device
    )

    part_all = get_partitions(tgt, P)

    for it in range(args.kmeans_iters):
        print(f"[pkmeans] iter {it + 1}/{args.kmeans_iters}")

        sums = torch.zeros(P, Kp, D, device=device)
        counts = torch.zeros(P, Kp, device=device, dtype=torch.long)

        for start in tqdm(range(0, len(ctx), batch_size), desc="[pkmeans assign/update]"):
            xb = ctx[start:start+batch_size].to(device)
            part = part_all[start:start+batch_size].to(device)

            z = model.encode_context(xb).float()
            z = F.normalize(z, dim=-1)

            ids = assign_partitioned(
                z,
                centers,
                part,
                Kp,
                partition_base=args.partition_base,
            )

            _, local_ids = global_to_local(
                ids,
                codes_per_partition=Kp,
                partition_base=args.partition_base,
            )

            for p in part.unique():
                m = part == p
                sums[p].index_add_(0, local_ids[m], z[m])
                counts[p].index_add_(0, local_ids[m], torch.ones_like(local_ids[m], dtype=torch.long))

        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = sums[nonempty] / counts[nonempty].float().unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=-1)

        shift = (new_centers - centers).pow(2).sum(dim=-1).sqrt().mean()
        centers = new_centers

        used = int(nonempty.sum().item())
        print(f"[pkmeans] used={used}/{args.codebook_size} shift={shift.item():.6f}")

    return centers

@torch.no_grad()
def predict_partitioned_streaming(model, centers, ctx, tgt, batch_size, device, args):
    model.eval()
    ids_all = []

    P = args.n_partitions
    Kp = args.codebook_size // P
    part_all = get_partitions(tgt, P)

    centers = centers.to(device)

    print("[pkmeans] predict partitioned")
    for start in tqdm(range(0, len(ctx), batch_size), desc="[pkmeans predict]"):
        xb = ctx[start:start+batch_size].to(device)
        part = part_all[start:start+batch_size].to(device)

        z = model.encode_context(xb).float()
        pred = assign_partitioned(
            z,
            centers,
            part,
            Kp,
            partition_base=args.partition_base,
        )
        ids_all.append(pred.cpu())

    return torch.cat(ids_all, dim=0)


@torch.no_grad()
def init_centers_partitioned_random(model, ctx, tgt, P, Kp, batch_size, device):
    D = model.tok_emb.embedding_dim
    centers = torch.zeros(P, Kp, D, device=device)

    part_all = get_partitions(tgt, P)

    for p in tqdm(range(P), desc="[partition init]"):
        idx = torch.where(part_all == p)[0]

        if len(idx) == 0:
            centers[p] = F.normalize(torch.randn(Kp, D, device=device), dim=-1)
            continue

        if len(idx) >= Kp:
            sel = idx[torch.randperm(len(idx))[:Kp]]
        else:
            sel = idx[torch.randint(0, len(idx), (Kp,))]

        zs = []
        for s in range(0, len(sel), batch_size):
            xb = ctx[sel[s:s+batch_size]].to(device)
            z = model.encode_context(xb)
            zs.append(z.float())

        centers[p] = F.normalize(torch.cat(zs, dim=0)[:Kp], dim=-1)

    return centers

@torch.no_grad()
def init_centers_random(model, ctx, K, batch_size, device):
    D = model.tok_emb.embedding_dim
    N = len(ctx)

    if N >= K:
        sel = torch.randperm(N)[:K]
    else:
        sel = torch.randint(0, N, (K,))

    zs = []
    for s in tqdm(range(0, len(sel), batch_size), desc="[kmeans init]"):
        xb = ctx[sel[s:s + batch_size]].to(device)
        z = model.encode_context(xb).float()
        zs.append(z.cpu())

    centers = torch.cat(zs, dim=0)[:K].to(device)
    return F.normalize(centers, dim=-1)

@torch.no_grad()
def assign_blockwise(z, centers, k_block=4096):
    z = F.normalize(z.float(), dim=-1)
    centers = F.normalize(centers.float(), dim=-1)

    B = z.size(0)
    best_sim = torch.full((B,), -1e9, device=z.device)
    best_id = torch.zeros(B, dtype=torch.long, device=z.device)

    for s in range(0, centers.size(0), k_block):
        c = centers[s:s + k_block]
        sim = z @ c.T
        val, idx = sim.max(dim=1)

        m = val > best_sim
        best_sim[m] = val[m]
        best_id[m] = idx[m] + s

    return best_id

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
def update_usage_ema(usage_ema, ids, K, decay=0.99):
    batch_counts = torch.bincount(ids, minlength=K).float().to(usage_ema.device)
    batch_probs = batch_counts / batch_counts.sum().clamp_min(1.0)
    usage_ema.mul_(decay).add_(batch_probs, alpha=1.0 - decay)
    usage_ema.div_(usage_ema.sum().clamp_min(1e-12))
    return usage_ema

def entropy_loss_from_probs(p):
    p = p[p > 0]
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return -entropy

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

def get_partitions(target_ids, n_partitions):
    return target_ids % n_partitions

def local_to_global(part_ids, local_ids, codes_per_partition, partition_base=0):
    return partition_base + part_ids * codes_per_partition + local_ids


def global_to_local(global_ids, codes_per_partition, partition_base=0):
    x = global_ids - partition_base
    part_ids = x // codes_per_partition
    local_ids = x % codes_per_partition
    return part_ids, local_ids

def make_windows(token_ids, hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (hop, hop), value=pad_id)

    ctx, tgt = [], []
    for i in range(len(ids)):
        ctx.append(padded[i:i + 2 * hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt, dtype=torch.long)


def soft_entropy_loss(z, centers, temperature=0.1):
    z = F.normalize(z.float(), dim=-1)
    centers = F.normalize(centers.float(), dim=-1)

    sim = z @ centers.T
    prob = F.softmax(sim / temperature, dim=-1)

    usage = prob.mean(dim=0)
    entropy = -(usage * torch.log(usage + 1e-12)).sum()

    return -entropy

def entropy_loss_from_ids(ids, K, device):
    counts = torch.bincount(ids, minlength=K).float().to(device)
    p = counts / counts.sum().clamp_min(1.0)
    p = p[p > 0]
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return -entropy

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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--entropy_temp", type=float, default=0.1)

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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[data] windows={len(tgt):,} vocab={vocab_size}")
    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=args.d_model,
        hop=args.hop,
        n_layers=args.n_layers,
    ).to(device)

    usage_ema = torch.full(
        (args.codebook_size,),
        1.0 / args.codebook_size,
        device=device,
    )

    if args.epochs > 0:
        print(f"[vq-opt] epochs={args.epochs}")

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )

        centers = init_centers_random(
            model=model,
            ctx=ctx,
            K=args.codebook_size,
            batch_size=args.batch_size,
            device=device,
        ).detach()

        for ep in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(ctx), device="cpu")

            total_loss = 0.0
            total_commit = 0.0
            total_ent = 0.0
            total_n = 0

            pbar = tqdm(
                range(0, len(ctx), args.batch_size),
                desc=f"[vq-opt] epoch {ep}",
            )

            for start in pbar:
                idx = perm[start:start + args.batch_size]

                xb = ctx[idx].to(device)

                z = model.encode_context(xb).float()
                z = F.normalize(z, dim=-1)

                with torch.no_grad():
                    ids = assign_blockwise(
                        z,
                        centers,
                        k_block=args.k_block,
                    )

                    q = centers[ids].detach()

                commit_loss = F.mse_loss(z, q)

                soft_ent_loss = soft_entropy_loss(
                    z=z,
                    centers=centers,
                    temperature=args.entropy_temp,
                )

                with torch.no_grad():
                    usage_ema = update_usage_ema(
                        usage_ema,
                        ids.detach(),
                        args.codebook_size,
                        decay=args.ema_decay,
                    )
                    ema_ent_loss = entropy_loss_from_probs(usage_ema)

                ent_loss = soft_ent_loss

                loss = commit_loss + args.vq_beta * ent_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_commit += commit_loss.item() * bs
                total_ent += ent_loss.item() * bs
                total_n += bs

                pbar.set_postfix(
                    loss=f"{total_loss / total_n:.4f}",
                    commit=f"{total_commit / total_n:.4f}",
                    soft_ent=f"{total_ent / total_n:.4f}",
                    ema_ent=f"{ema_ent_loss.item():.4f}",
                )

            print(
                f"[vq-opt] ep={ep} "
                f"loss={total_loss / total_n:.4f} "
                f"commit={total_commit / total_n:.4f} "
                f"ent={total_ent / total_n:.4f}"
            )

        print("[vq-opt] done")
    else:
        print("[vq-opt] skipped")

    centroids = fit_kmeans_partitioned_streaming(
        model=model,
        ctx=ctx,
        tgt=tgt,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    vq_ids = predict_partitioned_streaming(
        model=model,
        centers=centroids,
        ctx=ctx,
        tgt=tgt,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    from collections import defaultdict, Counter

    cluster_counter = defaultdict(Counter)

    tgt_cpu = tgt.cpu().tolist()
    vq_cpu = vq_ids.cpu().tolist()

    for wid, cid in zip(tgt_cpu, vq_cpu):
        cluster_counter[cid][wid] += 1

    cluster_dict = {
        cid: [
            (wid, id2word[wid], count)
            for wid, count in counter.most_common()
        ]
        for cid, counter in cluster_counter.items()
    }

    dict_out = args.out.replace(".pt", "_dictionary.pt")
    torch.save(cluster_dict, dict_out)
    print(f"[save dictionary] {dict_out}")

    # keep the clean partitioned KMeans IDs
    vq_ids_kmeans = vq_ids.clone()

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
    vq_ids = vq_ids_kmeans.long()

    metrics = compute_cluster_metrics(vq_ids, K_req=args.codebook_size)
    print(
        f"[FINAL CLST] N={metrics['N']} "
        f"K_eff={metrics['K_eff']}/{metrics['K_req']} "
        f"max_frac={metrics['max_frac']:.4f} "
        f"top5_frac={metrics['top5_frac']:.4f} "
        f"H={metrics['entropy']:.4f} "
        f"ppl={metrics['perplexity']:.2f} "
        f"singleton_ratio={metrics['singleton_ratio']:.4f}"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "centroids": centroids.cpu().float(),
            "args": vars(args),
            "word2id": word2id,
            "id2word": id2word,
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": "word",
            "partitioned": True,
            "n_partitions": args.n_partitions,
            "codes_per_partition": args.codebook_size // args.n_partitions,
            "id_scheme": "partition_offset",
            "partition_base": args.partition_base,
            "global_id_min": args.partition_base,
            "global_id_max": args.partition_base + args.codebook_size - 1,
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
            "partitioned": True,
            "n_partitions": args.n_partitions,
            "codes_per_partition": args.codebook_size // args.n_partitions,
            "id_scheme": "partition_offset",
            "partition_base": args.partition_base,
            "global_id_min": args.partition_base,
            "global_id_max": args.partition_base + args.codebook_size - 1,
        },
        id_out,
    )

    print(f"[save model] {args.out}")
    print(f"[save ids] {id_out}")

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()