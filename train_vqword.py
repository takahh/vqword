# !/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans


@torch.no_grad()
def init_centers_kmeanspp(z, k, k_block=4096):
    """
    z: [N, D], normalized
    returns centers: [k, D]
    cosine distance版 kmeans++
    """
    z = F.normalize(z.float(), dim=-1)
    N, D = z.shape

    centers = torch.empty((k, D), device=z.device, dtype=z.dtype)

    # 1個目はランダム
    first = torch.randint(0, N, (1,), device=z.device).item()
    centers[0] = z[first]

    # 各点の最近中心までの距離^2
    # normalized cosine: dist^2 ≒ 2 - 2*cos
    closest_dist = torch.full((N,), float("inf"), device=z.device)

    for c in range(1, k):
        new_center = centers[c - 1:c]

        for s in range(0, N, k_block):
            zz = z[s:s + k_block]
            sim = (zz @ new_center.T).squeeze(1)
            dist = (2.0 - 2.0 * sim).clamp_min(0.0)
            closest_dist[s:s + k_block] = torch.minimum(
                closest_dist[s:s + k_block],
                dist,
            )

        total = closest_dist.sum()

        if not torch.isfinite(total) or total <= 1e-12:
            idx = torch.randint(0, N, (1,), device=z.device).item()
        else:
            probs = closest_dist / total
            idx = torch.multinomial(probs, 1).item()

        centers[c] = z[idx]

    return F.normalize(centers, dim=-1)


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
def prune_centers_iterative(z, centers, prune_frac, kmeans_iters=2):
    centers = F.normalize(centers.float(), dim=-1)

    while centers.size(0) >= 3 and prune_frac > 0:
        k = centers.size(0)

        sim = z @ centers.T
        y = sim.argmax(dim=1)
        counts = torch.bincount(y, minlength=k)

        sim_cc = centers @ centers.T
        dist_cc = (2.0 - 2.0 * sim_cc).clamp_min(0.0)
        dist_cc.fill_diagonal_(float("inf"))

        finite_dist = dist_cc[torch.isfinite(dist_cc)]
        if finite_dist.numel() == 0:
            break

        scale = torch.quantile(finite_dist, 0.95).item()
        thresh = scale * prune_frac
        if thresh <= 0:
            break

        pairs = torch.nonzero(dist_cc < thresh, as_tuple=False)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]

        if pairs.numel() == 0:
            break

        # 一番近いペアだけ削除対象にする
        pair_dist = dist_cc[pairs[:, 0], pairs[:, 1]]
        best = pair_dist.argmin()
        a, b = pairs[best].tolist()

        drop = a if counts[a] <= counts[b] else b
        keep = torch.ones(k, dtype=torch.bool, device=centers.device)
        keep[drop] = False

        centers = centers[keep]
        centers = F.normalize(centers, dim=-1)

        # 削除後に少しだけ再収束
        for _ in range(kmeans_iters):
            sim = z @ centers.T
            y = sim.argmax(dim=1)

            sums = torch.zeros_like(centers)
            counts = torch.zeros(centers.size(0), device=z.device)

            sums.index_add_(0, y, z)
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))

            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
            centers = F.normalize(centers, dim=-1)

    return centers


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
            xb = ctx[start:start + batch_size].to(device)
            part = part_all[start:start + batch_size].to(device)

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
        xb = ctx[start:start + batch_size].to(device)
        part = part_all[start:start + batch_size].to(device)

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
            xb = ctx[sel[s:s + batch_size]].to(device)
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


def make_adj_left(seq_len, hop, device):
    """
    adj[i, j] = 1 のとき、位置 i が位置 j から情報を受け取る。
    自分自身および左側hop以内だけを見る。
    """
    pos = torch.arange(seq_len, device=device)
    receiver = pos[:, None]
    sender = pos[None, :]
    distance = receiver - sender
    adj = (
            (distance >= 0) &
            (distance <= hop)
    ).float()
    deg = adj.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)

    return adj / deg


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
    def __init__(self, vocab_size, d_model=256, hop=3, n_layers=3, dropout=0.1, center_scale=1.0):
        super().__init__()
        self.hop = hop
        self.seq_len = hop + 1
        self.center_idx = hop
        self.center_scale = center_scale
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        self.layers = nn.ModuleList([
            AdjGNNLayer(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

    @torch.no_grad()
    def prune_centers_iterative(z, centers, prune_frac, kmeans_iters=2):
        centers = F.normalize(centers.float(), dim=-1)

        while centers.size(0) >= 3 and prune_frac > 0:
            k = centers.size(0)

            sim = z @ centers.T
            y = sim.argmax(dim=1)
            counts = torch.bincount(y, minlength=k)

            sim_cc = centers @ centers.T
            dist_cc = (2.0 - 2.0 * sim_cc).clamp_min(0.0)
            dist_cc.fill_diagonal_(float("inf"))

            finite_dist = dist_cc[torch.isfinite(dist_cc)]
            if finite_dist.numel() == 0:
                break

            scale = torch.quantile(finite_dist, 0.95).item()
            thresh = scale * prune_frac
            if thresh <= 0:
                break

            pairs = torch.nonzero(dist_cc < thresh, as_tuple=False)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]

            if pairs.numel() == 0:
                break

            # 一番近いペアだけ削除対象にする
            pair_dist = dist_cc[pairs[:, 0], pairs[:, 1]]
            best = pair_dist.argmin()
            a, b = pairs[best].tolist()

            drop = a if counts[a] <= counts[b] else b
            keep = torch.ones(k, dtype=torch.bool, device=centers.device)
            keep[drop] = False

            centers = centers[keep]
            centers = F.normalize(centers, dim=-1)

            # 削除後に少しだけ再収束
            for _ in range(kmeans_iters):
                sim = z @ centers.T
                y = sim.argmax(dim=1)

                sums = torch.zeros_like(centers)
                counts = torch.zeros(centers.size(0), device=z.device)

                sums.index_add_(0, y, z)
                counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))

                nonempty = counts > 0
                centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
                centers = F.normalize(centers, dim=-1)

        return centers

    def encode_context(self, ctx_ids):
        # ctx_ids: [B, 2hop+1]
        B, L = ctx_ids.shape
        pos = torch.arange(L, device=ctx_ids.device).unsqueeze(0).expand(B, L)

        tok_h = self.tok_emb(ctx_ids)
        tok_h[:, self.center_idx, :] *= self.center_scale

        h = tok_h + self.pos_emb(pos)
        h = self.dropout(h)

        adj = make_adj_left(L, self.hop, ctx_ids.device)

        for layer in self.layers:
            h = layer(h, adj)

        z = h[:, self.center_idx]
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
    """
    各ターゲット位置 i に対して、

        [i-hop, ..., i-1, i]

    の左文脈 hop 個＋中心トークン自身を返す。

    ctx length = hop + 1
    center index = hop
    """
    ids = torch.tensor(token_ids, dtype=torch.long)

    # 左側だけpadする
    padded = F.pad(ids, (hop, 0), value=pad_id)

    ctx = []
    tgt = []

    for i in range(len(ids)):
        ctx.append(padded[i:i + hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt, dtype=torch.long)


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

        for start in tqdm(range(0, len(ctx), batch_size),
                          desc="[kmeans assign/update]"):
            xb = ctx[start:start + batch_size].to(device)

            z = model.encode_context(xb).float()
            z = F.normalize(z, dim=-1)

            y = assign_blockwise(
                z,
                centers,
                k_block=args.k_block,
            )

            sums.index_add_(0, y, z)
            counts.index_add_(0, y,
                              torch.ones_like(y, dtype=torch.long))

        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = (
                sums[nonempty] /
                counts[nonempty].float().unsqueeze(1)
        )
        new_centers = F.normalize(new_centers, dim=-1)

        shift = (
            (new_centers - centers)
            .pow(2)
            .sum(dim=1)
            .sqrt()
            .mean()
        )

        centers = new_centers

        used = int(nonempty.sum().item())
        print(
            f"[kmeans] used={used}/{K} "
            f"shift={shift.item():.6f}"
        )

    return centers


@torch.no_grad()
def predict_kmeans_torch_streaming(
        model,
        centers,
        ctx,
        batch_size,
        device,
        args,
):
    model.eval()

    ids = []

    centers = centers.to(device)

    print("[kmeans] predict torch blockwise")

    for start in tqdm(range(0, len(ctx), batch_size),
                      desc="[kmeans predict]"):
        xb = ctx[start:start + batch_size].to(device)

        z = model.encode_context(xb).float()

        pred = assign_blockwise(
            z,
            centers,
            k_block=args.k_block,
        )

        ids.append(pred.cpu())

    return torch.cat(ids, dim=0)


@torch.no_grad()
def fit_kmeans_per_token(model, ctx, tgt, batch_size, device, args):
    model.eval()

    print("[per-token kmeans] collect embeddings")
    z_all = collect_embeddings(model, ctx, batch_size, device)

    tgt_cpu = tgt.cpu()

    centers_by_token = {}
    local_counts_by_token = {}
    local_ids = torch.zeros(len(tgt), dtype=torch.long)

    unique_tokens = torch.unique(tgt_cpu)
    print(
        f"[per-token kmeans] tokens={len(unique_tokens):,} "
        f"maxK={args.max_clusters_per_token}"
    )

    for wid in tqdm(unique_tokens.tolist(), desc="[per-token kmeans]"):
        idx = torch.where(tgt_cpu == wid)[0]
        n = len(idx)

        if n < args.min_token_count:
            centers_by_token[int(wid)] = z_all[idx[:1]].clone()
            local_counts_by_token[int(wid)] = torch.tensor(
                [n],
                dtype=torch.long,
            )
            local_ids[idx] = 0
            continue

        k = choose_k_by_freq(n, args)
        k = min(k, n)

        z = z_all[idx].to(device).float()
        z = F.normalize(z, dim=-1)

        centers = init_centers_kmeanspp(
            z=z,
            k=k,
            k_block=args.k_block,
        )

        # -----------------------------
        # normal kmeans iterations
        # -----------------------------
        for _ in range(args.kmeans_iters):
            sim = z @ centers.T
            y = sim.argmax(dim=1)

            sums = torch.zeros_like(centers)
            counts = torch.zeros(k, device=device)

            y_dev = y.to(device)
            sums.index_add_(0, y_dev, z)
            counts.index_add_(0, y_dev, torch.ones_like(y_dev, dtype=torch.float))

            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
            centers = F.normalize(centers, dim=-1)

        # -----------------------------
        # prune near-duplicate centers only once after convergence
        # -----------------------------
        # -----------------------------
        # iterative prune
        # -----------------------------
        if k >= 3 and args.center_prune_frac > 0:
            centers = prune_centers_iterative(
                z=z,
                centers=centers,
                prune_frac=args.center_prune_frac,
                kmeans_iters=2,
            )
            k = centers.size(0)

        # -----------------------------
        # final assignment
        # -----------------------------
        sim = z @ centers.T
        y = sim.argmax(dim=1).cpu()

        # -----------------------------
        # save visualization data
        # -----------------------------
        if int(wid) in args.vis_token_set:
            vis_out = f"{args.vis_dir}/vis_token_{int(wid)}.pt"
            torch.save(
                {
                    "wid": int(wid),
                    "z": z.detach().cpu(),
                    "centers": centers.detach().cpu(),
                    "cluster": y.detach().cpu(),
                    "idx": idx.detach().cpu(),
                    "token_text": None,
                },
                vis_out,
            )
            print(f"[vis] saved token={wid} n={n} k={k} -> {vis_out}")

        # -----------------------------
        # compress local cluster ids
        # -----------------------------
        used = torch.unique(y, sorted=True)

        old2new = {
            int(old): new
            for new, old in enumerate(used.tolist())
        }

        y_compact = torch.tensor(
            [old2new[int(v)] for v in y.tolist()],
            dtype=torch.long,
        )

        centers_compact = centers[used.to(device)].cpu()

        local_ids[idx] = y_compact
        centers_by_token[int(wid)] = centers_compact

        local_counts_by_token[int(wid)] = torch.bincount(
            y_compact,
            minlength=centers_compact.size(0),
        ).long()

    from collections import Counter

    hist = Counter(len(v) for v in centers_by_token.values())

    print("\n[cluster count distribution]")
    for k in sorted(hist):
        print(f"K={k}: {hist[k]:6d} tokens ({hist[k] / len(centers_by_token):6.2%})")

    return centers_by_token, local_counts_by_token, local_ids


@torch.no_grad()
def allocate_k_per_ivf_list(
        ivf_labels,
        local_center_weights,
        requested_k,
        n_lists,
):
    """
    IVFリストごとの第2段階K数を、各リストの出現重み合計に比例して配分する。

    制約:
      - 空でないリストには最低1中心
      - 各リストのKは、そのリスト内local center数を超えない
      - 可能な限り合計Kをrequested_kに一致させる
    """
    counts = torch.bincount(ivf_labels, minlength=n_lists).long()
    weight_sums = torch.zeros(n_lists, dtype=torch.float64)
    weight_sums.index_add_(
        0,
        ivf_labels,
        local_center_weights.double(),
    )

    nonempty = counts > 0
    max_total = int(counts.sum().item())
    target_k = min(int(requested_k), max_total)

    n_nonempty = int(nonempty.sum().item())
    if target_k < n_nonempty:
        raise ValueError(
            f"global_codebook_size={target_k} is smaller than "
            f"nonempty IVF lists={n_nonempty}. Reduce --ivf_nlist."
        )

    # まず各非空リストに1個ずつ保証
    k_per_list = torch.zeros(n_lists, dtype=torch.long)
    k_per_list[nonempty] = 1
    remaining = target_k - n_nonempty

    if remaining <= 0:
        return k_per_list

    capacity = (counts - k_per_list).clamp_min(0)
    active_weight = weight_sums.clone()
    active_weight[~nonempty] = 0.0

    if active_weight.sum() <= 0:
        active_weight = counts.double()

    ideal_extra = remaining * active_weight / active_weight.sum()
    base_extra = torch.floor(ideal_extra).long()
    base_extra = torch.minimum(base_extra, capacity)

    k_per_list += base_extra
    remaining -= int(base_extra.sum().item())

    # 端数の大きい順に1個ずつ追加。capacityに達したリストは飛ばす。
    fractional = ideal_extra - torch.floor(ideal_extra)
    while remaining > 0:
        available = k_per_list < counts
        if not available.any():
            break

        score = fractional.clone()
        score[~available] = -1.0
        order = torch.argsort(score, descending=True)

        added = 0
        for lid in order.tolist():
            if remaining <= 0:
                break
            if k_per_list[lid] >= counts[lid]:
                continue
            k_per_list[lid] += 1
            remaining -= 1
            added += 1

        if added == 0:
            break

    if int(k_per_list.sum().item()) != target_k:
        raise RuntimeError(
            f"Failed to allocate stage2 K: "
            f"allocated={int(k_per_list.sum())} target={target_k}"
        )

    return k_per_list


@torch.no_grad()
def fit_global_ivf_then_kmeans_from_local_centers(
        centers_by_token,
        local_counts_by_token,
        args,
):
    """
    全BPEのlocal centerを、次の2段階で全体共有IDへ統合する。

    Stage 2A: IVF
        local centerを粗いMiniBatchKMeansでivf_nlist個の領域へ単一割当する。

    Stage 2B: KMeans
        各IVF領域の内部だけでKMeansを行う。
        最終Kは領域の出現重み合計に比例配分する。

    マルチアサインは行わない。
    """
    all_centers = []
    all_weights = []
    local_pairs = []

    for wid in sorted(centers_by_token.keys()):
        centers = F.normalize(centers_by_token[wid].float(), dim=-1)
        counts = local_counts_by_token[wid].long()

        if centers.size(0) != counts.numel():
            raise ValueError(
                f"center/count mismatch for token={wid}: "
                f"centers={centers.size(0)} counts={counts.numel()}"
            )

        all_centers.append(centers)
        all_weights.append(counts.float())
        local_pairs.extend(
            (int(wid), int(local_id))
            for local_id in range(centers.size(0))
        )

    local_center_tensor = torch.cat(all_centers, dim=0).float()
    local_center_weights = torch.cat(all_weights, dim=0).float()

    n_local, d_model = local_center_tensor.shape
    requested_k = min(int(args.global_codebook_size), n_local)
    ivf_nlist = min(int(args.ivf_nlist), n_local)

    if ivf_nlist > requested_k:
        raise ValueError(
            f"ivf_nlist={ivf_nlist} must be <= "
            f"global_codebook_size={requested_k}"
        )

    print(
        f"[global IVF] local_centers={n_local:,} "
        f"nlist={ivf_nlist:,} final_K={requested_k:,} D={d_model}"
    )

    x = local_center_tensor.numpy()
    sample_weight = local_center_weights.numpy()

    # ------------------------------------------------------------
    # Stage 2A: coarse IVF partitioning (single assignment)
    # ------------------------------------------------------------
    ivf = MiniBatchKMeans(
        n_clusters=ivf_nlist,
        init="k-means++",
        n_init=1,
        max_iter=args.ivf_iters,
        batch_size=args.ivf_batch_size,
        random_state=args.seed,
        reassignment_ratio=0.01,
        verbose=1,
    )
    ivf.fit(x, sample_weight=sample_weight)

    ivf_labels = torch.from_numpy(
        ivf.labels_.astype(np.int64)
    )
    ivf_counts = torch.bincount(ivf_labels, minlength=ivf_nlist)

    print(
        f"[global IVF] nonempty={int((ivf_counts > 0).sum())}/{ivf_nlist} "
        f"mean={ivf_counts.float().mean().item():.2f} "
        f"max={int(ivf_counts.max())}"
    )

    k_per_list = allocate_k_per_ivf_list(
        ivf_labels=ivf_labels,
        local_center_weights=local_center_weights,
        requested_k=requested_k,
        n_lists=ivf_nlist,
    )

    nonzero_k = k_per_list[k_per_list > 0]
    print(
        f"[global IVF->KMeans] allocated_K={int(k_per_list.sum())} "
        f"min={int(nonzero_k.min())} "
        f"mean={nonzero_k.float().mean().item():.2f} "
        f"max={int(nonzero_k.max())}"
    )

    # ------------------------------------------------------------
    # Stage 2B: independent KMeans inside each IVF list
    # ------------------------------------------------------------
    global_labels = torch.empty(n_local, dtype=torch.long)
    global_centers_parts = []
    next_global_id = 0

    for list_id in tqdm(range(ivf_nlist), desc="[IVF list kmeans]"):
        idx = torch.where(ivf_labels == list_id)[0]
        n_list = int(idx.numel())
        if n_list == 0:
            continue

        k_list = int(k_per_list[list_id].item())
        if k_list <= 0:
            raise RuntimeError(f"nonempty IVF list {list_id} received K=0")

        x_list = x[idx.numpy()]
        w_list = sample_weight[idx.numpy()]

        if k_list == 1:
            weighted_sum = (x_list * w_list[:, None]).sum(axis=0)
            denom = max(float(w_list.sum()), 1e-12)
            center_np = (weighted_sum / denom)[None, :]
            labels_np = np.zeros(n_list, dtype=np.int64)
        elif k_list == n_list:
            center_np = x_list.copy()
            labels_np = np.arange(n_list, dtype=np.int64)
        else:
            km = MiniBatchKMeans(
                n_clusters=k_list,
                init="k-means++",
                n_init=1,
                max_iter=args.global_kmeans_iters,
                batch_size=min(args.global_batch_size, max(k_list * 4, 256)),
                random_state=args.seed + list_id + 1,
                reassignment_ratio=0.01,
                verbose=0,
            )
            km.fit(x_list, sample_weight=w_list)
            center_np = km.cluster_centers_
            labels_np = km.labels_.astype(np.int64)

        labels = torch.from_numpy(labels_np).long()
        used = torch.unique(labels, sorted=True)
        old_to_new = torch.full(
            (k_list,),
            -1,
            dtype=torch.long,
        )
        old_to_new[used] = torch.arange(used.numel())
        labels = old_to_new[labels]

        centers_list = torch.from_numpy(center_np).float()[used]
        centers_list = F.normalize(centers_list, dim=-1)

        global_labels[idx] = labels + next_global_id
        global_centers_parts.append(centers_list)
        next_global_id += centers_list.size(0)

    global_centers = torch.cat(global_centers_parts, dim=0)

    pair_to_global = {}
    global_to_pairs = defaultdict(list)
    for pair, global_id in zip(local_pairs, global_labels.tolist()):
        gid = int(global_id)
        pair_to_global[pair] = gid
        global_to_pairs[gid].append(pair)

    global_to_pairs = dict(global_to_pairs)

    merge_sizes = torch.bincount(
        global_labels,
        minlength=global_centers.size(0),
    )

    print(
        f"[global IVF->KMeans] K_eff={global_centers.size(0):,}/"
        f"{requested_k:,} "
        f"mean_local_centers_per_global={merge_sizes.float().mean().item():.3f} "
        f"max={int(merge_sizes.max())} "
        f"singletons={int((merge_sizes == 1).sum())}"
    )

    return (
        global_centers,
        pair_to_global,
        global_to_pairs,
        global_labels,
        ivf_labels,
        k_per_list,
    )


@torch.no_grad()
def map_local_ids_to_global(
        tgt,
        local_ids,
        pair_to_global,
        vocab_size,
        max_clusters_per_token,
):
    """
    各出現の
        (BPE token ID, local cluster ID)
    をglobal semantic IDへ変換する。
    """

    max_local = max_clusters_per_token

    lookup = torch.full(
        (vocab_size, max_local),
        -1,
        dtype=torch.long,
    )

    for (wid, local_id), global_id in pair_to_global.items():
        if local_id >= max_local:
            raise ValueError(
                f"local_id={local_id} exceeds lookup width={max_local}"
            )

        lookup[int(wid), int(local_id)] = int(global_id)

    global_ids = lookup[
        tgt.long(),
        local_ids.long(),
    ]

    bad = global_ids.lt(0)

    if bad.any():
        bad_idx = torch.where(bad)[0][:20]

        examples = [
            (
                int(tgt[i]),
                int(local_ids[i]),
            )
            for i in bad_idx.tolist()
        ]

        raise ValueError(
            f"Missing local->global mappings: "
            f"count={int(bad.sum())}, examples={examples}"
        )

    return global_ids


def choose_k_by_freq(freq, args):
    if freq < args.min_token_count:
        return 1

    k = int(freq ** args.cluster_freq_power)

    k = max(1, k)
    k = min(k, args.max_clusters_per_token)

    return k


def compact_per_token_ids(tgt, local_ids):
    """
    (original_token_id, local_cluster_id) を 0..K_eff-1 に詰める
    """
    pair_to_compact = {}
    compact_ids = torch.empty_like(local_ids)

    next_id = 0
    for i, pair in enumerate(zip(tgt.cpu().tolist(), local_ids.cpu().tolist())):
        key = (int(pair[0]), int(pair[1]))
        if key not in pair_to_compact:
            pair_to_compact[key] = next_id
            next_id += 1
        compact_ids[i] = pair_to_compact[key]

    compact_to_pair = {
        cid: pair for pair, cid in pair_to_compact.items()
    }

    return compact_ids.long(), pair_to_compact, compact_to_pair


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

    N = len(ctx)
    D = model.tok_emb.embedding_dim

    # 最初から保存先を1個だけ確保する
    z_all = torch.empty((N, D), dtype=torch.float16, device="cpu")

    for start in tqdm(range(0, N, batch_size), desc="[embed]"):
        end = min(start + batch_size, N)

        xb = ctx[start:end].to(device)
        z = model.encode_context(xb)

        # catしない。直接書き込む
        z_all[start:end] = z.detach().cpu().half()

    return z_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--center_scale", type=float, default=1.0)
    ap.add_argument(
        "--tokenizer",
        default="gpt2"
    )
    ap.add_argument("--vis_token", type=int, default=-1)
    ap.add_argument("--vis_out", default="vis_token.pt")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--dataset_config", default=None)
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
    ap.add_argument("--max_clusters_per_token", type=int, default=8)
    ap.add_argument("--cluster_freq_power", type=float, default=0.5)
    ap.add_argument("--min_token_count", type=int, default=2)
    ap.add_argument("--center_prune_frac", type=float, default=0.05)
    ap.add_argument("--vis_tokens", default="")
    ap.add_argument("--vis_dir", default=".")
    ap.add_argument(
        "--ivf_nlist",
        type=int,
        default=256,
        help="第2段階の前半で使うIVF粗領域数",
    )
    ap.add_argument(
        "--ivf_iters",
        type=int,
        default=30,
    )
    ap.add_argument(
        "--ivf_batch_size",
        type=int,
        default=8192,
    )
    ap.add_argument(
        "--global_codebook_size",
        type=int,
        default=50000,
        help="第2段階の全体共有VQ語彙数",
    )
    ap.add_argument(
        "--global_kmeans_iters",
        type=int,
        default=100,
    )
    ap.add_argument(
        "--global_batch_size",
        type=int,
        default=8192,
    )
    args = ap.parse_args()
    args.vis_token_set = set()
    if args.vis_tokens:
        args.vis_token_set = set(int(x) for x in args.vis_tokens.split(",") if x.strip())
    elif args.vis_token >= 0:
        args.vis_token_set = {int(args.vis_token)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = tok.vocab_size

    pad_id = tok.pad_token_id

    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split="train")

    vocab_size = tok.vocab_size
    print(f"[tokenizer] {args.tokenizer}")
    print(f"[vocab_size] {vocab_size}")

    all_ctx, all_tgt = [], []

    print("[data] tokenizing")
    for ex in tqdm(ds.select(range(min(args.max_samples, len(ds))))):
        text = ex[args.text_col]
        ids = tok.encode(
            text,
            add_special_tokens=False
        )[:args.seq_len]
        if len(ids) < 2:
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
        center_scale=args.center_scale,
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

                # 1. current GNN embedding
                z = model.encode_context(xb).float()
                z = F.normalize(z, dim=-1)

                # 2. assign to current centers, but stop gradients to centers
                with torch.no_grad():
                    ids = assign_blockwise(
                        z,
                        centers,
                        k_block=args.k_block,
                    )
                    q = centers[ids].detach()

                # 3. update GNN
                commit_loss = F.mse_loss(z, q)

                soft_ent_loss = soft_entropy_loss(
                    z=z,
                    centers=centers,
                    temperature=args.entropy_temp,
                )

                ent_loss = soft_ent_loss
                loss = commit_loss + args.vq_beta * ent_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                # 4. update codebook EMA after GNN update
                with torch.no_grad():
                    z_new = model.encode_context(xb).float()
                    z_new = F.normalize(z_new, dim=-1)

                    ids_new = assign_blockwise(
                        z_new,
                        centers,
                        k_block=args.k_block,
                    )

                    ema_update_codebook(
                        centroids=centers,
                        z=z_new,
                        ids=ids_new,
                        decay=args.ema_decay,
                    )

                    usage_ema = update_usage_ema(
                        usage_ema,
                        ids_new.detach(),
                        args.codebook_size,
                        decay=args.ema_decay,
                    )
                    ema_ent_loss = entropy_loss_from_probs(usage_ema)

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

        print("[vq-opt] done")
    else:
        print("[vq-opt] skipped")

    # ============================================================
    # Stage 1: BPEごとの内部クラスタリング
    # ============================================================
    centers_by_token, local_counts_by_token, local_ids = (
        fit_kmeans_per_token(
            model=model,
            ctx=ctx,
            tgt=tgt,
            batch_size=args.batch_size,
            device=device,
            args=args,
        )
    )

    n_local_centers = sum(
        centers.size(0)
        for centers in centers_by_token.values()
    )

    print(
        f"[stage1 per-token] "
        f"local_vocab_size={n_local_centers:,}"
    )

    # ============================================================
    # Stage 2: 全local centerをBPE横断でクラスタリング
    # ============================================================
    (
        global_centers,
        pair_to_global,
        global_to_pairs,
        local_center_global_ids,
        local_center_ivf_ids,
        k_per_ivf_list,
    ) = fit_global_ivf_then_kmeans_from_local_centers(
        centers_by_token=centers_by_token,
        local_counts_by_token=local_counts_by_token,
        args=args,
    )

    # 各出現をglobal semantic IDへ変換
    vq_ids = map_local_ids_to_global(
        tgt=tgt,
        local_ids=local_ids,
        pair_to_global=pair_to_global,
        vocab_size=vocab_size,
        max_clusters_per_token=args.max_clusters_per_token,
    )

    global_vq_vocab_size = int(global_centers.size(0))

    print(
        f"[stage2 global] "
        f"global_vq_vocab_size={global_vq_vocab_size:,}"
    )

    from collections import defaultdict, Counter

    cluster_counter = defaultdict(Counter)

    tgt_cpu = tgt.cpu().tolist()
    vq_cpu = vq_ids.cpu().tolist()

    for wid, cid in zip(tgt_cpu, vq_cpu):
        cluster_counter[cid][wid] += 1

    cluster_dict = {
        cid: [
            (wid, tok.decode([wid]), count)
            for wid, count in counter.most_common()
        ]
        for cid, counter in cluster_counter.items()
    }

    dict_out = args.out.replace(".pt", "_dictionary.pt")
    torch.save(cluster_dict, dict_out)
    print(f"[save dictionary] {dict_out}")

    # keep the clean partitioned KMeans IDs
    vq_ids_kmeans = vq_ids.clone()

    metrics = compute_cluster_metrics(vq_ids, K_req=global_vq_vocab_size)
    print(
        f"[CLST] N={metrics['N']} "
        f"K_eff={metrics['K_eff']}/{metrics['K_req']} "
        f"max_frac={metrics['max_frac']:.4f} "
        f"top5_frac={metrics['top5_frac']:.4f} "
        f"H={metrics['entropy']:.4f} "
        f"ppl={metrics['perplexity']:.2f} "
        f"singleton_ratio={metrics['singleton_ratio']:.4f}"
    )

    vq_ids = vq_ids_kmeans.long()

    metrics = compute_cluster_metrics(vq_ids, K_req=global_vq_vocab_size)
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

            # Stage 1
            "centers_by_token": centers_by_token,
            "local_counts_by_token": local_counts_by_token,

            # Stage 2
            "global_centers": global_centers,
            "pair_to_global": pair_to_global,
            "global_to_pairs": global_to_pairs,
            "local_center_global_ids": local_center_global_ids,
            "local_center_ivf_ids": local_center_ivf_ids,
            "k_per_ivf_list": k_per_ivf_list,

            "args": vars(args),
            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,
            "unk_token_id": None,
            "vocab_type": "byte_bpe",

            "partitioned": True,
            "partition_type": "per_token_then_ivf_then_kmeans",

            "local_vq_vocab_size": n_local_centers,
            "vq_vocab_size": global_vq_vocab_size,

            "id_scheme": "hierarchical_local_to_ivf_to_global_kmeans",
            "global_id_min": 0,
            "global_id_max": global_vq_vocab_size - 1,
        },
        args.out,
    )

    id_out = args.out.replace(".pt", "_ids.pt")

    id_out = args.out.replace(".pt", "_ids.pt")

    torch.save(
        {
            # 第2段階global semantic ID
            "vq_ids": vq_ids.to(torch.int32),

            # 元BPE
            "tgt": tgt.to(torch.int32),

            # 第1段階local ID
            "local_ids": local_ids.to(torch.int16),

            "pair_to_global": pair_to_global,
            "global_to_pairs": global_to_pairs,
            "local_center_ivf_ids": local_center_ivf_ids,
            "k_per_ivf_list": k_per_ivf_list,

            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,
            "unk_token_id": None,
            "vocab_type": "byte_bpe",

            "partitioned": True,
            "partition_type": "per_token_then_ivf_then_kmeans",

            "local_vq_vocab_size": n_local_centers,
            "vq_vocab_size": global_vq_vocab_size,

            "id_scheme": "hierarchical_local_to_ivf_to_global_kmeans",
            "global_id_min": 0,
            "global_id_max": global_vq_vocab_size - 1,
        },
        id_out,
    )

    print(f"[save model] {args.out}")
    print(f"[save ids] {id_out}")

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()