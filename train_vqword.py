#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoTokenizer

@torch.no_grad()
def compute_cluster_metrics(y, k_req, topk=5):
    y = y.long().view(-1)
    n = y.numel()
    bc = torch.bincount(y, minlength=k_req)
    nz = bc[bc > 0]
    p = nz.float() / max(1, n)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum() if nz.numel() else torch.tensor(0.0)

    return {
        "N": int(n),
        "K_req": int(k_req),
        "K_eff": int(nz.numel()),
        "max_frac": float(p.max().item()) if nz.numel() else 0.0,
        "top5_frac": float(torch.topk(p, min(topk, p.numel())).values.sum().item()) if nz.numel() else 0.0,
        "entropy": float(entropy.item()),
        "perplexity": float(torch.exp(entropy).item()) if nz.numel() else 1.0,
        "singleton_ratio": float((nz == 1).float().mean().item()) if nz.numel() else 0.0,
    }


def make_adj_left(seq_len, hop, device):
    pos = torch.arange(seq_len, device=device)
    receiver = pos[:, None]
    sender = pos[None, :]
    distance = receiver - sender
    adj = ((distance >= 0) & (distance <= hop)).float()
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return adj / deg


class AdjGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.nei_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        m = torch.einsum("ij,bjd->bid", adj, h)
        out = self.self_lin(h) + self.nei_lin(m)
        out = F.gelu(out)
        return self.norm(h + out)


class VQWordGNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        hop=3,
        n_layers=3,
        dropout=0.1,
        center_scale=1.0,
    ):
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

    def encode_context(self, ctx_ids):
        batch_size, length = ctx_ids.shape
        pos = torch.arange(length, device=ctx_ids.device)
        pos = pos.unsqueeze(0).expand(batch_size, length)

        tok_h = self.tok_emb(ctx_ids)
        tok_h[:, self.center_idx, :] *= self.center_scale

        h = self.dropout(tok_h + self.pos_emb(pos))
        adj = make_adj_left(length, self.hop, ctx_ids.device)

        for layer in self.layers:
            h = layer(h, adj)

        return F.normalize(h[:, self.center_idx], dim=-1)

    def forward(self, ctx_ids, target_ids):
        z = self.encode_context(ctx_ids)
        logits = self.decoder(z)
        loss = F.cross_entropy(logits, target_ids)
        return loss, logits, z


@torch.no_grad()
def assign_blockwise(z, centers, k_block=4096):
    z = F.normalize(z.float(), dim=-1)
    centers = F.normalize(centers.float(), dim=-1)

    best_sim = torch.full((z.size(0),), -1e9, device=z.device)
    best_id = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

    for start in range(0, centers.size(0), k_block):
        c = centers[start:start + k_block]
        sim = z @ c.T
        value, index = sim.max(dim=1)
        mask = value > best_sim
        best_sim[mask] = value[mask]
        best_id[mask] = index[mask] + start

    return best_id


def make_windows(token_ids, hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (hop, 0), value=pad_id)

    ctx = []
    tgt = []
    for i in range(len(ids)):
        ctx.append(padded[i:i + hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt, dtype=torch.long)


@torch.no_grad()
def encode_batch(model, ctx, start, batch_size, device):
    end = min(start + batch_size, len(ctx))
    xb = ctx[start:end].to(device)
    z = model.encode_context(xb).float()
    return z, end


@torch.no_grad()
def fit_ivf_streaming(model, ctx, batch_size, device, args):
    """Fit coarse IVF centers without storing all embeddings."""
    n = len(ctx)
    nlist = min(args.ivf_nlist, n)
    if nlist < 1:
        raise ValueError("ivf_nlist must be positive")

    ivf = MiniBatchKMeans(
        n_clusters=nlist,
        init="k-means++",
        n_init=1,
        batch_size=max(args.ivf_batch_size, nlist),
        random_state=args.seed,
        reassignment_ratio=0.01,
        verbose=0,
    )

    initialized = False
    pending = []
    pending_n = 0

    for epoch in range(args.ivf_iters):
        pbar = tqdm(range(0, n, batch_size), desc=f"[IVF fit] pass {epoch + 1}/{args.ivf_iters}")
        for start in pbar:
            z, _ = encode_batch(model, ctx, start, batch_size, device)
            x = z.cpu().numpy().astype(np.float32, copy=False)

            if not initialized:
                pending.append(x)
                pending_n += len(x)
                if pending_n < nlist:
                    continue
                first = np.concatenate(pending, axis=0)
                ivf.partial_fit(first)
                initialized = True
                pending.clear()
            else:
                ivf.partial_fit(x)

    if not initialized:
        raise RuntimeError("Not enough samples to initialize IVF")

    centers = torch.from_numpy(ivf.cluster_centers_).float()
    centers = F.normalize(centers, dim=-1)
    return centers


@torch.no_grad()
def count_ivf_lists(model, ctx, ivf_centers, batch_size, device, k_block):
    centers = ivf_centers.to(device)
    counts = torch.zeros(centers.size(0), dtype=torch.long)

    for start in tqdm(range(0, len(ctx), batch_size), desc="[IVF count]"):
        z, _ = encode_batch(model, ctx, start, batch_size, device)
        ids = assign_blockwise(z, centers, k_block=k_block).cpu()
        counts += torch.bincount(ids, minlength=centers.size(0))

    return counts


def allocate_k_per_ivf_list(ivf_counts, requested_k):
    """Allocate final centers in proportion to point counts."""
    counts = ivf_counts.long()
    nonempty = counts > 0
    n_nonempty = int(nonempty.sum().item())
    target_k = min(int(requested_k), int(counts.sum().item()))

    if target_k < n_nonempty:
        raise ValueError(
            f"global_codebook_size={target_k} is smaller than "
            f"nonempty IVF lists={n_nonempty}. Reduce --ivf_nlist."
        )

    k_per_list = torch.zeros_like(counts)
    k_per_list[nonempty] = 1
    remaining = target_k - n_nonempty
    if remaining == 0:
        return k_per_list

    capacity = (counts - k_per_list).clamp_min(0)
    weights = counts.double()
    ideal = remaining * weights / weights.sum().clamp_min(1.0)
    extra = torch.minimum(torch.floor(ideal).long(), capacity)
    k_per_list += extra
    remaining -= int(extra.sum().item())

    fractional = ideal - torch.floor(ideal)
    while remaining > 0:
        available = k_per_list < counts
        if not available.any():
            break
        score = fractional.clone()
        score[~available] = -1.0
        for list_id in torch.argsort(score, descending=True).tolist():
            if remaining <= 0:
                break
            if k_per_list[list_id] >= counts[list_id]:
                continue
            k_per_list[list_id] += 1
            remaining -= 1

    if int(k_per_list.sum().item()) != target_k:
        raise RuntimeError(
            f"Failed to allocate final K: allocated={int(k_per_list.sum())}, "
            f"target={target_k}"
        )

    return k_per_list


@torch.no_grad()
def initialize_fine_centers_streaming(
    model,
    ctx,
    ivf_centers,
    k_per_list,
    batch_size,
    device,
    k_block,
):
    """Collect K initial points from each IVF list without per-point Python loops."""
    nlist = ivf_centers.size(0)
    d_model = ivf_centers.size(1)
    offsets = torch.zeros(nlist + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(k_per_list, dim=0)
    total_k = int(offsets[-1].item())

    initial = torch.empty((total_k, d_model), dtype=torch.float32)
    filled = torch.zeros(nlist, dtype=torch.long)
    coarse = ivf_centers.to(device)

    for start in tqdm(range(0, len(ctx), batch_size), desc="[fine init]"):
        z, _ = encode_batch(model, ctx, start, batch_size, device)
        ivf_ids = assign_blockwise(z, coarse, k_block=k_block).cpu()
        z_cpu = z.cpu()

        for list_id in torch.unique(ivf_ids).tolist():
            need = int(k_per_list[list_id] - filled[list_id])
            if need <= 0:
                continue

            candidates = z_cpu[ivf_ids == list_id]
            take = min(need, candidates.size(0))
            if take == 0:
                continue

            begin = int(offsets[list_id] + filled[list_id])
            initial[begin:begin + take] = candidates[:take]
            filled[list_id] += take

        if torch.equal(filled, k_per_list):
            break

    if not torch.equal(filled, k_per_list):
        bad = torch.where(filled != k_per_list)[0].tolist()[:20]
        raise RuntimeError(f"Failed to initialize fine centers for IVF lists: {bad}")

    return F.normalize(initial, dim=-1), offsets


@torch.no_grad()
def fit_fine_kmeans_streaming(
    model,
    ctx,
    ivf_centers,
    fine_centers,
    offsets,
    batch_size,
    device,
    args,
):
    coarse = ivf_centers.to(device)
    centers = fine_centers.to(device)
    total_k, d_model = centers.shape

    for iteration in range(args.global_kmeans_iters):
        sums = torch.zeros((total_k, d_model), device=device)
        counts = torch.zeros(total_k, device=device)

        pbar = tqdm(
            range(0, len(ctx), batch_size),
            desc=f"[fine kmeans] iter {iteration + 1}/{args.global_kmeans_iters}",
        )
        for start in pbar:
            z, _ = encode_batch(model, ctx, start, batch_size, device)
            ivf_ids = assign_blockwise(z, coarse, k_block=args.k_block)

            for list_id in torch.unique(ivf_ids).tolist():
                mask = ivf_ids == list_id
                begin = int(offsets[list_id])
                end = int(offsets[list_id + 1])
                local_centers = centers[begin:end]
                local_ids = assign_blockwise(
                    z[mask],
                    local_centers,
                    k_block=args.k_block,
                )
                global_ids = local_ids + begin
                sums.index_add_(0, global_ids, z[mask])
                counts.index_add_(
                    0,
                    global_ids,
                    torch.ones_like(global_ids, dtype=torch.float),
                )

        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=-1)
        shift = (new_centers - centers).pow(2).sum(dim=1).sqrt().mean().item()
        centers = new_centers

        print(
            f"[fine kmeans] used={int(nonempty.sum())}/{total_k} "
            f"shift={shift:.6f}"
        )

    return centers.cpu()


@torch.no_grad()
def assign_global_ids_streaming(
    model,
    ctx,
    ivf_centers,
    fine_centers,
    offsets,
    batch_size,
    device,
    k_block,
):
    coarse = ivf_centers.to(device)
    fine = fine_centers.to(device)
    vq_ids = torch.empty(len(ctx), dtype=torch.long)
    ivf_ids_all = torch.empty(len(ctx), dtype=torch.long)

    for start in tqdm(range(0, len(ctx), batch_size), desc="[final assign]"):
        z, end = encode_batch(model, ctx, start, batch_size, device)
        ivf_ids = assign_blockwise(z, coarse, k_block=k_block)
        batch_global = torch.empty(z.size(0), dtype=torch.long, device=device)

        for list_id in torch.unique(ivf_ids).tolist():
            mask = ivf_ids == list_id
            begin = int(offsets[list_id])
            finish = int(offsets[list_id + 1])
            local_ids = assign_blockwise(
                z[mask],
                fine[begin:finish],
                k_block=k_block,
            )
            batch_global[mask] = local_ids + begin

        vq_ids[start:end] = batch_global.cpu()
        ivf_ids_all[start:end] = ivf_ids.cpu()

    return vq_ids, ivf_ids_all


@torch.no_grad()
def fit_global_ivf_then_kmeans_streaming(model, ctx, batch_size, device, args):
    print("[stage 1] fit global IVF")
    ivf_centers = fit_ivf_streaming(
        model=model,
        ctx=ctx,
        batch_size=batch_size,
        device=device,
        args=args,
    )

    print("[stage 1] count points in IVF lists")
    ivf_counts = count_ivf_lists(
        model=model,
        ctx=ctx,
        ivf_centers=ivf_centers,
        batch_size=batch_size,
        device=device,
        k_block=args.k_block,
    )

    k_per_list = allocate_k_per_ivf_list(
        ivf_counts=ivf_counts,
        requested_k=args.global_codebook_size,
    )
    nonzero = k_per_list[k_per_list > 0]
    print(
        f"[stage 2 allocation] total={int(k_per_list.sum())} "
        f"min={int(nonzero.min())} mean={nonzero.float().mean().item():.2f} "
        f"max={int(nonzero.max())}"
    )

    print("[stage 2] initialize fine centers")
    fine_centers, offsets = initialize_fine_centers_streaming(
        model=model,
        ctx=ctx,
        ivf_centers=ivf_centers,
        k_per_list=k_per_list,
        batch_size=batch_size,
        device=device,
        k_block=args.k_block,
    )

    print("[stage 2] fit KMeans inside each IVF list")
    global_centers = fit_fine_kmeans_streaming(
        model=model,
        ctx=ctx,
        ivf_centers=ivf_centers,
        fine_centers=fine_centers,
        offsets=offsets,
        batch_size=batch_size,
        device=device,
        args=args,
    )

    print("[stage 2] final assignment")
    vq_ids, ivf_ids = assign_global_ids_streaming(
        model=model,
        ctx=ctx,
        ivf_centers=ivf_centers,
        fine_centers=global_centers,
        offsets=offsets,
        batch_size=batch_size,
        device=device,
        k_block=args.k_block,
    )

    return (
        global_centers,
        vq_ids,
        ivf_centers,
        ivf_ids,
        k_per_list,
        offsets,
        ivf_counts,
    )


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hop", type=int, default=3)

    # Model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--center_scale", type=float, default=1.0)

    # Global IVF -> KMeans
    ap.add_argument("--ivf_nlist", type=int, default=128)
    ap.add_argument(
        "--ivf_iters",
        type=int,
        default=1,
        help="Number of full streaming passes used to fit coarse IVF",
    )
    ap.add_argument("--ivf_batch_size", type=int, default=8192)
    ap.add_argument("--global_codebook_size", type=int, default=50000)
    ap.add_argument(
        "--global_kmeans_iters",
        type=int,
        default=5,
        help="Number of full streaming KMeans passes inside IVF lists",
    )
    ap.add_argument("--global_batch_size", type=int, default=8192)

    # Utilities
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--k_block", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vqword_global_ivf.pt")

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = tok.vocab_size
    pad_id = tok.pad_token_id

    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split="train")

    print(f"[tokenizer] {args.tokenizer}")
    print(f"[vocab_size] {vocab_size}")

    all_ctx = []
    all_tgt = []
    print("[data] tokenizing")
    limit = min(args.max_samples, len(ds))
    for ex in tqdm(ds.select(range(limit))):
        ids = tok.encode(ex[args.text_col], add_special_tokens=False)[:args.seq_len]
        if len(ids) < 2:
            continue
        ctx, tgt = make_windows(ids, args.hop, pad_id)
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    if not all_ctx:
        raise ValueError("No usable tokenized samples")

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    print(f"[data] windows={len(tgt):,} vocab={vocab_size}")

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=args.d_model,
        hop=args.hop,
        n_layers=args.n_layers,
        center_scale=args.center_scale,
    ).to(device)

    model.eval()
    (
        global_centers,
        vq_ids,
        ivf_centers,
        ivf_ids,
        k_per_ivf_list,
        global_offsets,
        ivf_counts,
    ) = fit_global_ivf_then_kmeans_streaming(
        model=model,
        ctx=ctx,
        batch_size=args.batch_size,
        device=device,
        args=args,
    )

    global_vq_vocab_size = int(global_centers.size(0))
    print(f"[global] vq_vocab_size={global_vq_vocab_size:,}")

    cluster_counter = defaultdict(Counter)
    for word_id, cluster_id in zip(tgt.tolist(), vq_ids.tolist()):
        cluster_counter[int(cluster_id)][int(word_id)] += 1

    cluster_dict = {
        cluster_id: [
            (word_id, tok.decode([word_id]), count)
            for word_id, count in counter.most_common()
        ]
        for cluster_id, counter in cluster_counter.items()
    }

    dictionary_out = args.out.replace(".pt", "_dictionary.pt")
    torch.save(cluster_dict, dictionary_out)
    print(f"[save dictionary] {dictionary_out}")

    metrics = compute_cluster_metrics(vq_ids, k_req=global_vq_vocab_size)
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
            "ivf_centers": ivf_centers,
            "global_centers": global_centers,
            "k_per_ivf_list": k_per_ivf_list,
            "global_offsets": global_offsets,
            "ivf_counts": ivf_counts,
            "args": vars(args),
            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,
            "unk_token_id": None,
            "vocab_type": "byte_bpe",
            "partitioned": False,
            "partition_type": "global_ivf_then_kmeans",
            "vq_vocab_size": global_vq_vocab_size,
            "id_scheme": "global_ivf_then_local_kmeans",
            "global_id_min": 0,
            "global_id_max": global_vq_vocab_size - 1,
        },
        args.out,
    )

    ids_out = args.out.replace(".pt", "_ids.pt")
    torch.save(
        {
            "vq_ids": vq_ids.to(torch.int32),
            "tgt": tgt.to(torch.int32),
            "ivf_ids": ivf_ids.to(torch.int16),
            "k_per_ivf_list": k_per_ivf_list,
            "global_offsets": global_offsets,
            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,
            "unk_token_id": None,
            "vocab_type": "byte_bpe",
            "partitioned": False,
            "partition_type": "global_ivf_then_kmeans",
            "vq_vocab_size": global_vq_vocab_size,
            "id_scheme": "global_ivf_then_local_kmeans",
            "global_id_min": 0,
            "global_id_max": global_vq_vocab_size - 1,
        },
        ids_out,
    )

    print(f"[save model] {args.out}")
    print(f"[save ids] {ids_out}")


if __name__ == "__main__":
    main()
