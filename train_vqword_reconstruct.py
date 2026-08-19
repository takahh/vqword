#!/usr/bin/env python3
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

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


def make_adj_within_window(seq_len, hop, device):
    pos = torch.arange(seq_len, device=device)
    receiver = pos[:, None]
    sender = pos[None, :]
    distance = (receiver - sender).abs()
    adj = (distance <= hop).float()
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
        n_layers=1,
        dropout=0.1,
        center_scale=1.0,
    ):
        super().__init__()
        if int(n_layers) != 1:
            raise ValueError(
                "shared recurrent HOP mode requires --n_layers 1: "
                "one application of the tied GNN cell must equal one HOP"
            )
        self.hop = int(hop)

        # 両側hop個 + 現在位置
        self.seq_len = 2 * hop + 1
        self.center_idx = hop
        self.center_scale = center_scale

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)
        # A single parameter-tied message-passing cell is reused for every
        # HOP.  This is what keeps HOP1..HOP10 in one learned coordinate
        # system instead of creating an independent encoder per HOP.
        self.shared_gnn = AdjGNNLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

    def encode_context_hops(self, ctx_ids, requested_hops=None):
        batch_size, length = ctx_ids.shape
        if length != self.seq_len:
            raise ValueError(
                f"expected context width {self.seq_len}, got {length}"
            )
        pos = torch.arange(length, device=ctx_ids.device)
        pos = pos.unsqueeze(0).expand(batch_size, length)

        tok_h = self.tok_emb(ctx_ids)
        tok_h[:, self.center_idx, :] *= self.center_scale

        h = self.dropout(tok_h + self.pos_emb(pos))
        # Only immediate neighbours are connected. Reusing the same cell once
        # expands the receptive field by exactly one token on each side.
        adj = make_adj_within_window(length, 1, ctx_ids.device)
        wanted = (
            set(range(self.hop + 1))
            if requested_hops is None else
            {int(x) for x in requested_hops}
        )
        if min(wanted, default=0) < 0 or max(wanted, default=0) > self.hop:
            raise ValueError(f"requested HOPs outside 0..{self.hop}: {wanted}")

        states = {}
        if 0 in wanted:
            states[0] = F.normalize(h[:, self.center_idx], dim=-1)
        for current_hop in range(1, self.hop + 1):
            h = self.shared_gnn(h, adj)
            if current_hop in wanted:
                states[current_hop] = F.normalize(
                    h[:, self.center_idx], dim=-1
                )
        return states

    def encode_context(self, ctx_ids, hop=None):
        selected_hop = self.hop if hop is None else int(hop)
        return self.encode_context_hops(ctx_ids, [selected_hop])[selected_hop]

    def forward(self, ctx_ids, target_ids):
        z = self.encode_context(ctx_ids)
        logits = self.decoder(z)
        loss = F.cross_entropy(logits, target_ids)
        return loss, logits, z


@torch.no_grad()
def fit_bpe_local_kmeans_shared_hops(
    model,
    ctx,
    tgt,
    hops,
    batch_size,
    device,
    max_clusters=5,
    seed=0,
    vocab_size=None,
    sample_positions_per_bpe=256,
):
    """Fit one BPE-local codebook jointly over all requested HOP states.

    The GNN cell, latent axes, projection (identity here), and cluster centers
    are shared. HOP is not part of the token ID. ID assignment is performed
    separately, one HOP at a time, so completed HOPs can be checkpointed.
    """
    hops = [int(h) for h in hops]
    if not hops or len(set(hops)) != len(hops):
        raise ValueError(f"invalid HOP list: {hops}")
    if vocab_size is None:
        vocab_size = int(tgt.max().item()) + 1
    vocab_size = int(vocab_size)
    rng = torch.Generator().manual_seed(int(seed))
    centers_by_bpe = {}
    k_by_bpe = torch.zeros(vocab_size, dtype=torch.long)

    # Fit each shared local codebook from the same sampled physical positions
    # observed at every requested HOP. This avoids retaining N*H latent vectors.
    for bpe_tensor in tqdm(torch.unique(tgt), desc="[shared-HOP BPE-local KMeans]"):
        bpe_id = int(bpe_tensor)
        indices = torch.where(tgt == bpe_id)[0]
        if len(indices) > int(sample_positions_per_bpe):
            perm = torch.randperm(len(indices), generator=rng)
            indices = indices[perm[:int(sample_positions_per_bpe)]]

        pieces = []
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            states = model.encode_context_hops(ctx[idx].to(device), hops)
            pieces.extend(states[h].cpu() for h in hops)
        x = F.normalize(torch.cat(pieces, dim=0).float(), dim=-1)
        k = min(int(max_clusters), x.size(0))
        k_by_bpe[bpe_id] = k
        if k == 1:
            centers_by_bpe[bpe_id] = F.normalize(
                x.mean(dim=0, keepdim=True), dim=-1
            )
            continue
        km = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            batch_size=min(4096, max(k, x.size(0))),
            random_state=seed,
            reassignment_ratio=0.01,
        )
        km.fit(x.numpy().astype(np.float32, copy=False))
        centers_by_bpe[bpe_id] = F.normalize(
            torch.from_numpy(km.cluster_centers_).float(), dim=-1
        )

    return centers_by_bpe, k_by_bpe


def atomic_torch_save(obj, path):
    """Write a PyTorch file atomically to avoid accepting partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


@torch.no_grad()
def assign_bpe_local_ids_one_hop(
    model,
    ctx,
    tgt,
    hop,
    centers_by_bpe,
    k_by_bpe,
    batch_size,
    device,
    max_clusters,
    vocab_size,
):
    """Assign every physical position for exactly one recurrent HOP."""
    if not centers_by_bpe:
        raise ValueError("centers_by_bpe is empty")

    d_model = next(iter(centers_by_bpe.values())).size(1)
    center_table = torch.zeros(
        int(vocab_size), int(max_clusters), d_model, dtype=torch.float32
    )
    valid_center = torch.zeros(
        int(vocab_size), int(max_clusters), dtype=torch.bool
    )
    for bpe_id, centers in centers_by_bpe.items():
        k = centers.size(0)
        center_table[int(bpe_id), :k] = centers
        valid_center[int(bpe_id), :k] = True
    center_table = center_table.to(device)
    valid_center = valid_center.to(device)

    ids = torch.empty(len(tgt), dtype=torch.long)
    for start in tqdm(
        range(0, len(ctx), batch_size), desc=f"[assign HOP{int(hop)} IDs]"
    ):
        end = min(start + batch_size, len(ctx))
        yb = tgt[start:end].to(device)
        z = model.encode_context_hops(
            ctx[start:end].to(device), [int(hop)]
        )[int(hop)]
        candidate_centers = center_table[yb]
        candidate_valid = valid_center[yb]
        z = F.normalize(z.float(), dim=-1)
        sim = torch.einsum("bd,bkd->bk", z, candidate_centers)
        sim = sim.masked_fill(~candidate_valid, float("-inf"))
        ids[start:end] = sim.argmax(dim=1).cpu()

    return ids

@torch.no_grad()
def fit_bpe_local_kmeans(
    model,
    ctx,
    tgt,
    batch_size,
    device,
    max_clusters=5,
    seed=0,
    vocab_size=None,
):
    """
    Cluster latent vectors independently inside each BPE token.

    Returns
    -------
    local_vq_ids:
        [N] tensor. Local cluster ID within each BPE, 0..K_b-1.

    centers_by_bpe:
        dict[bpe_id] -> [K_b, d_model] tensor

    k_by_bpe:
        [vocab_size] tensor containing number of local clusters.
    """

    # ---------------------------------------------------------
    # 1. Encode all positions
    # ---------------------------------------------------------
    all_z = []

    for start in tqdm(
        range(0, len(ctx), batch_size),
        desc="[encode for BPE-local clustering]",
    ):
        z, _ = encode_batch(
            model,
            ctx,
            start,
            batch_size,
            device,
        )
        all_z.append(z.cpu())

    z_all = torch.cat(all_z, dim=0)
    z_all = F.normalize(z_all.float(), dim=-1)

    local_vq_ids = torch.zeros(
        len(tgt),
        dtype=torch.long,
    )

    centers_by_bpe = {}
    if vocab_size is None:
        vocab_size = int(tgt.max().item()) + 1
    k_by_bpe = torch.zeros(int(vocab_size), dtype=torch.long)

    # ---------------------------------------------------------
    # 2. Partition by BPE ID
    # ---------------------------------------------------------
    unique_bpes = torch.unique(tgt)

    for bpe_id_tensor in tqdm(
        unique_bpes,
        desc="[BPE-local KMeans]",
    ):
        bpe_id = int(bpe_id_tensor.item())

        indices = torch.where(tgt == bpe_id)[0]
        x = z_all[indices]

        n = x.size(0)
        k = min(int(max_clusters), n)

        k_by_bpe[bpe_id] = k

        # singleton BPE
        if k == 1:
            center = F.normalize(
                x.mean(dim=0, keepdim=True),
                dim=-1,
            )

            centers_by_bpe[bpe_id] = center
            local_vq_ids[indices] = 0
            continue

        km = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            batch_size=min(4096, max(k, n)),
            random_state=seed,
            reassignment_ratio=0.01,
        )

        labels = km.fit_predict(
            x.numpy().astype(np.float32, copy=False)
        )

        centers = torch.from_numpy(
            km.cluster_centers_
        ).float()

        centers = F.normalize(
            centers,
            dim=-1,
        )

        centers_by_bpe[bpe_id] = centers

        local_vq_ids[indices] = torch.from_numpy(
            labels
        ).long()

    return (
        local_vq_ids,
        centers_by_bpe,
        k_by_bpe,
    )


def iter_index_batches(n, batch_size, shuffle, device=None):
    if shuffle:
        order = torch.randperm(n)
    else:
        order = torch.arange(n)

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if device is not None:
            idx = idx.to(device)
        yield idx


@torch.no_grad()
def evaluate_discrete_decoder(
    decoder,
    centers,
    vq_ids,
    tgt,
    batch_size,
    device,
    max_items=None,
):
    decoder.eval()

    n = len(tgt)
    if max_items is not None:
        n = min(n, int(max_items))

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_count = 0

    centers = centers.to(device)

    for idx in iter_index_batches(n, batch_size, shuffle=False):
        ids = vq_ids[idx].to(device)
        yb = tgt[idx].to(device)

        q = centers[ids]
        logits = decoder(q)
        loss = F.cross_entropy(logits, yb, reduction="sum")

        k = min(5, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices

        total_loss += float(loss.item())
        total_top1 += int(topk[:, 0].eq(yb).sum().item())
        total_top5 += int(topk.eq(yb[:, None]).any(dim=1).sum().item())
        total_count += yb.numel()

    mean_loss = total_loss / max(total_count, 1)

    return {
        "loss": mean_loss,
        "ppl": float(np.exp(min(mean_loss, 20.0))),
        "top1": total_top1 / max(total_count, 1),
        "top5": total_top5 / max(total_count, 1),
        "count": total_count,
    }


def train_discrete_decoder(
    decoder,
    centers,
    vq_ids,
    tgt,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
    eval_size,
):
    """
    Train VQ center -> BPE decoder after clustering.

    Only the lightweight decoder is optimized. The GNN and clustered centers
    remain fixed. This directly measures how much physical-token information
    center-aware clustering retained.
    """
    epochs = int(epochs)

    if epochs <= 0:
        print("[discrete decoder] skipped")
        return []

    decoder.train()
    decoder.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    centers_device = centers.to(device)
    history = []

    for epoch in range(1, epochs + 1):
        decoder.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        pbar = tqdm(
            iter_index_batches(
                len(tgt),
                batch_size,
                shuffle=True,
            ),
            total=(len(tgt) + batch_size - 1) // batch_size,
            desc=f"[discrete decoder] epoch {epoch}/{epochs}",
        )

        for idx in pbar:
            ids = vq_ids[idx].to(device)
            yb = tgt[idx].to(device)

            q = centers_device[ids]

            optimizer.zero_grad(set_to_none=True)

            logits = decoder(q)
            loss = F.cross_entropy(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            count = yb.numel()
            total_loss += float(loss.item()) * count
            total_correct += int(logits.argmax(dim=-1).eq(yb).sum().item())
            total_count += count

            mean_loss = total_loss / max(total_count, 1)
            pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                ppl=f"{np.exp(min(mean_loss, 20.0)):.2f}",
                acc=f"{total_correct / max(total_count, 1):.4f}",
            )

        metrics = evaluate_discrete_decoder(
            decoder=decoder,
            centers=centers,
            vq_ids=vq_ids,
            tgt=tgt,
            batch_size=batch_size,
            device=device,
            max_items=eval_size,
        )

        history.append({
            "epoch": epoch,
            **metrics,
        })

        print(
            f"[discrete decoder eval] "
            f"epoch={epoch} "
            f"loss={metrics['loss']:.6f} "
            f"ppl={metrics['ppl']:.4f} "
            f"top1={metrics['top1']:.4f} "
            f"top5={metrics['top5']:.4f} "
            f"N={metrics['count']:,}"
        )

    return history


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
    """Create bilateral windows [i-hop, ..., i, ..., i+hop].

    The target is always the physical BPE token at the center position.
    Padding is applied independently to the left and right document edges.
    """
    ids = torch.tensor(token_ids, dtype=torch.long)

    # 両側をpaddingする
    padded = F.pad(ids, (hop, hop), value=pad_id)

    ctx = []
    tgt = []
    width = 2 * hop + 1

    for i in range(len(ids)):
        # 左hop個 + 現在位置 + 右hop個
        ctx.append(padded[i:i + width])
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


def allocate_k_per_ivf_list(
    ivf_counts: torch.Tensor,
    requested_k: int,
) -> torch.Tensor:
    """
    各IVFリストに割り当てるfine center数を決める。

    方針:
      - すべてのIVFリストに最低1個割り当てる
      - 残りを、各リストのデータ数に比例して配分する
      - 合計はrequested_kに厳密に一致させる
    """
    ivf_counts = ivf_counts.to(dtype=torch.long)
    requested_k = int(requested_k)

    nlist = int(ivf_counts.numel())

    if requested_k < nlist:
        raise ValueError(
            f"requested_k={requested_k} is smaller than "
            f"number of IVF lists={nlist}. "
            "Cannot assign at least one fine center to every IVF list."
        )

    device = ivf_counts.device

    # すべてのIVFリストに最低1個
    k_per_list = torch.ones(
        nlist,
        dtype=torch.long,
        device=device,
    )

    remaining = requested_k - nlist

    if remaining == 0:
        return k_per_list

    weights = ivf_counts.to(dtype=torch.float64)

    # 念のため、全カウントが0の場合にも対応
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)

    # 残りを点数に比例配分
    raw_extra = weights / weights.sum() * remaining
    extra = torch.floor(raw_extra).to(dtype=torch.long)

    k_per_list += extra

    # floorにより余ったcenterを小数部分が大きい順に配る
    leftover = requested_k - int(k_per_list.sum().item())

    if leftover > 0:
        fractional = raw_extra - extra.to(dtype=raw_extra.dtype)
        order = torch.argsort(fractional, descending=True)
        k_per_list[order[:leftover]] += 1

    # 最終チェック
    actual_total = int(k_per_list.sum().item())

    if actual_total != requested_k:
        raise RuntimeError(
            f"k allocation mismatch: "
            f"actual={actual_total}, requested={requested_k}"
        )

    if int(k_per_list.min().item()) < 1:
        zero_lists = torch.where(k_per_list < 1)[0]
        raise RuntimeError(
            "Some IVF lists received no fine centers: "
            f"{zero_lists[:50].tolist()}"
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
    """
    Collect initial fine centers from each IVF list.

    実データ点を優先して使い、必要数に足りないIVFリストについては、
    対応するcoarse IVF centerで不足分を埋める。
    """
    nlist = ivf_centers.size(0)
    d_model = ivf_centers.size(1)

    # CPU上で扱う型を明示的に統一
    k_per_list = k_per_list.to(
        device="cpu",
        dtype=torch.long,
    )

    if k_per_list.numel() != nlist:
        raise ValueError(
            f"k_per_list size mismatch: "
            f"{k_per_list.numel()} != nlist={nlist}"
        )

    if torch.any(k_per_list <= 0):
        bad = torch.where(k_per_list <= 0)[0]
        raise ValueError(
            "Every IVF list must receive at least one fine center. "
            f"Bad lists: {bad[:20].tolist()}"
        )

    offsets = torch.zeros(
        nlist + 1,
        dtype=torch.long,
    )
    offsets[1:] = torch.cumsum(k_per_list, dim=0)

    total_k = int(offsets[-1].item())

    initial = torch.empty(
        (total_k, d_model),
        dtype=torch.float32,
    )

    filled = torch.zeros(
        nlist,
        dtype=torch.long,
    )

    # coarseはassignment用としてdevice上に置く
    coarse = F.normalize(
        ivf_centers.to(
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    for start in tqdm(
        range(0, len(ctx), batch_size),
        desc="[fine init]",
    ):
        z, _ = encode_batch(
            model,
            ctx,
            start,
            batch_size,
            device,
        )

        z = F.normalize(
            z.to(dtype=torch.float32),
            dim=-1,
        )

        ivf_ids = assign_blockwise(
            z,
            coarse,
            k_block=k_block,
        ).cpu()

        z_cpu = z.cpu()

        for list_id in torch.unique(ivf_ids).tolist():
            list_id = int(list_id)

            need = int(
                k_per_list[list_id].item()
                - filled[list_id].item()
            )

            if need <= 0:
                continue

            candidates = z_cpu[ivf_ids == list_id]

            take = min(
                need,
                int(candidates.size(0)),
            )

            if take <= 0:
                continue

            begin = int(
                offsets[list_id].item()
                + filled[list_id].item()
            )

            initial[begin : begin + take] = candidates[:take]
            filled[list_id] += take

        if torch.equal(filled, k_per_list):
            break

    # --------------------------------------------------------
    # 不足分を対応するcoarse IVF centerで埋める
    # --------------------------------------------------------
    coarse_cpu = coarse.cpu()

    fallback_lists = torch.where(
        filled < k_per_list
    )[0]

    fallback_center_count = 0

    for list_id_tensor in fallback_lists:
        list_id = int(list_id_tensor.item())

        begin = int(
            offsets[list_id].item()
            + filled[list_id].item()
        )
        end = int(offsets[list_id + 1].item())

        missing = end - begin

        if missing <= 0:
            continue

        initial[begin:end] = coarse_cpu[list_id].unsqueeze(0).expand(
            missing,
            -1,
        )

        filled[list_id] += missing
        fallback_center_count += missing

    # --------------------------------------------------------
    # 最終検査
    # --------------------------------------------------------
    if not torch.equal(filled, k_per_list):
        bad = torch.where(
            filled != k_per_list
        )[0]

        details = [
            {
                "list": int(i),
                "filled": int(filled[i]),
                "required": int(k_per_list[i]),
            }
            for i in bad[:20].tolist()
        ]

        raise RuntimeError(
            "Failed to initialize fine centers: "
            f"{details}"
        )

    sizes_from_offsets = offsets[1:] - offsets[:-1]

    if not torch.equal(
        sizes_from_offsets,
        k_per_list,
    ):
        bad = torch.where(
            sizes_from_offsets != k_per_list
        )[0]

        raise RuntimeError(
            "offsets do not match k_per_list: "
            f"{bad[:20].tolist()}"
        )

    empty_lists = torch.where(
        sizes_from_offsets == 0
    )[0]

    if empty_lists.numel() > 0:
        raise RuntimeError(
            "Some IVF lists have zero fine centers: "
            f"{empty_lists[:20].tolist()}"
        )

    print(
        f"[fine init] total_centers={total_k} "
        f"fallback_lists={int(fallback_lists.numel())} "
        f"fallback_centers={fallback_center_count}"
    )

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

    zero_lists = torch.where(k_per_list == 0)[0]
    nonzero = k_per_list[k_per_list > 0]

    print(
        f"[stage 2 allocation] total={int(k_per_list.sum())} "
        f"min={int(nonzero.min())} "
        f"mean={nonzero.float().mean().item():.2f} "
        f"max={int(nonzero.max())} "
        f"zero_lists={int(zero_lists.numel())}"
    )

    if zero_lists.numel() > 0:
        raise RuntimeError(
            "Allocation produced zero-center IVF lists: "
            f"{zero_lists[:50].tolist()}"
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

    expected_total = int(k_per_list.sum().item())
    actual_total = int(fine_centers.size(0))

    if actual_total != expected_total:
        raise RuntimeError(
            f"Fine center count mismatch: "
            f"actual={actual_total}, expected={expected_total}"
        )

    offset_sizes = offsets[1:] - offsets[:-1]

    if not torch.equal(offset_sizes.cpu(), k_per_list.cpu()):
        bad = torch.where(
            offset_sizes.cpu() != k_per_list.cpu()
        )[0]

        raise RuntimeError(
            "global_offsets and k_per_list mismatch: "
            f"{bad[:20].tolist()}"
        )

    empty_lists = torch.where(offset_sizes <= 0)[0]

    if empty_lists.numel() > 0:
        raise RuntimeError(
            "IVF lists without fine centers remain: "
            f"{empty_lists[:20].tolist()}"
        )

    print(
        f"[stage 2 validation] "
        f"fine_centers={actual_total} "
        f"empty_lists=0"
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
    ap.add_argument(
        "--local_clusters",
        type=int,
        default=5,
        help="maximum number of latent clusters within each BPE token",
    )
    # Data
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument(
        "--all_hops",
        action="store_true",
        help="use one tied GNN and one shared codebook for min_hop..max_hop",
    )
    ap.add_argument("--min_hop", type=int, default=1)
    ap.add_argument("--max_hop", type=int, default=10)

    # Model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument(
        "--n_layers", type=int, default=1,
        help="must be 1: one tied GNN-cell application equals one HOP",
    )
    ap.add_argument("--center_scale", type=float, default=1.0)

    ap.add_argument(
        "--decoder_epochs",
        type=int,
        default=3,
        help="epochs for fixed-center VQ-to-BPE decoder training",
    )
    ap.add_argument(
        "--decoder_lr",
        type=float,
        default=1e-3,
    )
    ap.add_argument(
        "--decoder_weight_decay",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--decoder_eval_size",
        type=int,
        default=100000,
        help="maximum number of windows used for discrete decoder evaluation",
    )

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
    ap.add_argument(
        "--cluster_samples_per_bpe",
        type=int,
        default=256,
        help="physical positions sampled per BPE for shared-HOP KMeans",
    )
    ap.add_argument("--k_block", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vqword_global_ivf.pt")
    ap.add_argument(
        "--hop_parts_dir",
        default=None,
        help=(
            "directory for per-HOP checkpoints; default: "
            "<out_without_.pt>_hop_parts"
        ),
    )
    ap.add_argument(
        "--cleanup_hop_parts",
        action="store_true",
        help="remove per-HOP ID files after the final IDs file is saved",
    )

    args = ap.parse_args()

    if args.n_layers != 1:
        raise ValueError("set --n_layers 1 for exact tied recurrent HOPs")
    if args.all_hops:
        if args.min_hop < 0 or args.max_hop < args.min_hop:
            raise ValueError(
                f"invalid hop range: min_hop={args.min_hop}, "
                f"max_hop={args.max_hop}"
            )
        hops = list(range(args.min_hop, args.max_hop + 1))
    else:
        if args.hop < 0:
            raise ValueError("--hop must be non-negative")
        hops = [int(args.hop)]
    encoder_max_hop = max(hops)

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
        ctx, tgt = make_windows(ids, encoder_max_hop, pad_id)
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
        hop=encoder_max_hop,
        n_layers=args.n_layers,
        center_scale=args.center_scale,
    ).to(device)

    # Keep the GNN encoder unchanged.
    # With center_scale > 0, the current BPE embedding is already included
    # in the representation before clustering.
    model.eval()

    print(
        f"[shared HOP] hops={hops} tied_gnn=True "
        f"shared_codebook=True hop_embedding=False"
    )
    out_path = Path(args.out)
    hop_parts_dir = Path(
        args.hop_parts_dir
        if args.hop_parts_dir is not None else
        str(out_path.with_suffix("")) + "_hop_parts"
    )
    hop_parts_dir.mkdir(parents=True, exist_ok=True)
    shared_checkpoint = hop_parts_dir / "shared_codebook.pt"

    checkpoint_signature = {
        "hops": hops,
        "seed": int(args.seed),
        "local_clusters": int(args.local_clusters),
        "cluster_samples_per_bpe": int(args.cluster_samples_per_bpe),
        "d_model": int(args.d_model),
        "center_scale": float(args.center_scale),
        "tokenizer_name": args.tokenizer,
        "vocab_size": int(vocab_size),
        "num_positions": int(len(tgt)),
        "encoder_max_hop": int(encoder_max_hop),
    }

    if shared_checkpoint.exists():
        saved = torch.load(shared_checkpoint, map_location="cpu")
        if saved.get("signature") != checkpoint_signature:
            raise RuntimeError(
                f"checkpoint settings do not match current run: "
                f"{shared_checkpoint}. Remove that directory or choose a "
                "different --hop_parts_dir."
            )
        model.load_state_dict(saved["model"])
        centers_by_bpe = saved["centers_by_bpe"]
        k_by_bpe = saved["k_by_bpe"]
        print(f"[resume shared codebook] {shared_checkpoint}")
    else:
        centers_by_bpe, k_by_bpe = fit_bpe_local_kmeans_shared_hops(
            model=model,
            ctx=ctx,
            tgt=tgt,
            hops=hops,
            batch_size=args.batch_size,
            device=device,
            max_clusters=args.local_clusters,
            seed=args.seed,
            vocab_size=vocab_size,
            sample_positions_per_bpe=args.cluster_samples_per_bpe,
        )
        atomic_torch_save(
            {
                "signature": checkpoint_signature,
                "model": model.state_dict(),
                "centers_by_bpe": centers_by_bpe,
                "k_by_bpe": k_by_bpe,
            },
            shared_checkpoint,
        )
        print(f"[save shared codebook] {shared_checkpoint}")

    # Assign and persist one HOP at a time. A completed part is reused after
    # interruption, while atomic writes prevent a partial part being accepted.
    hop_part_paths = []
    id_dtype = torch.uint8 if args.local_clusters <= 256 else torch.int16
    for hop in hops:
        part_path = hop_parts_dir / f"hop_{int(hop):03d}_ids.pt"
        hop_part_paths.append(part_path)
        if part_path.exists():
            part = torch.load(part_path, map_location="cpu")
            if (
                part.get("signature") != checkpoint_signature
                or int(part.get("hop", -1)) != int(hop)
                or len(part.get("local_vq_ids", [])) != len(tgt)
            ):
                raise RuntimeError(
                    f"invalid or mismatched HOP checkpoint: {part_path}"
                )
            print(f"[resume HOP{hop}] {part_path}")
            continue

        hop_ids = assign_bpe_local_ids_one_hop(
            model=model,
            ctx=ctx,
            tgt=tgt,
            hop=hop,
            centers_by_bpe=centers_by_bpe,
            k_by_bpe=k_by_bpe,
            batch_size=args.batch_size,
            device=device,
            max_clusters=args.local_clusters,
            vocab_size=vocab_size,
        )
        atomic_torch_save(
            {
                "signature": checkpoint_signature,
                "hop": int(hop),
                "local_vq_ids": hop_ids.to(id_dtype),
            },
            part_path,
        )
        print(f"[save HOP{hop}] {part_path}")
        del hop_ids

    print("[merge HOP parts]")
    local_vq_ids_by_hop = torch.stack([
        torch.load(path, map_location="cpu")["local_vq_ids"]
        for path in hop_part_paths
    ], dim=0)

    # The pair itself is the token: no lossy VQW -> BPE decoder is needed.
    # local_vq_id only has meaning inside its accompanying BPE partition.
    pair_counts = torch.zeros(
        (vocab_size, args.local_clusters), dtype=torch.int64
    )
    repeated_bpe = tgt.long().repeat(len(hops))
    pair_counts.index_put_(
        (repeated_bpe, local_vq_ids_by_hop.reshape(-1).long()),
        torch.ones_like(repeated_bpe, dtype=torch.int64),
        accumulate=True,
    )
    observed_bpes = k_by_bpe > 0
    used_pairs = int((pair_counts > 0).sum().item())
    print(
        f"[BPE-local clustering] observed_bpes={int(observed_bpes.sum())}/"
        f"{vocab_size} used_pairs={used_pairs} "
        f"max_local_clusters={args.local_clusters}"
    )

    dictionary = {
        "centers_by_bpe": centers_by_bpe,
        "k_by_bpe": k_by_bpe,
        "pair_counts": pair_counts,
        "bpe_id_to_token": [
            tok.convert_ids_to_tokens(i) for i in range(vocab_size)
        ],
        "token_vocab_size": vocab_size,
        "tokenizer_name": args.tokenizer,
        "pad_token_id": pad_id,
        "unk_token_id": tok.unk_token_id,
        "partitioned": True,
        "partition_type": "bpe_local_kmeans",
        "max_local_clusters": int(args.local_clusters),
        "id_scheme": "(bpe_id, local_vq_id)",
        "reconstruction": "exact_from_bpe_id",
        "context_type": "bilateral",
        "hop": int(encoder_max_hop),
        "hops": hops,
        "hop_axis": 0,
        "context_width": int(2 * encoder_max_hop + 1),
        "shared_gnn_across_hops": True,
        "shared_codebook_across_hops": True,
        "hop_embedding": False,
    }

    dictionary_out = args.out.replace(".pt", "_dictionary.pt")
    atomic_torch_save(dictionary, dictionary_out)
    print(f"[save dictionary] {dictionary_out}")
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "centers_by_bpe": centers_by_bpe,
            "k_by_bpe": k_by_bpe,

            "args": vars(args),
            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,

            "vocab_type": "byte_bpe",
            "partitioned": True,
            "partition_type": "bpe_local_kmeans",

            "max_local_clusters": int(args.local_clusters),

            "context_type": "bilateral",
            "hop": int(encoder_max_hop),
            "hops": hops,
            "context_width": int(2 * encoder_max_hop + 1),
            "shared_gnn_across_hops": True,
            "shared_codebook_across_hops": True,
            "hop_embedding": False,

            "id_scheme": "(bpe_id, local_vq_id)",
        },
        args.out,
    )
    ids_out = args.out.replace(".pt", "_ids.pt")
    atomic_torch_save(
        {
            "bpe_ids": tgt.to(torch.int32),
            "local_vq_ids_by_hop": local_vq_ids_by_hop.to(
                torch.uint8 if args.local_clusters <= 256 else torch.int16
            ),
            # Backward-compatible alias: the final row is the largest HOP.
            "local_vq_ids": local_vq_ids_by_hop[-1].to(
                torch.uint8 if args.local_clusters <= 256 else torch.int16
            ),
            "k_by_bpe": k_by_bpe,

            "tokenizer_name": args.tokenizer,
            "pad_token_id": pad_id,

            "partitioned": True,
            "partition_type": "bpe_local_kmeans",
            "max_local_clusters": args.local_clusters,

            "id_scheme": "(bpe_id, local_vq_id)",
            "hop": int(encoder_max_hop),
            "hops": hops,
            "hop_axis": 0,
            "hop_to_row": {int(h): i for i, h in enumerate(hops)},
            "context_width": int(2 * encoder_max_hop + 1),
            "shared_gnn_across_hops": True,
            "shared_codebook_across_hops": True,
            "hop_embedding": False,

            "bpe_id_to_token": [
                tok.convert_ids_to_tokens(i)
                for i in range(vocab_size)
            ],
        },
        ids_out,
    )

    print(f"[save model] {args.out}")
    print(f"[save ids] {ids_out}")

    if args.cleanup_hop_parts:
        for path in hop_part_paths:
            path.unlink(missing_ok=True)
        print(f"[cleanup HOP parts] {hop_parts_dir}")


if __name__ == "__main__":
    main()