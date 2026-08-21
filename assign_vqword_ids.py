#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def make_adj_within_window(length, hop, device):
    idx = torch.arange(length, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    adj = (dist <= int(hop)).float()
    adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj


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
    """Exact tied recurrent GNN architecture used by the shared-HOP checkpoint."""

    def __init__(
        self,
        vocab_size,
        d_model=256,
        hop=10,
        n_layers=1,
        dropout=0.1,
        center_scale=1.0,
    ):
        super().__init__()
        if int(n_layers) != 1:
            raise ValueError("tied recurrent HOP checkpoint requires n_layers=1")
        self.hop = int(hop)
        self.seq_len = 2 * self.hop + 1
        self.center_idx = self.hop
        self.center_scale = float(center_scale)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)
        self.shared_gnn = AdjGNNLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

    def encode_context_hops(self, ctx_ids, requested_hops):
        wanted = sorted({int(h) for h in requested_hops})
        if not wanted:
            return {}
        if wanted[0] < 0 or wanted[-1] > self.hop:
            raise ValueError(f"requested HOPs outside 0..{self.hop}: {wanted}")

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

        adj = make_adj_within_window(length, 1, ctx_ids.device)
        wanted_set = set(wanted)
        states = {}
        if 0 in wanted_set:
            states[0] = F.normalize(h[:, self.center_idx], dim=-1)

        # Stop at the largest requested HOP; no needless recurrent passes.
        for current_hop in range(1, wanted[-1] + 1):
            h = self.shared_gnn(h, adj)
            if current_hop in wanted_set:
                states[current_hop] = F.normalize(
                    h[:, self.center_idx], dim=-1
                )
        return states


def make_windows(token_ids, max_hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (max_hop, max_hop), value=pad_id)
    width = 2 * max_hop + 1
    ctx = torch.stack([padded[i:i + width] for i in range(len(ids))])
    return ctx, ids


def get_hop_row(ckpt, hop):
    hops = [int(h) for h in ckpt.get("hops", [])]
    if not hops:
        raise ValueError("checkpoint is missing non-empty 'hops' metadata")
    try:
        return hops.index(int(hop))
    except ValueError as exc:
        raise ValueError(f"HOP{hop} not present in checkpoint hops={hops}") from exc


def normalize_centers_dict(raw):
    return {int(k): v for k, v in raw.items()}


def build_center_table(
    centers_by_bpe,
    k_by_bpe,
    vocab_size,
    max_local_clusters,
    d_model,
    device,
):
    # One HOP table at a time: ~vocab * K * d_model, avoiding 10x GPU memory.
    table = torch.zeros(
        vocab_size,
        max_local_clusters,
        d_model,
        dtype=torch.float32,
    )
    valid = torch.zeros(
        vocab_size,
        max_local_clusters,
        dtype=torch.bool,
    )

    for bpe_id, centers in centers_by_bpe.items():
        bpe_id = int(bpe_id)
        if not (0 <= bpe_id < vocab_size):
            raise ValueError(f"invalid BPE ID in centers_by_bpe: {bpe_id}")
        if not torch.is_tensor(centers):
            centers = torch.tensor(centers)
        centers = F.normalize(centers.float(), dim=-1)
        k = int(centers.size(0))
        expected_k = int(k_by_bpe[bpe_id].item())
        if k != expected_k:
            raise ValueError(
                f"HOP codebook mismatch for BPE {bpe_id}: "
                f"centers={k}, k_by_bpe={expected_k}"
            )
        if k > max_local_clusters:
            raise ValueError(
                f"BPE {bpe_id}: k={k} exceeds max_local_clusters={max_local_clusters}"
            )
        table[bpe_id, :k] = centers
        valid[bpe_id, :k] = True

    return table.to(device), valid.to(device)


@torch.no_grad()
def assign_one_hop(
    model,
    ctx,
    tgt,
    hop,
    centers_by_bpe,
    k_by_bpe,
    max_local_clusters,
    batch_size,
    device,
    missing_bpe_policy,
):
    model.eval()
    vocab_size = int(k_by_bpe.numel())
    d_model = int(model.tok_emb.embedding_dim)
    center_table, valid_center = build_center_table(
        centers_by_bpe=centers_by_bpe,
        k_by_bpe=k_by_bpe,
        vocab_size=vocab_size,
        max_local_clusters=max_local_clusters,
        d_model=d_model,
        device=device,
    )

    out = torch.empty(len(tgt), dtype=torch.long)
    missing_bpe_ids = set()

    for start in tqdm(
        range(0, len(ctx), batch_size),
        desc=f"[assign HOP{hop}]",
    ):
        end = min(start + batch_size, len(ctx))
        xb = ctx[start:end].to(device)
        yb = tgt[start:end].long().to(device)

        z = model.encode_context_hops(xb, [hop])[hop]
        candidate_centers = center_table[yb]
        candidate_valid = valid_center[yb]
        has_any = candidate_valid.any(dim=1)

        if not has_any.all():
            missing_ids = torch.unique(yb[~has_any]).tolist()
            missing_bpe_ids.update(int(x) for x in missing_ids)
            if missing_bpe_policy == "error":
                raise RuntimeError(
                    f"HOP{hop}: BPEs without centers encountered: {missing_ids[:30]}"
                )

        sim = torch.einsum("bd,bkd->bk", z.float(), candidate_centers)
        sim = sim.masked_fill(~candidate_valid, float("-inf"))
        local_ids = sim.argmax(dim=1)
        # For missing-center BPEs argmax(all -inf) is implementation-defined;
        # force the documented zero fallback explicitly.
        local_ids[~has_any] = 0
        out[start:end] = local_ids.cpu()

    del center_table, valid_center
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out, sorted(missing_bpe_ids)


def render_out_path(pattern, hop):
    # Supports {hop}, {hop02}; the latter keeps filenames lexically ordered.
    return pattern.replace("{hop02}", f"{int(hop):02d}").replace(
        "{hop}", str(int(hop))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hops", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument(
        "--out_pattern",
        required=True,
        help="Output path containing {hop} or {hop02}",
    )
    ap.add_argument(
        "--missing_bpe_policy",
        choices=["zero", "error"],
        default="zero",
    )
    args = ap.parse_args()

    if "{hop}" not in args.out_pattern and "{hop02}" not in args.out_pattern:
        raise ValueError("--out_pattern must contain {hop} or {hop02}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    required = {
        "model",
        "args",
        "centers_by_bpe_by_hop",
        "k_by_bpe_by_hop",
        "hops",
        "max_local_clusters",
    }
    missing = sorted(required - set(ckpt.keys()))
    if missing:
        raise ValueError(f"tied-GNN checkpoint missing keys: {missing}")

    if ckpt.get("shared_gnn_across_hops") is not True:
        raise ValueError("checkpoint is not marked shared_gnn_across_hops=True")
    if ckpt.get("shared_codebook_across_hops") is not False:
        raise ValueError("expected separate codebooks: shared_codebook_across_hops=False")
    if ckpt.get("partition_type") != "bpe_local_kmeans":
        raise ValueError(f"unexpected partition_type={ckpt.get('partition_type')}")

    available_hops = [int(h) for h in ckpt["hops"]]
    requested_hops = [int(h) for h in args.hops]
    if len(set(requested_hops)) != len(requested_hops):
        raise ValueError(f"duplicate HOP in --hops: {requested_hops}")
    for hop in requested_hops:
        if hop not in available_hops:
            raise ValueError(
                f"requested HOP{hop} is absent; checkpoint has {available_hops}"
            )

    cargs = ckpt["args"]
    encoder_max_hop = int(ckpt.get("hop", max(available_hops)))
    if encoder_max_hop != max(available_hops):
        raise ValueError(
            f"checkpoint max-hop mismatch: hop={encoder_max_hop}, hops={available_hops}"
        )

    tokenizer_name = args.tokenizer or ckpt.get("tokenizer_name") or cargs.get("tokenizer")
    if tokenizer_name is None:
        raise ValueError("tokenizer is unavailable")
    print("[tokenizer]", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = int(ckpt["model"]["tok_emb.weight"].shape[0])
    pad_id = ckpt.get("pad_token_id", tokenizer.pad_token_id)
    if pad_id is None:
        raise ValueError("pad_token_id is unavailable")
    pad_id = int(pad_id)
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        unk_id = pad_id
    unk_id = int(unk_id)

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=int(cargs["d_model"]),
        hop=encoder_max_hop,
        n_layers=int(cargs.get("n_layers", 1)),
        center_scale=float(cargs.get("center_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    max_local_clusters = int(ckpt["max_local_clusters"])
    k_all = ckpt["k_by_bpe_by_hop"].long().cpu()
    if k_all.ndim != 2:
        raise ValueError(f"k_by_bpe_by_hop must be rank-2, got {tuple(k_all.shape)}")
    if k_all.size(0) != len(available_hops) or k_all.size(1) != vocab_size:
        raise ValueError(
            "k_by_bpe_by_hop shape mismatch: "
            f"{tuple(k_all.shape)} vs ({len(available_hops)}, {vocab_size})"
        )

    # Tokenize once. Every HOP uses the same physical positions and the same
    # max-HOP bilateral window, exactly as in tied-GNN training.
    print("[data] loading/tokenizing once for all HOPs")
    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split=args.split)
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    all_ctx = []
    all_tgt = []
    samples = []
    cursor = 0
    limit = min(args.max_samples, len(ds))

    for sample_idx, ex in enumerate(tqdm(ds.select(range(limit)), desc="[tokenize]")):
        token_ids = tokenizer.encode(
            ex[args.text_col], add_special_tokens=False
        )[:args.seq_len]
        if len(token_ids) < 2:
            continue
        token_ids = [tid if tid < vocab_size else unk_id for tid in token_ids]
        ctx_i, tgt_i = make_windows(token_ids, encoder_max_hop, pad_id)
        start = cursor
        end = start + len(tgt_i)
        cursor = end
        samples.append(
            {
                "sample_idx": int(sample_idx),
                "start": int(start),
                "end": int(end),
                "length": int(len(tgt_i)),
            }
        )
        all_ctx.append(ctx_i)
        all_tgt.append(tgt_i)

    if not all_ctx:
        raise ValueError("No usable tokenized samples")
    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    print("[data] windows", f"{len(tgt):,}")
    print("[checkpoint hops]", available_hops)
    print("[requested hops]", requested_hops)

    centers_all = ckpt["centers_by_bpe_by_hop"]

    for hop in requested_hops:
        print("=" * 68)
        print(f"[HOP{hop}] assignment")
        row = get_hop_row(ckpt, hop)
        k_by_bpe = k_all[row]
        centers_by_bpe = normalize_centers_dict(centers_all[int(hop)])

        if int(k_by_bpe.max().item()) > max_local_clusters:
            raise ValueError(
                f"HOP{hop}: max k={int(k_by_bpe.max())} > {max_local_clusters}"
            )

        local_ids, missing_bpe_ids = assign_one_hop(
            model=model,
            ctx=ctx,
            tgt=tgt,
            hop=hop,
            centers_by_bpe=centers_by_bpe,
            k_by_bpe=k_by_bpe,
            max_local_clusters=max_local_clusters,
            batch_size=args.batch_size,
            device=device,
            missing_bpe_policy=args.missing_bpe_policy,
        )

        token_k = k_by_bpe[tgt.long()]
        has_centers = token_k > 0
        if has_centers.any():
            bad = local_ids[has_centers] >= token_k[has_centers]
            if bad.any():
                raise RuntimeError(f"HOP{hop}: local VQ ID outside BPE-specific range")
        missing_mask = ~has_centers
        if missing_mask.any() and not torch.all(local_ids[missing_mask] == 0):
            raise RuntimeError(f"HOP{hop}: missing-center BPE did not map to ID 0")

        out_path = Path(render_out_path(args.out_pattern, hop))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "samples": samples,
            "token_ids_flat": tgt.to(torch.int32),
            "offsets": [
                (s["sample_idx"], s["start"], s["end"], s["length"])
                for s in samples
            ],
            "vq_ids_flat": local_ids.to(torch.int16),
            "local_vq_ids_flat": local_ids.to(torch.int16),
            "k_by_bpe": k_by_bpe.to(torch.int16),
            "max_local_clusters": max_local_clusters,
            "vq_vocab_size": max_local_clusters,
            "vq_pad_id": max_local_clusters,
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": ckpt.get("vocab_type", "byte_bpe"),
            "hop": int(hop),
            "available_hops": available_hops,
            "encoder_max_hop": encoder_max_hop,
            "center_scale": float(cargs.get("center_scale", 1.0)),
            "context_type": ckpt.get("context_type", "bilateral"),
            "context_width": int(ckpt.get("context_width", 2 * encoder_max_hop + 1)),
            "ckpt": args.ckpt,
            "tokenizer": tokenizer_name,
            "partitioned": True,
            "partition_type": "bpe_local_kmeans",
            "id_scheme": "(bpe_id, local_vq_id)",
            "shared_gnn_across_hops": True,
            "shared_codebook_across_hops": False,
            "missing_bpe_policy": args.missing_bpe_policy,
            "missing_bpe_ids": missing_bpe_ids,
        }
        print("=== SAVE DEBUG ===")
        print("len(samples):", len(samples))
        print("tgt:", tgt.shape, tgt.dtype,
              tgt.numel() * tgt.element_size() / 1024 ** 3, "GiB")
        print("local_ids:", local_ids.shape, local_ids.dtype,
              local_ids.numel() * local_ids.element_size() / 1024 ** 3, "GiB")

        for k, v in out_data.items():
            if torch.is_tensor(v):
                print(
                    k,
                    tuple(v.shape),
                    v.dtype,
                    f"{v.numel() * v.element_size() / 1024 ** 3:.3f} GiB"
                )
        torch.save(out_data, out_path)

        print("[save]", out_path)
        print("[tokens]", local_ids.numel())
        print("[local VQ min/max]", int(local_ids.min()), int(local_ids.max()))
        print("[missing BPE types]", len(missing_bpe_ids))
        print("[missing BPE token count]", int(missing_mask.sum()))

    print("=" * 68)
    print("[all completed]", requested_hops)


if __name__ == "__main__":
    main()
