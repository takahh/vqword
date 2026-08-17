#!/usr/bin/env python3
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from train_vqword import VQWordGNN, make_windows


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


@torch.no_grad()
def assign_ids_bpe_local(
    model,
    ctx,
    tgt,
    centers_by_bpe,
    k_by_bpe,
    batch_size,
    device,
    k_block,
    missing_bpe_policy="zero",
):
    """
    Assign a local VQ ID within each target BPE partition.

    Output:
      local_vq_ids[i] is the local cluster ID for token tgt[i].
      missing_bpe_ids contains BPE IDs that had no learned local centers.

    missing_bpe_policy:
      "zero": assign local_vq_id=0 when a BPE has no centers.
      "error": raise immediately.
    """
    model.eval()

    local_vq_ids = torch.empty(len(ctx), dtype=torch.long)
    missing_bpe_ids = set()

    for start in tqdm(
        range(0, len(ctx), batch_size),
        desc="[assign BPE-local]",
    ):
        end = min(start + batch_size, len(ctx))

        xb = ctx[start:end].to(device)
        tb = tgt[start:end].long().to(device)

        z = F.normalize(
            model.encode_context(xb).float(),
            dim=-1,
        )

        batch_local_ids = torch.empty(
            z.size(0),
            dtype=torch.long,
            device=device,
        )

        for bpe_id in torch.unique(tb).tolist():
            bpe_id = int(bpe_id)
            mask = tb == bpe_id

            centers = centers_by_bpe.get(bpe_id)
            expected_k = (
                int(k_by_bpe[bpe_id].item())
                if 0 <= bpe_id < k_by_bpe.numel()
                else 0
            )

            if centers is None or expected_k <= 0:
                if missing_bpe_policy == "error":
                    raise RuntimeError(
                        f"No local centers for BPE ID {bpe_id}: "
                        f"k_by_bpe={expected_k}"
                    )

                missing_bpe_ids.add(bpe_id)
                batch_local_ids[mask] = 0
                continue

            if not torch.is_tensor(centers):
                centers = torch.tensor(centers)

            if centers.ndim != 2:
                raise RuntimeError(
                    f"Invalid centers for BPE ID {bpe_id}: "
                    f"shape={tuple(centers.shape)}"
                )

            actual_k = int(centers.size(0))
            if actual_k != expected_k:
                raise RuntimeError(
                    f"k mismatch for BPE ID {bpe_id}: "
                    f"k_by_bpe={expected_k}, centers={actual_k}"
                )

            centers = F.normalize(
                centers.to(device).float(),
                dim=-1,
            )

            local_ids = assign_blockwise(
                z[mask],
                centers,
                k_block=k_block,
            )

            if local_ids.numel() and int(local_ids.max()) >= actual_k:
                raise RuntimeError(
                    f"Local ID out of range for BPE ID {bpe_id}: "
                    f"max={int(local_ids.max())}, k={actual_k}"
                )

            batch_local_ids[mask] = local_ids

        local_vq_ids[start:end] = batch_local_ids.cpu()

    return local_vq_ids, sorted(missing_bpe_ids)


@torch.no_grad()
def assign_ids_global_ivf(
    model,
    ctx,
    ivf_centers,
    global_centers,
    global_offsets,
    batch_size,
    device,
    k_block,
):
    model.eval()

    coarse = F.normalize(ivf_centers.to(device).float(), dim=-1)
    fine = F.normalize(global_centers.to(device).float(), dim=-1)
    offsets = global_offsets.long().cpu()

    vq_ids = torch.empty(len(ctx), dtype=torch.long)
    ivf_ids_all = torch.empty(len(ctx), dtype=torch.long)

    for start in tqdm(
        range(0, len(ctx), batch_size),
        desc="[assign global IVF]",
    ):
        end = min(start + batch_size, len(ctx))
        xb = ctx[start:end].to(device)

        z = F.normalize(model.encode_context(xb).float(), dim=-1)
        ivf_ids = assign_blockwise(z, coarse, k_block=k_block)

        batch_global_ids = torch.empty(
            z.size(0),
            dtype=torch.long,
            device=device,
        )

        for list_id in torch.unique(ivf_ids).tolist():
            mask = ivf_ids == list_id
            begin = int(offsets[list_id].item())
            finish = int(offsets[list_id + 1].item())

            if finish <= begin:
                raise RuntimeError(
                    f"IVF list {list_id} has no fine centers: "
                    f"offsets=({begin}, {finish})"
                )

            local_ids = assign_blockwise(
                z[mask],
                fine[begin:finish],
                k_block=k_block,
            )
            batch_global_ids[mask] = local_ids + begin

        vq_ids[start:end] = batch_global_ids.cpu()
        ivf_ids_all[start:end] = ivf_ids.cpu()

    return vq_ids, ivf_ids_all


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
    ap.add_argument("--k_block", type=int, default=4096)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="vqword_ids.pt")
    ap.add_argument(
        "--missing_bpe_policy",
        choices=["zero", "error"],
        default="zero",
        help=(
            "For BPE-local checkpoints, what to do when a target BPE has no "
            "learned local centers. Default: zero."
        ),
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    ckpt = torch.load(
        args.ckpt,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in ckpt or "args" not in ckpt:
        raise ValueError("Checkpoint must contain 'model' and 'args'")

    cargs = ckpt["args"]

    # --------------------------------------------------------
    # Detect checkpoint type
    # --------------------------------------------------------
    is_bpe_local = (
        ckpt.get("partition_type") == "bpe_local_kmeans"
        and "centers_by_bpe" in ckpt
        and "k_by_bpe" in ckpt
    )

    is_global_ivf = all(
        key in ckpt
        for key in (
            "ivf_centers",
            "global_centers",
            "global_offsets",
            "vq_vocab_size",
        )
    )

    if is_bpe_local:
        assignment_mode = "bpe_local"
    elif is_global_ivf:
        assignment_mode = "global_ivf"
    else:
        raise ValueError(
            "Unknown checkpoint format. Expected either:\n"
            "  BPE-local: partition_type=bpe_local_kmeans, "
            "centers_by_bpe, k_by_bpe\n"
            "or\n"
            "  global-IVF: ivf_centers, global_centers, "
            "global_offsets, vq_vocab_size"
        )

    print("[assignment_mode]", assignment_mode)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------
    tokenizer_name = (
        args.tokenizer
        or ckpt.get("tokenizer_name")
        or cargs.get("tokenizer")
    )
    if tokenizer_name is None:
        raise ValueError(
            "Tokenizer is not specified in arguments or checkpoint"
        )

    print("[tokenizer]", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = int(ckpt["model"]["tok_emb.weight"].shape[0])

    pad_id = ckpt.get("pad_token_id", tokenizer.pad_token_id)
    if pad_id is None:
        raise ValueError("pad_token_id is unavailable")
    pad_id = int(pad_id)

    unk_id = ckpt.get("unk_token_id")
    if unk_id is None:
        unk_id = tokenizer.unk_token_id
    if unk_id is None:
        unk_id = pad_id
    unk_id = int(unk_id)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=int(cargs["d_model"]),
        hop=int(cargs["hop"]),
        n_layers=int(cargs["n_layers"]),
        center_scale=float(cargs.get("center_scale", 1.0)),
    ).to(device)

    ckpt_pos_shape = ckpt["model"]["pos_emb.weight"].shape

    if model.pos_emb.weight.shape != ckpt_pos_shape:
        model.pos_emb = torch.nn.Embedding(
            ckpt_pos_shape[0],
            ckpt_pos_shape[1],
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # --------------------------------------------------------
    # Check clustering metadata
    # --------------------------------------------------------
    if assignment_mode == "bpe_local":
        centers_by_bpe = ckpt["centers_by_bpe"]
        k_by_bpe = ckpt["k_by_bpe"].long().cpu()

        if k_by_bpe.numel() != vocab_size:
            raise ValueError(
                "k_by_bpe length must equal BPE vocabulary size: "
                f"{k_by_bpe.numel()} vs {vocab_size}"
            )

        max_local_clusters = int(
            ckpt.get(
                "max_local_clusters",
                int(k_by_bpe.max().item()) if k_by_bpe.numel() else 0,
            )
        )

        if max_local_clusters <= 0:
            raise ValueError(
                f"Invalid max_local_clusters={max_local_clusters}"
            )

        for bpe_id, centers in centers_by_bpe.items():
            bpe_id = int(bpe_id)

            if not (0 <= bpe_id < vocab_size):
                raise ValueError(
                    f"centers_by_bpe contains invalid BPE ID {bpe_id}"
                )

            if not torch.is_tensor(centers):
                centers = torch.tensor(centers)

            expected_k = int(k_by_bpe[bpe_id].item())
            actual_k = int(centers.size(0))

            if expected_k != actual_k:
                raise ValueError(
                    f"Checkpoint k mismatch for BPE ID {bpe_id}: "
                    f"k_by_bpe={expected_k}, centers={actual_k}"
                )

            if actual_k > max_local_clusters:
                raise ValueError(
                    f"BPE ID {bpe_id} has {actual_k} centers, "
                    f"larger than max_local_clusters={max_local_clusters}"
                )

        print("[partition_type]", ckpt.get("partition_type"))
        print("[id_scheme]", ckpt.get("id_scheme"))
        print("[max_local_clusters]", max_local_clusters)
        print("[BPEs with centers]", len(centers_by_bpe))

    else:
        ivf_centers = ckpt["ivf_centers"]
        global_centers = ckpt["global_centers"]
        global_offsets = ckpt["global_offsets"].long()

        if global_offsets.numel() != ivf_centers.size(0) + 1:
            raise ValueError(
                "global_offsets length must equal ivf_nlist + 1: "
                f"{global_offsets.numel()} vs {ivf_centers.size(0) + 1}"
            )

        if int(global_offsets[-1]) != global_centers.size(0):
            raise ValueError(
                "Last global offset must equal number of global centers: "
                f"{int(global_offsets[-1])} vs {global_centers.size(0)}"
            )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split=args.split)
    else:
        ds = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
        )

    all_ctx = []
    all_tgt = []
    samples = []
    cursor = 0

    print("[data] tokenizing")
    limit = min(args.max_samples, len(ds))

    for sample_idx, ex in enumerate(
        tqdm(ds.select(range(limit)), desc="[tokenize]")
    ):
        token_ids = tokenizer.encode(
            ex[args.text_col],
            add_special_tokens=False,
        )[:args.seq_len]

        if len(token_ids) < 2:
            continue

        token_ids = [
            token_id if token_id < vocab_size else unk_id
            for token_id in token_ids
        ]

        ctx_i, tgt_i = make_windows(
            token_ids,
            int(cargs["hop"]),
            pad_id,
        )

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
        raise ValueError("No windows were created")

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print("[data] windows", f"{len(tgt):,}")

    # --------------------------------------------------------
    # Assignment
    # --------------------------------------------------------
    common = {
        "samples": samples,
        "token_ids_flat": tgt.to(torch.int32),
        "offsets": [
            (
                s["sample_idx"],
                s["start"],
                s["end"],
                s["length"],
            )
            for s in samples
        ],
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "vocab_type": ckpt.get("vocab_type", "byte_bpe"),
        "hop": int(cargs["hop"]),
        "center_scale": float(cargs.get("center_scale", 1.0)),
        "context_type": ckpt.get("context_type"),
        "context_width": ckpt.get("context_width"),
        "ckpt": args.ckpt,
        "tokenizer": tokenizer_name,
    }

    if assignment_mode == "bpe_local":
        local_vq_ids, missing_bpe_ids = assign_ids_bpe_local(
            model=model,
            ctx=ctx,
            tgt=tgt,
            centers_by_bpe=centers_by_bpe,
            k_by_bpe=k_by_bpe,
            batch_size=args.batch_size,
            device=device,
            k_block=args.k_block,
            missing_bpe_policy=args.missing_bpe_policy,
        )

        # Validate IDs for BPEs that actually have learned centers.
        token_ids_long = tgt.long().reshape(-1)
        local_ids_long = local_vq_ids.long().reshape(-1)
        token_k = k_by_bpe[token_ids_long]

        seen_mask = token_k > 0
        if seen_mask.any():
            bad = local_ids_long[seen_mask] >= token_k[seen_mask]
            if bad.any():
                seen_positions = torch.nonzero(seen_mask, as_tuple=False).reshape(-1)
                rel_idx = int(torch.nonzero(bad, as_tuple=False)[0].item())
                idx = int(seen_positions[rel_idx].item())
                bpe_id = int(token_ids_long[idx].item())
                raise RuntimeError(
                    f"Invalid local VQ ID at position {idx}: "
                    f"bpe={bpe_id}, "
                    f"local={int(local_ids_long[idx].item())}, "
                    f"k={int(k_by_bpe[bpe_id].item())}"
                )

        if local_vq_ids.numel() and int(local_vq_ids.max()) >= max_local_clusters:
            raise RuntimeError(
                f"Assigned local VQ ID {int(local_vq_ids.max())} "
                f"exceeds max_local_clusters={max_local_clusters}"
            )

        # For pair representation, VQ vocabulary is only the local-ID axis.
        # This preserves compatibility with loaders that expect vq_ids_flat,
        # vq_vocab_size and vq_pad_id.
        out_data = {
            **common,
            "vq_ids_flat": local_vq_ids.to(torch.int16),
            "local_vq_ids_flat": local_vq_ids.to(torch.int16),
            "k_by_bpe": k_by_bpe.to(torch.int16),
            "max_local_clusters": max_local_clusters,
            "vq_vocab_size": max_local_clusters,
            "vq_pad_id": max_local_clusters,
            "partitioned": True,
            "partition_type": "bpe_local_kmeans",
            "id_scheme": "(bpe_id, local_vq_id)",
            "missing_bpe_policy": args.missing_bpe_policy,
            "missing_bpe_ids": missing_bpe_ids,
        }

        torch.save(out_data, args.out)

        print("[local VQ min/max]", int(local_vq_ids.min()), int(local_vq_ids.max()))
        print("[vq_vocab_size]", max_local_clusters)
        print("[vq_pad_id]", max_local_clusters)
        print("[missing BPE types]", len(missing_bpe_ids))
        if missing_bpe_ids:
            print("[missing BPE IDs first 30]", missing_bpe_ids[:30])

    else:
        vq_ids, ivf_ids = assign_ids_global_ivf(
            model=model,
            ctx=ctx,
            ivf_centers=ivf_centers,
            global_centers=global_centers,
            global_offsets=global_offsets,
            batch_size=args.batch_size,
            device=device,
            k_block=args.k_block,
        )

        vq_vocab_size = int(ckpt["vq_vocab_size"])

        if vq_ids.numel() and int(vq_ids.max()) >= vq_vocab_size:
            raise RuntimeError(
                f"Assigned ID {int(vq_ids.max())} "
                f"exceeds vq_vocab_size={vq_vocab_size}"
            )

        vq_pad_id = vq_vocab_size

        out_data = {
            **common,
            "vq_ids_flat": vq_ids.to(torch.int32),
            "ivf_ids_flat": ivf_ids.to(torch.int32),
            "vq_vocab_size": vq_vocab_size,
            "vq_pad_id": vq_pad_id,
            "partitioned": False,
            "partition_type": "global_ivf_then_kmeans",
            "id_scheme": "global_ivf_then_local_kmeans",
        }

        torch.save(out_data, args.out)

        print("[vq_vocab_size]", vq_vocab_size)
        print("[vq_pad_id]", vq_pad_id)

    print("[save]", args.out)


if __name__ == "__main__":
    main()
