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

    for start in tqdm(range(0, len(ctx), batch_size), desc="[assign global IVF]"):
        end = min(start + batch_size, len(ctx))
        xb = ctx[start:end].to(device)

        z = F.normalize(model.encode_context(xb).float(), dim=-1)
        ivf_ids = assign_blockwise(z, coarse, k_block=k_block)

        batch_global_ids = torch.empty(z.size(0), dtype=torch.long, device=device)

        for list_id in torch.unique(ivf_ids).tolist():
            mask = ivf_ids == list_id
            begin = int(offsets[list_id].item())
            finish = int(offsets[list_id + 1].item())

            if finish <= begin:
                raise RuntimeError(
                    f"IVF list {list_id} has no fine centers: offsets=({begin}, {finish})"
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
    ap.add_argument("--out", default="tiny_global_ivf_ids.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    ckpt = torch.load(
        args.ckpt,
        map_location="cpu",
        weights_only=False,
    )
    cargs = ckpt["args"]

    required_keys = {
        "model",
        "ivf_centers",
        "global_centers",
        "global_offsets",
        "vq_vocab_size",
    }
    missing_keys = sorted(required_keys - set(ckpt.keys()))
    if missing_keys:
        raise ValueError(
            f"Checkpoint is not a global IVF checkpoint. Missing keys: {missing_keys}"
        )

    tokenizer_name = args.tokenizer or ckpt.get("tokenizer_name") or cargs.get("tokenizer")
    if tokenizer_name is None:
        raise ValueError("Tokenizer is not specified in arguments or checkpoint")

    print("[tokenizer]", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = int(ckpt["model"]["tok_emb.weight"].shape[0])
    pad_id = int(ckpt.get("pad_token_id", tokenizer.pad_token_id))
    unk_id = ckpt.get("unk_token_id")
    if unk_id is None:
        unk_id = tokenizer.unk_token_id
    if unk_id is None:
        unk_id = pad_id
    unk_id = int(unk_id)

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=int(cargs["d_model"]),
        hop=int(cargs["hop"]),
        n_layers=int(cargs["n_layers"]),
        center_scale=float(cargs.get("center_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

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

    if args.dataset_config is None:
        ds = load_dataset(args.dataset, split=args.split)
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    all_ctx = []
    all_tgt = []
    samples = []
    cursor = 0

    print("[data] tokenizing")
    limit = min(args.max_samples, len(ds))

    for sample_idx, ex in enumerate(tqdm(ds.select(range(limit)))):
        token_ids = tokenizer.encode(
            ex[args.text_col],
            add_special_tokens=False,
        )[:args.seq_len]

        if len(token_ids) < 2:
            continue

        token_ids = [token_id if token_id < vocab_size else unk_id for token_id in token_ids]
        ctx_i, tgt_i = make_windows(token_ids, int(cargs["hop"]), pad_id)

        start = cursor
        end = start + len(tgt_i)
        cursor = end

        samples.append({
            "sample_idx": int(sample_idx),
            "start": int(start),
            "end": int(end),
            "length": int(len(tgt_i)),
        })
        all_ctx.append(ctx_i)
        all_tgt.append(tgt_i)

    if not all_ctx:
        raise ValueError("No windows were created")

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)
    print("[data] windows", f"{len(tgt):,}")

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
            f"Assigned ID {int(vq_ids.max())} exceeds vq_vocab_size={vq_vocab_size}"
        )

    vq_pad_id = vq_vocab_size

    torch.save(
        {
            "samples": samples,
            "vq_ids_flat": vq_ids.to(torch.int32),
            "token_ids_flat": tgt.to(torch.int32),
            "ivf_ids_flat": ivf_ids.to(torch.int32),
            "offsets": [
                (s["sample_idx"], s["start"], s["end"], s["length"])
                for s in samples
            ],
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": ckpt.get("vocab_type", "byte_bpe"),
            "hop": int(cargs["hop"]),
            "ckpt": args.ckpt,
            "tokenizer": tokenizer_name,
            "vq_vocab_size": vq_vocab_size,
            "vq_pad_id": vq_pad_id,
            "partitioned": False,
            "partition_type": "global_ivf_then_kmeans",
            "id_scheme": "global_ivf_then_local_kmeans",
        },
        args.out,
    )

    print("[vq_vocab_size]", vq_vocab_size)
    print("[vq_pad_id]", vq_pad_id)
    print("[save]", args.out)


if __name__ == "__main__":
    main()
