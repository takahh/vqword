#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from train_vqword import VQWordGNN, make_windows


@torch.no_grad()
def assign_ids_per_token(model, dictionary, ctx, tgt, batch_size, device, fallback_tok_to_vq):
    model.eval()

    centers_by_token = dictionary["centers_by_token"]
    pair_to_compact = dictionary.get("pair_to_compact", None)
    centers_by_token = {
        int(k): F.normalize(v.to(device).float(), dim=-1)
        for k, v in centers_by_token.items()
    }
    centers_keys = set(int(k) for k in centers_by_token.keys())
    missing = sorted(set(int(x) for x in tgt.unique().tolist()) - centers_keys)
    print("[missing tokens]", len(missing), missing[:20])

    if 7454 in missing:
        print("[7454 is missing from centers_by_token]")

    if pair_to_compact is None:
        raise ValueError("per-token ckpt requires pair_to_compact")

    all_ids = []

    for s in tqdm(range(0, len(ctx), batch_size), desc="[assign-pertok]"):
        xb = ctx[s:s+batch_size].to(device)
        tb = tgt[s:s+batch_size].to(device)

        z = model.encode_context(xb)
        z = F.normalize(z, dim=-1)

        out = torch.empty(len(tb), dtype=torch.long, device=device)

        for tok in tb.unique().tolist():
            tok = int(tok)
            mask = tb == tok

            if tok not in centers_by_token:
                out[mask] = fallback_tok_to_vq[tok]
                continue

            c = centers_by_token[tok].float()
            zz = z[mask].float()
            sim = zz @ c.T
            local_id = sim.argmax(dim=1)

            ids = [
                int(pair_to_compact[(tok, int(lid))])
                for lid in local_id.detach().cpu().tolist()
            ]

            out[mask] = torch.tensor(ids, dtype=torch.long, device=device)

        all_ids.append(out.cpu())

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
    ap.add_argument("--dictionary", default=None)

    # 追加
    ap.add_argument("--tokenizer", default=None)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    dictionary = None
    if args.dictionary is not None:
        dictionary = torch.load(args.dictionary, map_location="cpu")
        print("[dictionary] loaded:", args.dictionary)
    word2id = ckpt.get("word2id", None)
    id2word = ckpt.get("id2word", None)

    tokenizer_name = args.tokenizer or ckpt["args"].get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    vocab_size = ckpt.get("vocab_size", None)
    if vocab_size is None:
        vocab_size = ckpt.get("token_vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)

    pad_id = ckpt.get("pad_token_id", 0)
    unk_id = ckpt.get("unk_token_id", None)
    if unk_id is None:
        unk_id = pad_id
    cargs = ckpt["args"]

    tokenizer_name = args.tokenizer or cargs.get("tokenizer", None)

    if tokenizer_name is not None:
        print(f"[tokenizer] using HF tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_type = f"bpe:{tokenizer_name}"
    else:
        raise ValueError(
            "This assign script is now for tokenizer-based/BPE VQWord. "
            "Please pass --tokenizer gpt2."
        )

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=cargs["d_model"],
        hop=cargs["hop"],
        n_layers=cargs["n_layers"],
        center_scale=cargs.get("center_scale", 1.0),
    ).to(device)

    print("[center_scale]", cargs.get("center_scale", 1.0))

    model.load_state_dict(ckpt["model"])
    centroids = ckpt.get("centroids", None)

    if centroids is not None:
        if centroids.dim() == 3:
            centroids = centroids.reshape(-1, centroids.size(-1))
        print("[debug] centroids shape:", centroids.shape)

    ds = load_dataset(args.dataset, split=args.split)
    all_ctx = []
    all_tgt = []
    offsets = []

    print("[data] tokenizing")

    for i, ex in enumerate(tqdm(ds.select(range(min(args.max_samples, len(ds)))))):
        text = ex[args.text_col]

        # BPE token ids
        ids = tokenizer.encode(text, add_special_tokens=False)

        if len(ids) < 2 * cargs["hop"] + 2:
            continue

        # 念のためckpt vocab外はunkへ
        ids = [x if x < vocab_size else unk_id for x in ids]

        ids = torch.tensor(ids, dtype=torch.long)

        ctx_i, tgt_i = make_windows(ids, cargs["hop"], pad_id)

        if len(tgt_i) == 0:
            continue

        start = sum(len(x) for x in all_tgt)
        end = start + len(tgt_i)

        offsets.append((i, start, end, len(ids)))
        all_ctx.append(ctx_i)
        all_tgt.append(tgt_i)

    if len(all_ctx) == 0:
        raise ValueError("No windows created. Try checking text_col/tokenizer/make_windows.")

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print(f"[data] windows={len(tgt):,}")
    assign_source = ckpt
    print(ckpt.keys())

    if "centers_by_token" not in assign_source:
        raise ValueError("ckpt does not contain centers_by_token")

    if "pair_to_compact" not in assign_source:
        raise ValueError("ckpt does not contain pair_to_compact")

    base = max(int(v) for v in assign_source["pair_to_compact"].values()) + 1

    centers_keys = set(int(k) for k in assign_source["centers_by_token"].keys())

    fallback_tokens = sorted(
        tok for tok in range(vocab_size)
        if tok not in centers_keys
    )

    fallback_tok_to_vq = {
        tok: base + i
        for i, tok in enumerate(fallback_tokens)
    }

    base_vq_vocab_size = base + len(fallback_tok_to_vq)
    vq_pad_id = base_vq_vocab_size

    print("[semantic_vq_size]", base)
    print("[fallback_tokens]", len(fallback_tok_to_vq))
    print("[base_vq_vocab_size]", base_vq_vocab_size)
    print("[vq_pad_id]", vq_pad_id)

    vq_ids = assign_ids_per_token(
        model,
        assign_source,
        ctx,
        tgt,
        args.batch_size,
        device,
        fallback_tok_to_vq,
    )
    samples = []

    for sample_idx, start, end, n_tok in offsets:
        samples.append({
            "sample_idx": int(sample_idx),
            "start": int(start),
            "end": int(end),
            "length": int(n_tok),
        })
    if args.dictionary is not None:
        raw_dict = torch.load(args.dictionary, map_location="cpu")

        for tok, vq_id in fallback_tok_to_vq.items():
            raw_dict[int(vq_id)] = [
                (
                    int(tok),
                    tokenizer.decode([int(tok)]),
                    1,
                )
            ]

        raw_dict["vq_vocab_size"] = base_vq_vocab_size
        raw_dict["vq_pad_id"] = vq_pad_id
        raw_dict["fallback_tok_to_vq"] = fallback_tok_to_vq

        dict_out = args.out.replace("_ids.pt", "_dictionary.pt")
        torch.save(raw_dict, dict_out)
        print("[save dictionary with fallback]", dict_out)

    # 保存時だけ小さいdtypeにする
    vq_ids_save = vq_ids.to(torch.int32)
    token_ids_save = tgt.to(torch.int32)

    torch.save(
        {
            "samples": samples,

            # ID本体はflat配列として一度だけ保存
            "vq_ids_flat": vq_ids_save,
            "token_ids_flat": token_ids_save,

            "offsets": offsets,
            "word2id": word2id,
            "id2word": id2word,
            "pad_token_id": pad_id,
            "unk_token_id": unk_id,
            "vocab_type": vocab_type,
            "hop": cargs["hop"],
            "ckpt": args.ckpt,
            "tokenizer": tokenizer_name,
            "fallback_tok_to_vq": fallback_tok_to_vq,
            "vq_vocab_size": base_vq_vocab_size,
            "vq_pad_id": vq_pad_id,
        },
        args.out,
    )

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()