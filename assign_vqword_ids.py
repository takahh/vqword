#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from train_vqword import VQWordGNN, make_windows


@torch.no_grad()
def assign_ids(model, centroids, ctx, batch_size, device):
    model.eval()
    centroids = F.normalize(centroids.to(device), dim=-1)

    all_ids = []
    for s in tqdm(range(0, len(ctx), batch_size), desc="[assign]"):
        xb = ctx[s:s+batch_size].to(device)
        z = model.encode_context(xb)
        z = F.normalize(z, dim=-1)

        sim = z @ centroids.T
        pred = sim.argmax(dim=1)

        all_ids.append(pred.cpu())

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

    # 追加
    ap.add_argument("--tokenizer", default=None)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]

    pad_id = ckpt.get("pad_token_id", 0)
    unk_id = ckpt.get("unk_token_id", 1)

    cargs = ckpt["args"]
    vocab_size = len(word2id)

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
    ).to(device)

    model.load_state_dict(ckpt["model"])
    centroids = ckpt["centroids"]

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

    vq_ids = assign_ids(model, centroids, ctx, args.batch_size, device)

    samples = []
    for sample_idx, start, end, n_tok in offsets:
        samples.append({
            "sample_idx": sample_idx,
            "token_ids": tgt[start:end].cpu(),
            "vqword_ids": vq_ids[start:end].cpu(),
            "length": n_tok,
        })

    torch.save({
        "samples": samples,
        "vq_ids_flat": vq_ids,
        "token_ids_flat": tgt,
        "offsets": offsets,
        "word2id": word2id,
        "id2word": id2word,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "vocab_type": vocab_type,
        "hop": cargs["hop"],
        "ckpt": args.ckpt,
        "tokenizer": tokenizer_name,
    }, args.out)

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()