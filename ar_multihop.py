#!/usr/bin/env python3
"""
BPE + SC0VQW autoregressive language model.

Pipeline:
    input at position t:
        BPE[t] + alpha * projected(SC0VQW[t])
    causal Transformer
    target:
        BPE[t+1]

The SC0 VQ codebook centers are frozen. The BPE embedding, VQ projection,
Transformer, and BPE output head are trained end-to-end.
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class BPEPlusSC0Dataset(Dataset):
    def __init__(self, samples, token_ids_flat, vq_ids_flat, max_len=512):
        self.samples = []
        self.token_ids_flat = token_ids_flat.long().reshape(-1)
        self.vq_ids_flat = vq_ids_flat.long().reshape(-1)
        self.max_len = int(max_len)

        if self.token_ids_flat.numel() != self.vq_ids_flat.numel():
            raise ValueError(
                f"BPE/VQ length mismatch: {self.token_ids_flat.numel()} vs "
                f"{self.vq_ids_flat.numel()}"
            )

        for sample in samples:
            start = int(sample["start"])
            end = int(sample["end"])
            length = end - start
            if 2 <= length <= self.max_len + 1:
                self.samples.append((start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start, end = self.samples[index]
        return {
            "tok_in": self.token_ids_flat[start:end - 1],
            "vq_in": self.vq_ids_flat[start:end - 1],
            "tok_y": self.token_ids_flat[start + 1:end],
        }


def collate_batch(batch, tok_pad_id, vq_pad_id):
    batch_size = len(batch)
    max_len = max(item["tok_in"].numel() for item in batch)

    tok_in = torch.full((batch_size, max_len), tok_pad_id, dtype=torch.long)
    vq_in = torch.full((batch_size, max_len), vq_pad_id, dtype=torch.long)
    tok_y = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["tok_in"].numel()
        tok_in[i, :n] = item["tok_in"]
        vq_in[i, :n] = item["vq_in"]
        tok_y[i, :n] = item["tok_y"]
        attention_mask[i, :n] = True

    return tok_in, vq_in, tok_y, attention_mask


class FrozenCenterEmbedding(nn.Module):
    def __init__(self, centers):
        super().__init__()
        centers = F.normalize(centers.float(), dim=-1)
        self.padding_idx = int(centers.size(0))
        zero = torch.zeros(1, centers.size(1), dtype=centers.dtype)
        self.register_buffer(
            "weight", torch.cat([centers, zero], dim=0), persistent=False
        )

    def forward(self, ids):
        return F.embedding(ids, self.weight, padding_idx=self.padding_idx)


class BPEPlusSC0LM(nn.Module):
    def __init__(
        self,
        centers,
        token_vocab_size,
        input_mode="vqw",
        vq_input_weight=0.01,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=512,
        tie_weights=False,
    ):
        super().__init__()
        self.input_mode = str(input_mode)

        if self.input_mode not in {
            "vqw",
            "bpe2",
            "vq_shuffle",
            "zero",
        }:
            raise ValueError(
                f"unsupported input_mode: {self.input_mode}"
            )
        self.token_vocab_size = int(token_vocab_size)
        self.vq_vocab_size = int(centers.size(0))
        self.tok_pad_id = self.token_vocab_size
        self.vq_pad_id = self.vq_vocab_size
        self.vq_input_weight = float(vq_input_weight)

        self.tok_emb = nn.Embedding(
            self.token_vocab_size + 1,
            d_model,
            padding_idx=self.tok_pad_id,
        )
        self.tok_emb2 = None

        if self.input_mode == "bpe2":
            self.tok_emb2 = nn.Embedding(
                self.token_vocab_size + 1,
                d_model,
                padding_idx=self.tok_pad_id,
            )
        self.vq_emb = FrozenCenterEmbedding(centers)

        # VQコードブック中心をTransformerの隠れ次元へ写像
        self.vq_proj = nn.Linear(
            centers.size(1),
            d_model,
            bias=False,
        )

        # BPE埋め込みとVQ埋め込みを連結すると2 * d_modelになる。
        # それをTransformer入力のd_modelへ戻す。
        self.input_proj = nn.Linear(
            2 * d_model,
            d_model,
            bias=True,
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.bpe_head = nn.Linear(d_model, self.token_vocab_size, bias=False)
        self.bpe_bias = nn.Parameter(torch.zeros(self.token_vocab_size))

        if tie_weights:
            self.bpe_head.weight = self.tok_emb.weight[:self.token_vocab_size]

    def forward(self, tok_in, vq_in, key_padding_mask=None):
        batch_size, seq_len = tok_in.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len "
                f"{self.pos_emb.num_embeddings}"
            )

        bpe_h = self.tok_emb(tok_in)

        if self.input_mode == "bpe2":
            if self.tok_emb2 is None:
                raise RuntimeError("tok_emb2 is not initialized")

            second_h = self.tok_emb2(tok_in)

        elif self.input_mode in {"vqw", "vq_shuffle"}:
            second_h = self.vq_proj(
                self.vq_emb(vq_in)
            )

        elif self.input_mode == "zero":
            second_h = torch.zeros_like(bpe_h)

        else:
            raise RuntimeError(
                f"unsupported input_mode in forward: {self.input_mode}"
            )

        combined_h = torch.cat(
            [bpe_h, second_h],
            dim=-1,
        )

        h = self.input_proj(combined_h)
        h = self.input_norm(h)

        h = self.input_proj(combined_h)
        h = self.input_norm(h)

        pos = torch.arange(seq_len, device=tok_in.device)[None, :]
        h = h + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=h.device),
            diagonal=1,
        )
        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.norm(h)
        return self.bpe_head(h) + self.bpe_bias


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_top1 = 0
    total_top5 = 0

    for tok_in, vq_in, tok_y, attention_mask in tqdm(
        loader, desc="[eval]", leave=False
    ):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(tok_in, vq_in, key_padding_mask=~attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tok_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid = tok_y.ne(-100)
        n = int(valid.sum().item())
        pred = logits[valid]
        target = tok_y[valid]
        topk = pred.topk(min(5, pred.size(-1)), dim=-1).indices

        total_loss += float(loss.item())
        total_count += n
        total_top1 += int(topk[:, 0].eq(target).sum().item())
        total_top5 += int(topk.eq(target[:, None]).any(dim=1).sum().item())

    ce = total_loss / max(total_count, 1)
    return {
        "bpe_loss": ce,
        "bpe_ppl": math.exp(min(ce, 20.0)),
        "bpe_top1": total_top1 / max(total_count, 1),
        "bpe_top5": total_top5 / max(total_count, 1),
        "count": total_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="TinyStories SC0 VQ ID file")
    ap.add_argument("--codebook", required=True, help="matching SC0 VQ codebook")
    ap.add_argument("--out", default="ar_bpe_plus_sc0vqw.pt")
    ap.add_argument("--vq_input_weight", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=255)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tie_weights", action="store_true")
    ap.add_argument(
        "--input_mode",
        type=str,
        default="vqw",
        choices=[
            "vqw",
            "bpe2",
            "vq_shuffle",
            "zero",
        ],
        help=(
            "vqw: normal BPE + VQW concat; "
            "bpe2: two independently learned BPE embeddings; "
            "vq_shuffle: globally shuffled VQ IDs; "
            "zero: zero-valued second input channel"
        ),
    )

    ap.add_argument(
        "--control_seed",
        type=int,
        default=12345,
        help="seed used for control-input construction such as VQ shuffling",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    codebook = torch.load(args.codebook, map_location="cpu", weights_only=False)

    required = {"samples", "token_ids_flat", "vq_ids_flat"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise KeyError(f"data missing keys: {missing}")
    if "global_centers" not in codebook:
        raise KeyError("codebook missing global_centers")

    samples = list(data["samples"])
    token_ids_flat = data["token_ids_flat"].long().reshape(-1)
    vq_ids_flat = data["vq_ids_flat"].long().reshape(-1)
    centers = codebook["global_centers"].float()
    if args.input_mode == "vq_shuffle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.control_seed)

        permutation = torch.randperm(
            vq_ids_flat.numel(),
            generator=generator,
        )

        vq_ids_flat = vq_ids_flat[permutation]

        print(
            "[control] VQ IDs globally shuffled "
            f"with seed={args.control_seed}"
        )

    token_vocab_size = int(token_ids_flat.max().item()) + 1
    if "token_vocab_size" in data:
        token_vocab_size = int(data["token_vocab_size"])
    elif int(token_ids_flat.max().item()) < 50257:
        token_vocab_size = 50257

    vq_vocab_size = int(centers.size(0))
    if vq_ids_flat.min().item() < 0 or vq_ids_flat.max().item() >= vq_vocab_size:
        raise ValueError(
            f"VQ IDs out of range: {int(vq_ids_flat.min())}.."
            f"{int(vq_ids_flat.max())}, vocab={vq_vocab_size}"
        )

    data_hop = int(data.get("hop", -1))
    codebook_hop = int(codebook.get("args", {}).get("hop", -1))
    data_center_scale = data.get("center_scale", None)
    codebook_center_scale = codebook.get("args", {}).get("center_scale", None)

    print("[task] autoregressive BPE prediction")
    print(f"[input mode] {args.input_mode}")
    if args.input_mode == "vqw":
        print("[input] CAT(BPE embedding, projected VQW center)")

    elif args.input_mode == "bpe2":
        print("[input] CAT(BPE embedding 1, BPE embedding 2)")

    elif args.input_mode == "vq_shuffle":
        print("[input] CAT(BPE embedding, projected shuffled VQW center)")
        print(f"[control seed] {args.control_seed}")
    elif args.input_mode == "zero":
        print("[input] CAT(BPE embedding, zero vector)")
        
    print(f"[data hop] {data_hop}")
    print(f"[codebook hop] {codebook_hop}")
    print(f"[data center scale] {data_center_scale}")
    print(f"[codebook center scale] {codebook_center_scale}")
    print(f"[token vocab size] {token_vocab_size}")
    print(f"[VQ vocab size] {vq_vocab_size}")
    print("[fusion] concat + learned linear projection")
    print("[input fusion] concat + linear projection")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    train_ds = BPEPlusSC0Dataset(
        train_samples, token_ids_flat, vq_ids_flat, args.max_len
    )
    valid_ds = BPEPlusSC0Dataset(
        valid_samples, token_ids_flat, vq_ids_flat, args.max_len
    )
    test_ds = BPEPlusSC0Dataset(
        test_samples, token_ids_flat, vq_ids_flat, args.max_len
    )

    tok_pad_id = token_vocab_size
    vq_pad_id = vq_vocab_size

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tok_pad_id, vq_pad_id),
        )

    train_loader = make_loader(train_ds, True)
    valid_loader = make_loader(valid_ds, False)
    test_loader = make_loader(test_ds, False)

    model = BPEPlusSC0LM(
        centers=centers,
        token_vocab_size=token_vocab_size,
        input_mode=args.input_mode,
        vq_input_weight=args.vq_input_weight,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        tie_weights=args.tie_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    best_valid = float("inf")
    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}/{args.epochs}")
        for tok_in, vq_in, tok_y, attention_mask in pbar:
            tok_in = tok_in.to(device)
            vq_in = vq_in.to(device)
            tok_y = tok_y.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tok_in, vq_in, key_padding_mask=~attention_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tok_y.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_valid_tokens = int(tok_y.ne(-100).sum().item())
            running_loss += float(loss.item()) * n_valid_tokens
            running_count += n_valid_tokens
            pbar.set_postfix(
                bpe=f"{running_loss / max(running_count, 1):.4f}"
            )

        valid_metrics = evaluate(model, valid_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"[epoch {epoch}] "
            f"valid_bpe_ppl={valid_metrics['bpe_ppl']:.4f} "
            f"valid_bpe_top1={valid_metrics['bpe_top1']:.4f} "
            f"valid_bpe_top5={valid_metrics['bpe_top5']:.4f} "
            f"test_bpe_ppl={test_metrics['bpe_ppl']:.4f} "
            f"test_bpe_top1={test_metrics['bpe_top1']:.4f} "
            f"test_bpe_top5={test_metrics['bpe_top5']:.4f}"
        )

        record = {
            "epoch": epoch,
            "train_bpe_loss": running_loss / max(running_count, 1),
            "valid": valid_metrics,
            "test": test_metrics,
        }
        history.append(record)

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "input_mode": args.input_mode,
            "token_vocab_size": token_vocab_size,
            "vq_vocab_size": vq_vocab_size,
            "tok_pad_id": tok_pad_id,
            "vq_pad_id": vq_pad_id,
            "data_source": args.data,
            "codebook_source": args.codebook,
            "vq_centers_frozen": True,
            "last_valid": valid_metrics,
            "last_test": test_metrics,
        }
        torch.save(checkpoint, args.out)

        if valid_metrics["bpe_ppl"] < best_valid:
            best_valid = valid_metrics["bpe_ppl"]
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")

    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()
