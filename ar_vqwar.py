#!/usr/bin/env python3
"""Two leak-free HOP10 AR modes.

local_bpe_direct:
  (BPE[t-11], local-VQW[t-11]) + BPE[t-10:t-1] -> BPE[t]

global_vqwar:
  global-VQW[t-11] + BPE[t-10:t-1] -> global-VQW[t]
  -> frozen center-to-BPE decoder (evaluation only)
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


MODES = ("local_bpe_direct", "global_vqwar")


def checkpoint_hop(obj):
    return int(obj.get("hop", obj.get("args", {}).get("hop", -1)))


class GapPairDataset(Dataset):
    def __init__(self, samples, token_ids, vq_ids, vq_pad_id, gap=11,
                 local_bpe_tokens=10, max_len=255):
        self.token_ids = token_ids.long().reshape(-1)
        self.vq_ids = vq_ids.long().reshape(-1)
        self.vq_pad_id = int(vq_pad_id)
        self.gap = int(gap)
        self.local_bpe_tokens = int(local_bpe_tokens)
        self.max_len = int(max_len)
        if self.token_ids.numel() != self.vq_ids.numel():
            raise ValueError("token_ids and vq_ids have different lengths")
        if self.local_bpe_tokens != self.gap - 1:
            raise ValueError("local_bpe_tokens must equal gap - 1")

        self.examples = []
        for sample in samples:
            start, end = int(sample["start"]), int(sample["end"])
            aligned = end - start - self.gap
            if aligned < 1:
                continue
            for offset in range(0, aligned, self.max_len):
                length = min(self.max_len, aligned - offset)
                self.examples.append((start + offset, length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        source_start, length = self.examples[index]
        target_start = source_start + self.gap
        distant_bpe = torch.zeros(self.max_len, dtype=torch.long)
        vq_in = torch.full((self.max_len,), self.vq_pad_id, dtype=torch.long)
        vq_y = torch.full((self.max_len,), -100, dtype=torch.long)
        bpe_y = torch.full((self.max_len,), -100, dtype=torch.long)
        bpe_gap = torch.zeros(
            (self.max_len, self.local_bpe_tokens), dtype=torch.long
        )
        valid = torch.zeros(self.max_len, dtype=torch.bool)

        distant_bpe[:length] = self.token_ids[source_start:source_start + length]
        vq_in[:length] = self.vq_ids[source_start:source_start + length]
        vq_y[:length] = self.vq_ids[target_start:target_start + length]
        bpe_y[:length] = self.token_ids[target_start:target_start + length]
        local_segment = self.token_ids[
            source_start + 1:source_start + length + self.local_bpe_tokens
        ]
        bpe_gap[:length] = local_segment.unfold(
            0, self.local_bpe_tokens, 1
        )[:length]
        valid[:length] = True
        return distant_bpe, vq_in, bpe_gap, vq_y, bpe_y, valid


class RecentBPEFusion(nn.Module):
    def __init__(self, token_vocab_size, local_bpe_tokens, d_model):
        super().__init__()
        self.local_bpe_tokens = int(local_bpe_tokens)
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.relative_pos = nn.Embedding(local_bpe_tokens, d_model)
        self.score = nn.Linear(d_model, 1, bias=False)
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, bpe_gap):
        rel = torch.arange(self.local_bpe_tokens, device=bpe_gap.device)
        e = self.bpe_embedding(bpe_gap) + self.relative_pos(rel)[None, None]
        weight = torch.softmax(self.score(torch.tanh(e)), dim=2)
        return self.projection(torch.sum(weight * e, dim=2))


class ARBackbone(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout, max_len):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, key_padding_mask):
        length = h.size(1)
        pos = torch.arange(length, device=h.device)[None]
        h = h + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=h.device), 1
        )
        return self.norm(self.transformer(
            h, mask=causal, src_key_padding_mask=key_padding_mask
        ))


class LocalBPEDirectAR(nn.Module):
    def __init__(self, token_vocab_size, local_vq_vocab_size,
                 local_bpe_tokens=10, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=255):
        super().__init__()
        self.local_vq_vocab_size = int(local_vq_vocab_size)
        self.vq_pad_id = self.local_vq_vocab_size
        self.distant_bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.local_vq_embedding = nn.Embedding(
            self.local_vq_vocab_size + 1, d_model,
            padding_idx=self.vq_pad_id,
        )
        self.backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len
        )
        self.recent = RecentBPEFusion(
            token_vocab_size, local_bpe_tokens, d_model
        )
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.bpe_head = nn.Linear(d_model, token_vocab_size)

    def forward(self, distant_bpe, local_vq, bpe_gap, key_padding_mask=None):
        h = (
            self.distant_bpe_embedding(distant_bpe)
            + self.local_vq_embedding(local_vq)
        )
        h = self.backbone(h, key_padding_mask)
        h = self.fusion_norm(F.gelu(self.fusion(
            torch.cat([h, self.recent(bpe_gap)], dim=-1)
        )))
        return self.bpe_head(h)


class GlobalVQWAR(nn.Module):
    def __init__(self, centers, token_vocab_size, local_bpe_tokens=10,
                 d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=255):
        super().__init__()
        centers = centers.float()
        self.vq_vocab_size = int(centers.size(0))
        self.vq_pad_id = self.vq_vocab_size
        zero = torch.zeros(1, centers.size(1), dtype=centers.dtype)
        self.register_buffer("center_table", torch.cat([centers, zero], 0))
        self.input_projection = nn.Linear(centers.size(1), d_model, bias=False)
        self.backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len
        )
        self.recent = RecentBPEFusion(
            token_vocab_size, local_bpe_tokens, d_model
        )
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.vq_head = nn.Linear(d_model, self.vq_vocab_size)

    def forward(self, vq_in, bpe_gap, key_padding_mask=None):
        centers = F.embedding(
            vq_in, self.center_table, padding_idx=self.vq_pad_id
        )
        h = self.backbone(self.input_projection(centers), key_padding_mask)
        h = self.fusion_norm(F.gelu(self.fusion(
            torch.cat([h, self.recent(bpe_gap)], dim=-1)
        )))
        return self.vq_head(h)


class FrozenCenterToBPE(nn.Module):
    def __init__(self, centers, weight, bias):
        super().__init__()
        if weight.ndim != 2 or weight.size(1) != centers.size(1):
            raise ValueError("decoder/center dimension mismatch")
        self.register_buffer("centers", centers.float())
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())

    def forward(self, vq_ids):
        return F.linear(F.embedding(vq_ids, self.centers), self.weight, self.bias)

    def decode_centers(self, centers):
        return F.linear(centers, self.weight, self.bias)


def load_decoder(codebook):
    state = codebook.get("decoder_state_dict")
    if state is None:
        model_state = codebook.get("model", {})
        if "decoder.weight" in model_state and "decoder.bias" in model_state:
            state = {
                "weight": model_state["decoder.weight"],
                "bias": model_state["decoder.bias"],
            }
    if state is None or "weight" not in state or "bias" not in state:
        raise KeyError("No pretrained linear decoder in global codebook")
    return state["weight"], state["bias"]


def accumulate_topk(logits, targets, k=5):
    top = logits.topk(min(k, logits.size(-1)), dim=-1).indices
    return int(top[:, 0].eq(targets).sum()), int(
        top.eq(targets[:, None]).any(dim=1).sum()
    )


@torch.no_grad()
def evaluate_local(model, loader, device):
    model.eval()
    total = dict(count=0, loss=0.0, top1=0, top5=0)
    for distant_bpe, vq_in, bpe_gap, _vq_y, bpe_y, valid in tqdm(
        loader, desc="[eval local]", leave=False
    ):
        distant_bpe, vq_in = distant_bpe.to(device), vq_in.to(device)
        bpe_gap, bpe_y, valid = bpe_gap.to(device), bpe_y.to(device), valid.to(device)
        logits = model(distant_bpe, vq_in, bpe_gap, ~valid)
        mask = bpe_y.ne(-100)
        pred, target = logits[mask], bpe_y[mask]
        n = int(target.numel())
        total["count"] += n
        total["loss"] += float(F.cross_entropy(pred, target, reduction="sum"))
        a, b = accumulate_topk(pred, target)
        total["top1"] += a; total["top5"] += b
    n = max(total["count"], 1)
    ce = total["loss"] / n
    return dict(count=total["count"], bpe_loss=ce,
                bpe_ppl=math.exp(min(ce, 20.0)),
                bpe_top1=total["top1"] / n, bpe_top5=total["top5"] / n)


@torch.no_grad()
def evaluate_global(model, decoder, loader, device, mixture_topk):
    model.eval()
    keys = ("vq", "bpe", "mix_bpe", "oracle_bpe")
    total = {"count": 0}
    for key in keys:
        total.update({f"{key}_loss": 0.0, f"{key}_top1": 0, f"{key}_top5": 0})
    for _distant_bpe, vq_in, bpe_gap, vq_y, bpe_y, valid in tqdm(
        loader, desc="[eval global]", leave=False
    ):
        vq_in, vq_y = vq_in.to(device), vq_y.to(device)
        bpe_gap, bpe_y, valid = bpe_gap.to(device), bpe_y.to(device), valid.to(device)
        logits = model(vq_in, bpe_gap, ~valid)
        mask = vq_y.ne(-100)
        vl, vt, bt = logits[mask], vq_y[mask], bpe_y[mask]
        n = int(vt.numel()); total["count"] += n

        predictions = {"vq": (vl, vt)}
        predictions["bpe"] = (decoder(vl.argmax(-1)), bt)
        k = min(int(mixture_topk), vl.size(-1))
        values, ids = vl.topk(k, dim=-1)
        mixed = torch.sum(
            torch.softmax(values, -1).unsqueeze(-1)
            * F.embedding(ids, decoder.centers), dim=1
        )
        predictions["mix_bpe"] = (decoder.decode_centers(mixed), bt)
        predictions["oracle_bpe"] = (decoder(vt), bt)
        for key, (pred, target) in predictions.items():
            total[f"{key}_loss"] += float(
                F.cross_entropy(pred, target, reduction="sum")
            )
            a, b = accumulate_topk(pred, target)
            total[f"{key}_top1"] += a; total[f"{key}_top5"] += b
    n = max(total["count"], 1)
    result = {"count": total["count"]}
    for key in keys:
        ce = total[f"{key}_loss"] / n
        result[f"{key}_loss"] = ce
        result[f"{key}_ppl"] = math.exp(min(ce, 20.0))
        result[f"{key}_top1"] = total[f"{key}_top1"] / n
        result[f"{key}_top5"] = total[f"{key}_top5"] / n
    return result


def format_metrics(prefix, metrics):
    names = [key for key in metrics if key.endswith(("_ppl", "_top1"))]
    return " ".join(f"{prefix}_{key}={metrics[key]:.4f}" for key in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--data", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=int, default=11)
    ap.add_argument("--local_bpe_tokens", type=int, default=10)
    ap.add_argument("--mixture_topk", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=255)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.local_bpe_tokens != args.gap - 1:
        ap.error("--local_bpe_tokens must equal --gap - 1")

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    codebook = torch.load(args.codebook, map_location="cpu", weights_only=False)
    if checkpoint_hop(data) != 10 or checkpoint_hop(codebook) != 10:
        raise ValueError(
            f"HOP10 required: data={checkpoint_hop(data)}, "
            f"codebook={checkpoint_hop(codebook)}"
        )
    for key in ("samples", "token_ids_flat", "vq_ids_flat"):
        if key not in data:
            raise KeyError(f"data missing {key}")
    token_ids = data["token_ids_flat"].long().reshape(-1)
    vq_ids = data["vq_ids_flat"].long().reshape(-1)
    if token_ids.numel() != vq_ids.numel():
        raise ValueError("token/VQ length mismatch")
    token_vocab_size = int(codebook["model"]["tok_emb.weight"].shape[0])

    decoder = None
    if args.mode == "local_bpe_direct":
        if codebook.get("partition_type") != "bpe_local_kmeans":
            raise ValueError("local_bpe_direct requires bpe_local_kmeans codebook")
        vq_vocab_size = int(data.get(
            "vq_vocab_size", codebook.get("max_local_clusters", -1)
        ))
        if vq_vocab_size < 1:
            raise ValueError("invalid local vq_vocab_size")
        if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= vq_vocab_size:
            raise ValueError("local VQ ID out of range")
        model = LocalBPEDirectAR(
            token_vocab_size, vq_vocab_size, args.local_bpe_tokens,
            args.d_model, args.n_layers, args.n_heads, args.dropout, args.max_len
        ).to(device)
        selection_metric = "bpe_ppl"
        architecture = "bpe_plus_local_vqw_direct_bpe_ar"
    else:
        centers = codebook["global_centers"].float()
        vq_vocab_size = int(centers.size(0))
        if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= vq_vocab_size:
            raise ValueError("global VQ ID out of range")
        dec_weight, dec_bias = load_decoder(codebook)
        token_vocab_size = int(dec_weight.size(0))
        decoder = FrozenCenterToBPE(
            centers, dec_weight, dec_bias
        ).to(device).eval()
        model = GlobalVQWAR(
            centers, token_vocab_size, args.local_bpe_tokens,
            args.d_model, args.n_layers, args.n_heads, args.dropout, args.max_len
        ).to(device)
        selection_metric = "vq_ppl"
        architecture = "global_vqwid_ar_then_frozen_bpe_decoder"

    samples = list(data["samples"])
    random.shuffle(samples)
    n = len(samples); n_train = int(0.8 * n); n_valid = int(0.1 * n)
    splits = (samples[:n_train], samples[n_train:n_train+n_valid], samples[n_train+n_valid:])
    datasets = [GapPairDataset(
        split, token_ids, vq_ids, vq_vocab_size, args.gap,
        args.local_bpe_tokens, args.max_len
    ) for split in splits]
    loaders = [DataLoader(
        ds, batch_size=args.batch_size, shuffle=(i == 0), num_workers=4,
        pin_memory=True, persistent_workers=True
    ) for i, ds in enumerate(datasets)]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history, best_valid = [], float("inf")
    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
    print(f"[mode] {args.mode}")
    print(f"[architecture] {architecture}")
    print(f"[alignment] distant=t-{args.gap}; recent_bpe={args.local_bpe_tokens}")
    print(f"[vocab] bpe={token_vocab_size} vqw={vq_vocab_size}")

    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; total_count = 0
        pbar = tqdm(loaders[0], desc=f"[train] epoch {epoch}/{args.epochs}")
        for distant_bpe, vq_in, bpe_gap, vq_y, bpe_y, valid in pbar:
            distant_bpe, vq_in = distant_bpe.to(device), vq_in.to(device)
            bpe_gap, valid = bpe_gap.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "local_bpe_direct":
                target = bpe_y.to(device)
                logits = model(distant_bpe, vq_in, bpe_gap, ~valid)
            else:
                target = vq_y.to(device)
                logits = model(vq_in, bpe_gap, ~valid)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100
            )
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(target.ne(-100).sum())
            total_loss += float(loss) * count; total_count += count
            pbar.set_postfix(loss=f"{total_loss / max(total_count, 1):.4f}")

        if args.mode == "local_bpe_direct":
            valid_metrics = evaluate_local(model, loaders[1], device)
            test_metrics = evaluate_local(model, loaders[2], device)
        else:
            valid_metrics = evaluate_global(
                model, decoder, loaders[1], device, args.mixture_topk
            )
            test_metrics = evaluate_global(
                model, decoder, loaders[2], device, args.mixture_topk
            )
        print(f"[epoch {epoch}] {format_metrics('valid', valid_metrics)} "
              f"{format_metrics('test', test_metrics)}")
        history.append({"epoch": epoch, "train_loss": total_loss/max(total_count, 1),
                        "valid": valid_metrics, "test": test_metrics})
        checkpoint = {
            "model": model.state_dict(), "args": vars(args), "history": history,
            "mode": args.mode, "architecture": architecture,
            "vq_vocab_size": vq_vocab_size, "token_vocab_size": token_vocab_size,
            "decoder_frozen": args.mode == "global_vqwar",
            "decoder_source": args.codebook if decoder is not None else None,
            "last_valid": valid_metrics, "last_test": test_metrics,
        }
        torch.save(checkpoint, args.out)
        if valid_metrics[selection_metric] < best_valid:
            best_valid = valid_metrics[selection_metric]
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")
    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()
