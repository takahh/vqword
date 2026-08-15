#!/usr/bin/env python3
"""Pure fixed-HOP VQW autoregression followed by a frozen BPE decoder.

For target position t, the AR input ends at t-gap.  With bilateral HOP10,
gap=11 is the first leak-free alignment: the newest input VQW is centered at
t-11 and its right context ends at t-1.  The model is trained only with VQW
cross entropy.  BPE metrics are computed by decoding the predicted VQW with
the pretrained linear center-to-BPE decoder stored in the codebook checkpoint.
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


class GapVQARDataset(Dataset):
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
        if self.gap < 1:
            raise ValueError("gap must be positive")
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

        vq_in = torch.full((self.max_len,), self.vq_pad_id, dtype=torch.long)
        vq_y = torch.full((self.max_len,), -100, dtype=torch.long)
        bpe_y = torch.full((self.max_len,), -100, dtype=torch.long)
        bpe_gap = torch.zeros(
            (self.max_len, self.local_bpe_tokens), dtype=torch.long
        )
        valid = torch.zeros(self.max_len, dtype=torch.bool)

        vq_in[:length] = self.vq_ids[source_start:source_start + length]
        vq_y[:length] = self.vq_ids[target_start:target_start + length]
        bpe_y[:length] = self.token_ids[target_start:target_start + length]
        local_segment = self.token_ids[
            source_start + 1:
            source_start + length + self.local_bpe_tokens
        ]
        bpe_gap[:length] = local_segment.unfold(
            0, self.local_bpe_tokens, 1
        )[:length]
        valid[:length] = True
        return vq_in, bpe_gap, vq_y, bpe_y, valid


class PureVQWAR(nn.Module):
    def __init__(self, centers, token_vocab_size, local_bpe_tokens=10,
                 d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=255):
        super().__init__()
        centers = centers.float()
        self.vq_vocab_size = int(centers.size(0))
        self.vq_pad_id = self.vq_vocab_size
        self.local_bpe_tokens = int(local_bpe_tokens)
        zero = torch.zeros(1, centers.size(1), dtype=centers.dtype)
        self.register_buffer("center_table", torch.cat([centers, zero], 0))

        self.input_projection = nn.Linear(centers.size(1), d_model, bias=False)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.bpe_relative_pos = nn.Embedding(self.local_bpe_tokens, d_model)
        self.bpe_attention_score = nn.Linear(d_model, 1, bias=False)
        self.bpe_projection = nn.Linear(d_model, d_model, bias=False)
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.vq_head = nn.Linear(d_model, self.vq_vocab_size)

    def forward(self, vq_in, bpe_gap, key_padding_mask=None):
        _, length = vq_in.shape
        pos = torch.arange(length, device=vq_in.device)[None, :]
        centers = F.embedding(vq_in, self.center_table, padding_idx=self.vq_pad_id)
        h = self.input_projection(centers) + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=vq_in.device),
            diagonal=1,
        )
        h = self.transformer(h, mask=causal, src_key_padding_mask=key_padding_mask)
        vq_h = self.norm(h)

        rel = torch.arange(self.local_bpe_tokens, device=bpe_gap.device)
        bpe_e = (
            self.bpe_embedding(bpe_gap)
            + self.bpe_relative_pos(rel)[None, None, :, :]
        )
        bpe_weight = torch.softmax(
            self.bpe_attention_score(torch.tanh(bpe_e)), dim=2
        )
        bpe_h = self.bpe_projection(
            torch.sum(bpe_weight * bpe_e, dim=2)
        )
        fused = self.fusion(torch.cat([vq_h, bpe_h], dim=-1))
        fused = self.fusion_norm(F.gelu(fused))
        return self.vq_head(fused)


class FrozenCenterToBPE(nn.Module):
    def __init__(self, centers, weight, bias):
        super().__init__()
        if weight.ndim != 2 or weight.size(1) != centers.size(1):
            raise ValueError(
                f"decoder/center dimension mismatch: weight={tuple(weight.shape)}, "
                f"centers={tuple(centers.shape)}"
            )
        self.register_buffer("centers", centers.float())
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())

    def forward(self, vq_ids):
        q = F.embedding(vq_ids, self.centers)
        return F.linear(q, self.weight, self.bias)

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
        raise KeyError(
            "No pretrained linear decoder found. Expected decoder_state_dict "
            "or model['decoder.weight'/'decoder.bias'] in the codebook."
        )
    return state["weight"], state["bias"]


def accumulate_topk(logits, targets, k=5):
    top = logits.topk(min(k, logits.size(-1)), dim=-1).indices
    return (
        int(top[:, 0].eq(targets).sum().item()),
        int(top.eq(targets[:, None]).any(dim=1).sum().item()),
    )


@torch.no_grad()
def evaluate(model, decoder, loader, device, mixture_topk=32):
    model.eval()
    totals = dict(count=0, vq_loss=0.0, vq_top1=0, vq_top5=0,
                  bpe_loss=0.0, bpe_top1=0, bpe_top5=0,
                  mix_bpe_loss=0.0, mix_bpe_top1=0, mix_bpe_top5=0,
                  oracle_bpe_loss=0.0, oracle_bpe_top1=0, oracle_bpe_top5=0)

    for vq_in, bpe_gap, vq_y, bpe_y, valid in tqdm(
        loader, desc="[eval]", leave=False
    ):
        vq_in, vq_y = vq_in.to(device), vq_y.to(device)
        bpe_gap = bpe_gap.to(device)
        bpe_y, valid = bpe_y.to(device), valid.to(device)
        logits = model(vq_in, bpe_gap, key_padding_mask=~valid)
        mask = vq_y.ne(-100)
        vl, vt = logits[mask], vq_y[mask]
        bt = bpe_y[mask]
        n = int(vt.numel())

        totals["count"] += n
        totals["vq_loss"] += float(F.cross_entropy(vl, vt, reduction="sum").item())
        a, b = accumulate_topk(vl, vt)
        totals["vq_top1"] += a; totals["vq_top5"] += b

        pred_bpe = decoder(vl.argmax(dim=-1))
        totals["bpe_loss"] += float(F.cross_entropy(pred_bpe, bt, reduction="sum").item())
        a, b = accumulate_topk(pred_bpe, bt)
        totals["bpe_top1"] += a; totals["bpe_top5"] += b

        k = min(int(mixture_topk), vl.size(-1))
        top_values, top_ids = vl.topk(k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        candidate_centers = F.embedding(top_ids, decoder.centers)
        mixed_center = torch.sum(
            top_weights.unsqueeze(-1) * candidate_centers, dim=1
        )
        mixed_bpe = decoder.decode_centers(mixed_center)
        totals["mix_bpe_loss"] += float(
            F.cross_entropy(mixed_bpe, bt, reduction="sum").item()
        )
        a, b = accumulate_topk(mixed_bpe, bt)
        totals["mix_bpe_top1"] += a; totals["mix_bpe_top5"] += b

        oracle_bpe = decoder(vt)
        totals["oracle_bpe_loss"] += float(
            F.cross_entropy(oracle_bpe, bt, reduction="sum").item()
        )
        a, b = accumulate_topk(oracle_bpe, bt)
        totals["oracle_bpe_top1"] += a; totals["oracle_bpe_top5"] += b

    n = max(totals["count"], 1)
    result = {"count": totals["count"]}
    for prefix in ("vq", "bpe", "mix_bpe", "oracle_bpe"):
        ce = totals[f"{prefix}_loss"] / n
        result[f"{prefix}_loss"] = ce
        result[f"{prefix}_ppl"] = math.exp(min(ce, 20.0))
        result[f"{prefix}_top1"] = totals[f"{prefix}_top1"] / n
        result[f"{prefix}_top5"] = totals[f"{prefix}_top5"] / n
    return result


def main():
    ap = argparse.ArgumentParser()
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
    if args.mixture_topk < 1:
        ap.error("--mixture_topk must be >= 1")

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    codebook = torch.load(args.codebook, map_location="cpu", weights_only=False)
    if int(data.get("hop", -1)) != 10 or int(codebook.get("hop", -1)) != 10:
        raise ValueError(
            f"HOP10 required: data={data.get('hop')}, codebook={codebook.get('hop')}"
        )
    centers = codebook["global_centers"].float()
    vq_ids = data["vq_ids_flat"].long().reshape(-1)
    token_ids = data["token_ids_flat"].long().reshape(-1)
    if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= centers.size(0):
        raise ValueError("VQ IDs are outside the HOP10 codebook")
    dec_weight, dec_bias = load_decoder(codebook)
    decoder = FrozenCenterToBPE(centers, dec_weight, dec_bias).to(device).eval()

    samples = list(data["samples"])
    random.shuffle(samples)
    n = len(samples); n_train = int(0.8 * n); n_valid = int(0.1 * n)
    splits = (samples[:n_train], samples[n_train:n_train+n_valid], samples[n_train+n_valid:])
    datasets = [GapVQARDataset(
                    s, token_ids, vq_ids, centers.size(0), args.gap,
                    args.local_bpe_tokens, args.max_len
                )
                for s in splits]
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),
                          num_workers=4, pin_memory=True, persistent_workers=True)
               for i, ds in enumerate(datasets)]

    model = PureVQWAR(
        centers, dec_weight.size(0), args.local_bpe_tokens,
        args.d_model, args.n_layers, args.n_heads, args.dropout, args.max_len
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    history, best_valid = [], float("inf")
    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
    print(
        f"[architecture] HOP10 VQW-AR through t-{args.gap} + "
        f"local {args.local_bpe_tokens} BPE -> CAT -> VQW -> frozen decoder"
    )
    print(f"[decoder] frozen linear {centers.size(1)} -> {dec_weight.size(0)}")
    print(f"[mixture] topk={args.mixture_topk}")

    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; total_count = 0
        pbar = tqdm(loaders[0], desc=f"[train] epoch {epoch}/{args.epochs}")
        for vq_in, bpe_gap, vq_y, _bpe_y, valid in pbar:
            vq_in, vq_y, valid = vq_in.to(device), vq_y.to(device), valid.to(device)
            bpe_gap = bpe_gap.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(vq_in, bpe_gap, key_padding_mask=~valid)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   vq_y.reshape(-1), ignore_index=-100)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(vq_y.ne(-100).sum().item())
            total_loss += float(loss.item()) * count; total_count += count
            pbar.set_postfix(vq=f"{total_loss / max(total_count, 1):.4f}")

        valid_metrics = evaluate(
            model, decoder, loaders[1], device, args.mixture_topk
        )
        test_metrics = evaluate(
            model, decoder, loaders[2], device, args.mixture_topk
        )
        print(
            f"[epoch {epoch}] valid_vq_ppl={valid_metrics['vq_ppl']:.4f} "
            f"valid_vq_top1={valid_metrics['vq_top1']:.4f} "
            f"valid_bpe_ppl={valid_metrics['bpe_ppl']:.4f} "
            f"valid_bpe_top1={valid_metrics['bpe_top1']:.4f} "
            f"valid_mix_bpe_ppl={valid_metrics['mix_bpe_ppl']:.4f} "
            f"valid_mix_bpe_top1={valid_metrics['mix_bpe_top1']:.4f} "
            f"test_vq_ppl={test_metrics['vq_ppl']:.4f} "
            f"test_vq_top1={test_metrics['vq_top1']:.4f} "
            f"test_bpe_ppl={test_metrics['bpe_ppl']:.4f} "
            f"test_bpe_top1={test_metrics['bpe_top1']:.4f} "
            f"test_mix_bpe_ppl={test_metrics['mix_bpe_ppl']:.4f} "
            f"test_mix_bpe_top1={test_metrics['mix_bpe_top1']:.4f} "
            f"oracle_bpe_ppl={test_metrics['oracle_bpe_ppl']:.4f}"
        )
        history.append({"epoch": epoch, "train_vq_loss": total_loss/max(total_count, 1),
                        "valid": valid_metrics, "test": test_metrics})
        checkpoint = {
            "model": model.state_dict(), "args": vars(args), "history": history,
            "architecture": "hop10_vqwar_gap11_local10bpe_cat_frozen_decoder",
            "vq_vocab_size": int(centers.size(0)),
            "token_vocab_size": int(dec_weight.size(0)),
            "decoder_frozen": True, "decoder_source": args.codebook,
            "last_valid": valid_metrics, "last_test": test_metrics,
        }
        torch.save(checkpoint, args.out)
        if valid_metrics["vq_ppl"] < best_valid:
            best_valid = valid_metrics["vq_ppl"]
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")
    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()