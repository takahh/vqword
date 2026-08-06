#!/usr/bin/env python3
"""
Input-CAT BPE-only autoregressive language model.

Inputs:
    BPE[t] embedding
    Multi-hop VQ context from HOP0..HOP10 frozen centers

Fusion:
    CAT(BPE embedding, aggregated multi-hop VQ context)
    -> learned projection
    -> shared causal Transformer
    -> BPE[t+1] head only

Frozen:
    - all HOP0..HOP10 codebook centers

Trainable:
    - BPE embedding
    - HOP projections and gates
    - input fusion projection
    - shared causal Transformer
    - BPE output head
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


class MultiHopTwoStreamDataset(Dataset):
    def __init__(
        self,
        samples,
        token_ids_flat,
        hop_vq_ids_flat,
        vq_pad_id,
        max_len=255,
        target_hop=10,
    ):
        if len(hop_vq_ids_flat) != 11:
            raise ValueError("Exactly 11 HOP ID tensors are required")

        self.samples = []
        self.token_ids_flat = token_ids_flat.long().reshape(-1)
        self.hop_vq_ids_flat = [
            x.long().reshape(-1) for x in hop_vq_ids_flat
        ]
        self.vq_pad_id = int(vq_pad_id)
        self.max_len = int(max_len)
        self.target_hop = int(target_hop)

        for hop, ids in enumerate(self.hop_vq_ids_flat):
            if ids.numel() != self.token_ids_flat.numel():
                raise ValueError(
                    f"HOP{hop}: VQ/token length mismatch: "
                    f"{ids.numel()} vs {self.token_ids_flat.numel()}"
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
        length = end - start

        # One training row corresponds to predicting local positions 1..length-1.
        target_positions = torch.arange(1, length)

        # For target position u, HOP h reads source u-h-1.
        # Shape: [L-1, 11]
        vq_context = torch.full(
            (length - 1, 11),
            self.vq_pad_id,
            dtype=torch.long,
        )
        hop_valid = torch.zeros(
            (length - 1, 11),
            dtype=torch.bool,
        )

        for hop in range(11):
            source_positions = target_positions - hop - 1
            valid = source_positions >= 0
            if valid.any():
                global_source = start + source_positions[valid]
                vq_context[valid, hop] = self.hop_vq_ids_flat[hop][
                    global_source
                ]
                hop_valid[valid, hop] = True

        tok_in = self.token_ids_flat[start:end - 1]
        tok_y = self.token_ids_flat[start + 1:end]

        # Same target definition as the earlier VQ AR:
        # predict HOP10 VQW at the next physical-token position.
        vq_y = self.hop_vq_ids_flat[self.target_hop][
            start + 1:end
        ]

        return {
            "tok_in": tok_in,
            "vq_context": vq_context,
            "hop_valid": hop_valid,
            "tok_y": tok_y,
            "vq_y": vq_y,
        }


def collate_batch(batch, tok_pad_id, vq_pad_id):
    batch_size = len(batch)
    max_len = max(item["tok_in"].numel() for item in batch)

    tok_in = torch.full(
        (batch_size, max_len),
        tok_pad_id,
        dtype=torch.long,
    )
    vq_context = torch.full(
        (batch_size, max_len, 11),
        vq_pad_id,
        dtype=torch.long,
    )
    hop_valid = torch.zeros(
        (batch_size, max_len, 11),
        dtype=torch.bool,
    )
    tok_y = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )
    vq_y = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
    )

    for i, item in enumerate(batch):
        n = item["tok_in"].numel()
        tok_in[i, :n] = item["tok_in"]
        vq_context[i, :n] = item["vq_context"]
        hop_valid[i, :n] = item["hop_valid"]
        tok_y[i, :n] = item["tok_y"]
        vq_y[i, :n] = item["vq_y"]
        attention_mask[i, :n] = True

    return (
        tok_in,
        vq_context,
        hop_valid,
        tok_y,
        vq_y,
        attention_mask,
    )


class FrozenCenterEmbedding(nn.Module):
    def __init__(self, centers):
        super().__init__()
        centers = F.normalize(centers.float(), dim=-1)
        self.padding_idx = int(centers.size(0))
        zero = torch.zeros(1, centers.size(1), dtype=centers.dtype)
        self.register_buffer(
            "weight",
            torch.cat([centers, zero], dim=0),
            persistent=False,
        )

    def forward(self, ids):
        return F.embedding(
            ids,
            self.weight,
            padding_idx=self.padding_idx,
        )


class BPEMultiHopInputCatLM(nn.Module):
    def __init__(
        self,
        hop_centers,
        token_vocab_size,
        target_vq_vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=255,
        tie_weights=False,
        use_vqw=False,
    ):
        super().__init__()

        if len(hop_centers) != 11:
            raise ValueError("Expected 11 HOP center matrices")
        self.use_vqw = use_vqw
        self.num_hops = 11
        self.token_vocab_size = int(token_vocab_size)
        self.vq_vocab_size = int(target_vq_vocab_size)
        self.tok_pad_id = self.token_vocab_size
        self.vq_pad_id = self.vq_vocab_size
        self.d_model = int(d_model)

        # ---------------- BPE input ----------------
        self.tok_emb = nn.Embedding(
            self.token_vocab_size + 1,
            d_model,
            padding_idx=self.tok_pad_id,
        )

        # ---------------- Multi-hop VQ input ----------------
        self.center_embeddings = nn.ModuleList([
            FrozenCenterEmbedding(centers)
            for centers in hop_centers
        ])
        self.hop_projections = nn.ModuleList([
            nn.Linear(centers.size(1), d_model, bias=False)
            for centers in hop_centers
        ])
        self.hop_gates = nn.Parameter(torch.zeros(self.num_hops))
        self.vq_context_norm = nn.LayerNorm(d_model)

        # CAT([BPE embedding, multi-hop VQ context]) -> d_model
        self.input_fusion_proj = nn.Linear(
            2 * d_model,
            d_model,
            bias=True,
        )
        self.input_fusion_norm = nn.LayerNorm(d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # ---------------- Shared Transformer ----------------
        shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.shared_transformer = nn.TransformerEncoder(
            shared_layer,
            num_layers=n_layers,
        )
        self.shared_norm = nn.LayerNorm(d_model)

        # ---------------- BPE output head only ----------------
        # shared Transformerの出力から直接BPE[t+1]を予測
        self.bpe_head = nn.Linear(
            d_model,
            self.token_vocab_size,
            bias=False,
        )

        self.bpe_bias = nn.Parameter(
            torch.zeros(self.token_vocab_size)
        )

        if tie_weights:
            self.bpe_head.weight = self.tok_emb.weight[
                :self.token_vocab_size
            ]

    def forward(
        self,
        tok_in,
        vq_context,
        hop_valid,
        key_padding_mask=None,
    ):
        batch_size, seq_len = tok_in.shape

        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len "
                f"{self.pos_emb.num_embeddings}"
            )

        pos = torch.arange(
            seq_len,
            device=tok_in.device,
        )[None, :]

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=tok_in.device,
            ),
            diagonal=1,
        )

        # ---------------- BPE representation ----------------
        bpe_x = self.tok_emb(tok_in)

        # ---------------- Multi-hop VQ aggregation ----------------
        context_h = torch.zeros(
            batch_size,
            seq_len,
            self.d_model,
            device=tok_in.device,
            dtype=bpe_x.dtype,
        )

        gate_values = F.softplus(self.hop_gates)
        valid_count = hop_valid.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)

        for hop in range(self.num_hops):
            ids = vq_context[:, :, hop]
            center_h = self.center_embeddings[hop](ids)
            projected = self.hop_projections[hop](center_h)
            valid = hop_valid[:, :, hop].unsqueeze(-1)

            context_h = context_h + (
                gate_values[hop]
                * projected
                * valid
            )

        context_h = context_h / valid_count.sqrt()
        context_h = self.vq_context_norm(context_h)

        # Comparison baseline:
        # keep the same CAT/projection architecture,
        # but remove all information from the VQ stream.
        if not self.use_vqw:
            context_h = torch.zeros_like(context_h)

        # ---------------- Input CAT and shared processing ----------------
        shared_x = torch.cat([bpe_x, context_h], dim=-1)
        shared_x = self.input_fusion_proj(shared_x)
        shared_x = self.input_fusion_norm(shared_x)
        shared_x = shared_x + self.pos_emb(pos)

        shared_h = self.shared_transformer(
            shared_x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        shared_h = self.shared_norm(shared_h)

        # ---------------- BPE prediction only ----------------
        bpe_logits = self.bpe_head(shared_h) + self.bpe_bias

        return {
            "bpe_logits": bpe_logits,
            "shared_hidden": shared_h,
            "bpe_input": bpe_x,
            "vq_context_hidden": context_h,
        }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_bpe_loss = 0.0
    total_count = 0
    total_bpe_top1 = 0
    total_bpe_top5 = 0

    for (
        tok_in,
        vq_context,
        hop_valid,
        tok_y,
        _vq_y,
        attention_mask,
    ) in tqdm(loader, desc="[eval]", leave=False):

        tok_in = tok_in.to(device)
        vq_context = vq_context.to(device)
        hop_valid = hop_valid.to(device)
        tok_y = tok_y.to(device)
        attention_mask = attention_mask.to(device)

        output = model(
            tok_in=tok_in,
            vq_context=vq_context,
            hop_valid=hop_valid,
            key_padding_mask=~attention_mask,
        )

        bpe_logits = output["bpe_logits"]

        bpe_loss = F.cross_entropy(
            bpe_logits.reshape(-1, bpe_logits.size(-1)),
            tok_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid = tok_y.ne(-100)
        n = int(valid.sum().item())

        bpe_pred = bpe_logits[valid]
        bpe_target = tok_y[valid]
        bpe_topk = bpe_pred.topk(
            min(5, bpe_pred.size(-1)),
            dim=-1,
        ).indices

        total_bpe_loss += float(bpe_loss.item())
        total_count += n

        total_bpe_top1 += int(
            bpe_topk[:, 0].eq(bpe_target).sum().item()
        )
        total_bpe_top5 += int(
            bpe_topk.eq(bpe_target[:, None]).any(dim=1).sum().item()
        )

    bpe_ce = total_bpe_loss / max(total_count, 1)

    return {
        "bpe_loss": bpe_ce,
        "bpe_ppl": math.exp(min(bpe_ce, 20.0)),
        "bpe_top1": total_bpe_top1 / max(total_count, 1),
        "bpe_top5": total_bpe_top5 / max(total_count, 1),
        "count": total_count,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--hop_data",
        nargs=11,
        required=True,
        help="HOP0 ... HOP10 TinyStories VQ ID files",
    )
    ap.add_argument(
        "--hop_codebooks",
        nargs=11,
        required=True,
        help="HOP0 ... HOP10 codebook checkpoint files",
    )
    ap.add_argument(
        "--out",
        default="ar_bpe_multihop_input_cat_bpe_only.pt",
    )
    ap.add_argument(
        "--use_vqw",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use VQ context as input (1=yes, 0=no).",
    )
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

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[device] {device}")

    hop_data = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.hop_data
    ]
    reference = hop_data[0]

    samples = list(reference["samples"])
    token_ids_flat = reference["token_ids_flat"].long().reshape(-1)
    hop_vq_ids_flat = [
        data["vq_ids_flat"].long().reshape(-1)
        for data in hop_data
    ]

    for hop, data in enumerate(hop_data):
        current_tokens = data["token_ids_flat"].long().reshape(-1)
        if not torch.equal(token_ids_flat, current_tokens):
            raise ValueError(
                f"HOP{hop}: token_ids_flat does not match HOP0"
            )

        if len(data["samples"]) != len(samples):
            raise ValueError(
                f"HOP{hop}: sample count mismatch"
            )

        for i, (ref_sample, current_sample) in enumerate(
            zip(samples, data["samples"])
        ):
            for key in ("sample_idx", "start", "end", "length"):
                if int(ref_sample[key]) != int(current_sample[key]):
                    raise ValueError(
                        f"HOP{hop}: sample metadata mismatch "
                        f"at sample={i}, key={key}"
                    )

        recorded_hop = int(data.get("hop", -1))
        if recorded_hop != hop:
            raise ValueError(
                f"Expected HOP{hop} data, metadata says HOP{recorded_hop}"
            )

    hop_centers = []
    vq_vocab_size = None
    center_dim = None

    for expected_hop, path in enumerate(args.hop_codebooks):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        actual_hop = int(raw.get("args", {}).get("hop", -1))
        if actual_hop != expected_hop:
            raise ValueError(
                f"Expected HOP{expected_hop} codebook, "
                f"but metadata says HOP{actual_hop}"
            )

        centers = raw["global_centers"].float()

        if vq_vocab_size is None:
            vq_vocab_size = int(centers.size(0))
            center_dim = int(centers.size(1))
        elif tuple(centers.shape) != (vq_vocab_size, center_dim):
            raise ValueError(
                f"HOP{expected_hop}: center shape mismatch: "
                f"{tuple(centers.shape)} vs "
                f"{(vq_vocab_size, center_dim)}"
            )

        hop_centers.append(centers)

    token_vocab_size = int(
        reference.get("token_vocab_size", 50257)
    )

    for hop, ids in enumerate(hop_vq_ids_flat):
        vq_min = int(ids.min().item())
        vq_max = int(ids.max().item())
        if vq_min < 0 or vq_max >= vq_vocab_size:
            raise ValueError(
                f"HOP{hop}: VQ IDs out of range: "
                f"{vq_min}..{vq_max}, vocab={vq_vocab_size}"
            )
        print(
            f"[HOP{hop}] VQ range={vq_min}..{vq_max}, "
            f"used={torch.unique(ids).numel():,}"
        )

    print("[architecture] input CAT(BPE, multi-hop VQ) + shared Transformer + BPE head only")
    print(f"[token vocab size] {token_vocab_size}")
    print(f"[VQ vocab size] {vq_vocab_size}")
    print(f"[use VQW] {bool(args.use_vqw)}")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    tok_pad_id = token_vocab_size
    vq_pad_id = vq_vocab_size

    train_ds = MultiHopTwoStreamDataset(
        train_samples,
        token_ids_flat,
        hop_vq_ids_flat,
        vq_pad_id,
        max_len=args.max_len,
        target_hop=10,
    )
    valid_ds = MultiHopTwoStreamDataset(
        valid_samples,
        token_ids_flat,
        hop_vq_ids_flat,
        vq_pad_id,
        max_len=args.max_len,
        target_hop=10,
    )
    test_ds = MultiHopTwoStreamDataset(
        test_samples,
        token_ids_flat,
        hop_vq_ids_flat,
        vq_pad_id,
        max_len=args.max_len,
        target_hop=10,
    )

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_batch(
                batch,
                tok_pad_id,
                vq_pad_id,
            ),
        )

    train_loader = make_loader(train_ds, True)
    valid_loader = make_loader(valid_ds, False)
    test_loader = make_loader(test_ds, False)

    model = BPEMultiHopInputCatLM(
        hop_centers=hop_centers,
        token_vocab_size=token_vocab_size,
        target_vq_vocab_size=vq_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        tie_weights=args.tie_weights,
        use_vqw=args.use_vqw,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_valid = float("inf")
    best_path = str(
        Path(args.out).with_name(
            Path(args.out).stem + "_best.pt"
        )
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_bpe_loss = 0.0
        running_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"[train] epoch {epoch}/{args.epochs}",
        )

        for (
            tok_in,
            vq_context,
            hop_valid,
            tok_y,
            vq_y,
            attention_mask,
        ) in pbar:

            tok_in = tok_in.to(device)
            vq_context = vq_context.to(device)
            hop_valid = hop_valid.to(device)
            tok_y = tok_y.to(device)
            # vq_y is intentionally unused in the BPE-only experiment.
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(
                tok_in=tok_in,
                vq_context=vq_context,
                hop_valid=hop_valid,
                key_padding_mask=~attention_mask,
            )

            bpe_logits = output["bpe_logits"]

            bpe_loss = F.cross_entropy(
                bpe_logits.reshape(-1, bpe_logits.size(-1)),
                tok_y.reshape(-1),
                ignore_index=-100,
            )

            bpe_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

            n_valid_tokens = int(tok_y.ne(-100).sum().item())
            running_bpe_loss += (
                float(bpe_loss.item()) * n_valid_tokens
            )
            running_count += n_valid_tokens

            pbar.set_postfix(
                bpe=f"{running_bpe_loss / max(running_count, 1):.4f}",
            )

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
        )

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
            "train_bpe_loss": (
                running_bpe_loss / max(running_count, 1)
            ),
            "valid": valid_metrics,
            "test": test_metrics,
        }
        history.append(record)

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "architecture": "bpe_multihop_input_cat_bpe_only",
            "history": history,
            "token_vocab_size": token_vocab_size,
            "vq_vocab_size": vq_vocab_size,
            "tok_pad_id": tok_pad_id,
            "vq_pad_id": vq_pad_id,
            "hop_data_sources": list(args.hop_data),
            "hop_codebook_sources": list(args.hop_codebooks),
            "vq_centers_frozen": True,
            "vq_used_as_input_only": True,
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
