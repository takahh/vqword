#!/usr/bin/env python3
"""
Input-CAT BPE-only autoregressive language model.

Inputs:
    BPE[t] embedding
    Single-HOP VQ context from one frozen center table

Fusion:
    CAT(BPE embedding, aggregated multi-hop VQ context)
    -> learned projection
    -> shared causal Transformer
    -> BPE[t+1] head only

Frozen:
    - selected HOP codebook centers

Trainable:
    - BPE embedding
    - selected-HOP projection and scale
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

class SequenceARDataset(Dataset):
    """
    1サンプル = 最大max_len位置のAR系列。

    tok_in[:, q] から tok_y[:, q] を予測する。
    tok_y[:, q] = tok_inの次トークン。

    各入力位置には、その位置自身のselected-HOP VQ IDを保持する。
    実際にどのqueryから使用可能かはattention内で距離制御する。
    """

    def __init__(
        self,
        samples,
        token_ids_flat,
        vq_ids_flat,
        tok_pad_id,
        vq_pad_id,
        max_len=255,
    ):
        self.max_len = int(max_len)
        self.tok_pad_id = int(tok_pad_id)
        self.vq_pad_id = int(vq_pad_id)

        token_ids_flat = token_ids_flat.long().reshape(-1)
        vq_ids_flat = vq_ids_flat.long().reshape(-1)

        self.examples = []

        for sample in samples:
            start = int(sample["start"])
            end = int(sample["end"])

            # input+targetに最低2 token必要
            if end - start < 2:
                continue

            # 1チャンクにはinput max_len + 最後のtarget 1個が必要
            chunk_token_count = self.max_len + 1

            for chunk_start in range(
                start,
                end - 1,
                self.max_len,
            ):
                chunk_end = min(
                    chunk_start + chunk_token_count,
                    end,
                )

                tokens = token_ids_flat[
                    chunk_start:chunk_end
                ]

                if tokens.numel() < 2:
                    continue

                tok_in = tokens[:-1]
                tok_y = tokens[1:]

                # 入力位置そのもののselected-HOP ID
                vq_in = vq_ids_flat[
                    chunk_start:chunk_end - 1
                ]

                length = tok_in.numel()

                padded_tok_in = torch.full(
                    (self.max_len,),
                    self.tok_pad_id,
                    dtype=torch.long,
                )

                padded_vq_in = torch.full(
                    (self.max_len,),
                    self.vq_pad_id,
                    dtype=torch.long,
                )

                padded_tok_y = torch.full(
                    (self.max_len,),
                    -100,
                    dtype=torch.long,
                )

                attention_mask = torch.zeros(
                    self.max_len,
                    dtype=torch.bool,
                )

                # 右padding
                padded_tok_in[:length] = tok_in
                padded_vq_in[:length] = vq_in
                padded_tok_y[:length] = tok_y
                attention_mask[:length] = True

                self.examples.append(
                    (
                        padded_tok_in,
                        padded_vq_in,
                        padded_tok_y,
                        attention_mask,
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

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


class CausalTransformerBlock(nn.Module):
    """Standard pre-norm causal Transformer block."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, causal_mask, key_padding_mask=None):
        h = self.norm1(x)
        attn_output, _ = self.attention(
            query=h,
            key=h,
            value=h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class BPEVQWDistancePairAddLM(nn.Module):
    """
    Projection -> CAT -> MLP -> causal Transformer.

    To keep the same no-leak condition as the previous distance-aware
    attention, VQW at source position k is shifted to input position k+hop.
    Therefore query q can only receive VQW sources k <= q-hop.
    """

    def __init__(
        self,
        centers,
        token_vocab_size,
        target_vq_vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=255,
        tie_weights=False,
        use_vqw=False,
        vqw_init_scale=0.1,
        hop=10,
    ):
        super().__init__()

        self.use_vqw = bool(use_vqw)
        self.hop = int(hop)
        if self.hop < 0:
            raise ValueError(f"hop must be non-negative: {self.hop}")

        self.token_vocab_size = int(token_vocab_size)
        self.vq_vocab_size = int(target_vq_vocab_size)
        self.tok_pad_id = self.token_vocab_size
        self.vq_pad_id = self.vq_vocab_size
        self.d_model = int(d_model)
        self.tie_weights = bool(tie_weights)

        # BPE -> Linear
        self.tok_emb = nn.Embedding(
            self.token_vocab_size + 1,
            d_model,
            padding_idx=self.tok_pad_id,
        )
        self.bpe_projection = nn.Linear(d_model, d_model, bias=False)

        # VQW frozen centers -> Linear
        self.center_embedding = FrozenCenterEmbedding(centers)
        self.vqw_projection = nn.Linear(
            centers.size(1), d_model, bias=False
        )
        self.vqw_scale = nn.Parameter(
            torch.tensor(float(vqw_init_scale))
        )

        # learnable scalar gate
        # position-wise scalar gate
        # [B,L,2D] -> [B,L,1]
        self.vqw_gate = nn.Linear(
            2 * d_model,
            1,
            bias=True,
        )

        # 初期gateをだいたい0.1にしたい場合
        initial_gate = float(vqw_init_scale)
        initial_gate = min(max(initial_gate, 1e-6), 1.0 - 1e-6)

        with torch.no_grad():
            self.vqw_gate.weight.zero_()
            self.vqw_gate.bias.fill_(
                math.log(initial_gate / (1.0 - initial_gate))
            )
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.shared_blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.shared_norm = nn.LayerNorm(d_model)

        if not self.tie_weights:
            self.bpe_head = nn.Linear(
                d_model, self.token_vocab_size, bias=False
            )
        self.bpe_bias = nn.Parameter(
            torch.zeros(self.token_vocab_size)
        )

    def _shift_vqw(self, vqw_x, vqw_valid):
        """Move VQW[k] to position k+hop to enforce q-k >= hop."""
        if self.hop == 0:
            return vqw_x, vqw_valid

        shifted_x = torch.zeros_like(vqw_x)
        shifted_valid = torch.zeros_like(vqw_valid)

        if self.hop < vqw_x.size(1):
            shifted_x[:, self.hop:, :] = vqw_x[:, :-self.hop, :]
            shifted_valid[:, self.hop:] = vqw_valid[:, :-self.hop]

        return shifted_x, shifted_valid

    def forward(self, tok_in, vq_in, key_padding_mask=None):
        _, seq_len = tok_in.shape

        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len "
                f"{self.pos_emb.num_embeddings}"
            )

        pos = torch.arange(seq_len, device=tok_in.device)[None, :]

        # BPE -> Linear
        bpe_x = self.bpe_projection(self.tok_emb(tok_in))

        # VQW -> Linear, then shift by HOP for leakage prevention
        raw_vqw_valid = vq_in.ne(self.vq_pad_id)
        if self.use_vqw:
            vqw_x = self.vqw_projection(self.center_embedding(vq_in))
            vqw_x = vqw_x * raw_vqw_valid.unsqueeze(-1).to(vqw_x.dtype)
            vqw_x, vqw_valid = self._shift_vqw(vqw_x, raw_vqw_valid)
            vqw_x = self.vqw_scale * vqw_x
        else:
            vqw_x = torch.zeros_like(bpe_x)
            vqw_valid = torch.zeros_like(raw_vqw_valid)

        # position-wise scalar gate
        # gate_input: [B,L,2D]
        gate_input = torch.cat(
            [bpe_x, vqw_x],
            dim=-1,
        )

        # gate: [B,L,1]
        gate = torch.sigmoid(
            self.vqw_gate(gate_input)
        )

        # VQWが存在しない位置ではgate=0
        gate = (
                gate
                * vqw_valid.unsqueeze(-1).to(gate.dtype)
        )

        shared_h = (
                bpe_x
                + gate * vqw_x
                + self.pos_emb(pos)
        )

        # True above diagonal means masked for MultiheadAttention.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tok_in.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.shared_blocks:
            shared_h = block(
                x=shared_h,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        shared_h = self.shared_norm(shared_h)

        if self.tie_weights:
            bpe_logits = F.linear(
                shared_h,
                self.tok_emb.weight[:self.token_vocab_size],
                self.bpe_bias,
            )
        else:
            bpe_logits = self.bpe_head(shared_h) + self.bpe_bias

        return {
            "bpe_logits": bpe_logits,
            "hidden": shared_h,
            "bpe_input": bpe_x,
            "vqw_input": vqw_x,
            "vqw_valid": vqw_valid,
            "gate": gate,
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
            vq_in,
            tok_y,
            attention_mask,
    ) in tqdm(loader, desc="[eval]", leave=False):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        attention_mask = attention_mask.to(device)

        output = model(
            tok_in=tok_in,
            vq_in=vq_in,
            key_padding_mask=~attention_mask,
        )

        bpe_logits = output["bpe_logits"]

        flat_logits = bpe_logits.reshape(
            -1,
            bpe_logits.size(-1),
        )

        flat_targets = tok_y.reshape(-1)

        valid_target = flat_targets.ne(-100)

        valid_logits = flat_logits[valid_target]
        valid_targets = flat_targets[valid_target]

        bpe_loss = F.cross_entropy(
            valid_logits,
            valid_targets,
            reduction="sum",
        )

        bpe_topk = valid_logits.topk(
            min(5, valid_logits.size(-1)),
            dim=-1,
        ).indices

        batch_count = valid_targets.numel()

        total_bpe_loss += float(bpe_loss.item())
        total_count += batch_count

        total_bpe_top1 += int(
            bpe_topk[:, 0]
            .eq(valid_targets)
            .sum()
            .item()
        )

        total_bpe_top5 += int(
            bpe_topk
            .eq(valid_targets[:, None])
            .any(dim=1)
            .sum()
            .item()
        )

    bpe_ce = total_bpe_loss / max(total_count, 1)

    return {
        "bpe_loss": bpe_ce,
        "bpe_ppl": math.exp(min(bpe_ce, 20.0)),
        "bpe_top1": (
            total_bpe_top1 / max(total_count, 1)
        ),
        "bpe_top5": (
            total_bpe_top5 / max(total_count, 1)
        ),
        "count": total_count,
    }

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--hop_data",
        required=True,
        help="Selected-HOP TinyStories VQ ID file",
    )
    ap.add_argument(
        "--hop_codebook",
        required=True,
        help="Selected-HOP codebook checkpoint file",
    )
    ap.add_argument(
        "--hop",
        type=int,
        required=True,
        help="HOP number and minimum query-key distance for VQW use",
    )
    ap.add_argument(
        "--vqw_init_scale",
        type=float,
        default=0.1,
        help="Initial learnable scale for the VQW attention contribution.",
    )
    ap.add_argument(
        "--out",
        default="ar_bpe_singlehop_input_cat_bpe_only.pt",
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
    if args.hop < 0:
        raise ValueError(f"hop must be non-negative: {args.hop}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[device] {device}")

    reference = torch.load(
        args.hop_data, map_location="cpu", weights_only=False
    )
    samples = list(reference["samples"])
    token_ids_flat = reference["token_ids_flat"].long().reshape(-1)
    vq_ids_flat = reference["vq_ids_flat"].long().reshape(-1)

    recorded_hop = int(reference.get("hop", -1))
    if recorded_hop != args.hop:
        raise ValueError(
            f"Expected HOP{args.hop} data, metadata says HOP{recorded_hop}"
        )

    raw = torch.load(
        args.hop_codebook, map_location="cpu", weights_only=False
    )
    codebook_hop = int(raw.get("args", {}).get("hop", -1))
    if codebook_hop != args.hop:
        raise ValueError(
            f"Expected HOP{args.hop} codebook, metadata says HOP{codebook_hop}"
        )
    centers = raw["global_centers"].float()
    vq_vocab_size = int(centers.size(0))

    vq_min = int(vq_ids_flat.min().item())
    vq_max = int(vq_ids_flat.max().item())
    if vq_min < 0 or vq_max >= vq_vocab_size:
        raise ValueError(
            f"HOP{args.hop}: VQ IDs out of range: "
            f"{vq_min}..{vq_max}, vocab={vq_vocab_size}"
        )

    token_vocab_size = int(
        reference.get("token_vocab_size", 50257)
    )

    print("[architecture] BPE->Linear, VQW->Linear, CAT->MLP, causal Transformer, BPE head")
    print(f"[token vocab size] {token_vocab_size}")
    print(f"[VQ vocab size] {vq_vocab_size}")
    print(f"[use VQW] {bool(args.use_vqw)}")
    print(f"[VQW initial scale] {args.vqw_init_scale}")
    print(f"[HOP] {args.hop}")
    print(f"[VQ range] {vq_min}..{vq_max}, used={torch.unique(vq_ids_flat).numel():,}")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    tok_pad_id = token_vocab_size
    vq_pad_id = vq_vocab_size

    train_ds = SequenceARDataset(
        samples=train_samples,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_len=args.max_len,
    )

    valid_ds = SequenceARDataset(
        samples=valid_samples,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_len=args.max_len,
    )

    test_ds = SequenceARDataset(
        samples=test_samples,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_len=args.max_len,
    )

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    train_loader = make_loader(train_ds, True)
    valid_loader = make_loader(valid_ds, False)
    test_loader = make_loader(test_ds, False)

    model = BPEVQWDistancePairAddLM(
        centers=centers,
        token_vocab_size=token_vocab_size,
        target_vq_vocab_size=vq_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,

        # 系列長
        max_len=args.max_len,

        tie_weights=args.tie_weights,
        use_vqw=bool(args.use_vqw),
        vqw_init_scale=args.vqw_init_scale,
        hop=args.hop,
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
                vq_in,
                tok_y,
                attention_mask,
        ) in pbar:
            tok_in = tok_in.to(device)
            vq_in = vq_in.to(device)
            tok_y = tok_y.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(
                tok_in=tok_in,
                vq_in=vq_in,
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
        print(f"[gate] {torch.sigmoid(model.vqw_gate).item():.4f}")
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
            "architecture": "bpe_vqw_projection_cat_mlp_causal",
            "history": history,
            "token_vocab_size": token_vocab_size,
            "vq_vocab_size": vq_vocab_size,
            "tok_pad_id": tok_pad_id,
            "vq_pad_id": vq_pad_id,
            "hop": args.hop,
            "hop_data_source": args.hop_data,
            "hop_codebook_source": args.hop_codebook,
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
