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


class SingleHopDistanceAwareAttention(nn.Module):
    """
    通常のcausal attention。

    BPE value:
        全causal keyから利用

    selected-HOP VQ value:
        query-key距離が指定HOP以上のkeyだけ利用

    query qはtoken q+1を予測するため、
    q-k >= HOPなら、選択HOPの右文脈が予測対象へ届かない。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        vqw_init_scale=0.1,
        hop=10,
    ):
        super().__init__()

        self.hop = int(hop)
        if self.hop < 0:
            raise ValueError(f"hop must be non-negative: {self.hop}")

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"n_heads={n_heads}"
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.bpe_v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.vqw_v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # BPE outputとVQW outputをCATしてd_modelへ戻す
        self.out_proj = nn.Linear(
            2 * d_model,
            d_model,
            bias=True,
        )

        self.vqw_scale = nn.Parameter(
            torch.tensor(float(vqw_init_scale))
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape

        return (
            x.view(
                batch_size,
                seq_len,
                self.n_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.d_model,
            )
        )

    def forward(
        self,
        x,
        vqw_features,
        vqw_valid,
        key_padding_mask=None,
    ):
        """
        x:
            [B,L,D]

        vqw_features:
            各入力位置自身のselected-HOP特徴 [B,L,D]

        vqw_valid:
            実在するVQ ID位置 [B,L]
        """

        _, seq_len, _ = x.shape
        device = x.device

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        bpe_v = self._split_heads(self.bpe_v_proj(x))
        vqw_v = self._split_heads(
            self.vqw_v_proj(vqw_features)
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        positions = torch.arange(
            seq_len,
            device=device,
        )

        query_pos = positions[:, None]
        key_pos = positions[None, :]

        # 通常causal条件: key <= query
        causal_allowed = key_pos.le(query_pos)

        scores = scores.masked_fill(
            ~causal_allowed[None, None, :, :],
            float("-inf"),
        )

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attn_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attn_weights = torch.nan_to_num(
            attn_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if key_padding_mask is not None:
            query_valid = (
                ~key_padding_mask
            )[:, None, :, None]

            attn_weights = (
                attn_weights
                * query_valid.to(attn_weights.dtype)
            )

        attn_weights = self.attn_dropout(
            attn_weights
        )

        # 通常BPE value
        bpe_output = torch.matmul(
            attn_weights,
            bpe_v,
        )

        # q-k >= hop のときだけselected-HOP VQWを利用
        hop_allowed = (
            query_pos - key_pos
        ).ge(self.hop)

        vqw_weights = (
            attn_weights
            * hop_allowed[
                None,
                None,
                :,
                :,
            ].to(attn_weights.dtype)
        )

        # VQ IDが実在しないpadding keyも除外
        vqw_weights = (
            vqw_weights
            * vqw_valid[
                :,
                None,
                None,
                :,
            ].to(vqw_weights.dtype)
        )

        vqw_output = torch.matmul(
            vqw_weights,
            vqw_v,
        )

        bpe_output = self._merge_heads(
            bpe_output
        )

        vqw_output = self._merge_heads(
            vqw_output
        )

        # CAT方式
        combined = torch.cat(
            [
                bpe_output,
                self.vqw_scale * vqw_output,
            ],
            dim=-1,
        )

        return self.out_proj(combined)

class SingleHopTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        vqw_init_scale=0.1,
        hop=10,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = SingleHopDistanceAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            vqw_init_scale=vqw_init_scale,
            hop=hop,
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

    def forward(
        self,
        x,
        vqw_features,
        vqw_valid,
        key_padding_mask=None,
    ):
        attention_output = self.attention(
            x=self.norm1(x),
            vqw_features=vqw_features,
            vqw_valid=vqw_valid,
            key_padding_mask=key_padding_mask,
        )

        x = x + self.dropout1(attention_output)

        ffn_output = self.ffn(
            self.norm2(x)
        )

        x = x + self.dropout2(ffn_output)

        return x


class DistanceAwareVQTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        num_hops=11,
        vqw_init_scale=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = DistanceAwareVQAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            num_hops=num_hops,
            vqw_init_scale=vqw_init_scale,
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

    def forward(
        self,
        x,
        vqw_features,
        hop_valid,
        use_vqw=True,
        key_padding_mask=None,
    ):
        # Pre-norm attention
        attention_input = self.norm1(x)

        attention_output = self.attention(
            x=attention_input,
            vqw_features=vqw_features,
            hop_valid=hop_valid,
            use_vqw=use_vqw,
            key_padding_mask=key_padding_mask,
        )

        x = x + self.dropout1(attention_output)

        # Pre-norm feed-forward
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)

        return x

class BPEVQWDistancePairAddLM(nn.Module):
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

        self.use_vqw = use_vqw
        self.hop = int(hop)
        if self.hop < 0:
            raise ValueError(f"hop must be non-negative: {self.hop}")
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
        self.center_embedding = FrozenCenterEmbedding(centers)
        self.hop_projection = nn.Linear(
            centers.size(1), d_model, bias=False
        )
        # BPE 256次元 + selected-HOP VQW 256次元
        # を連結して256次元へ戻す
        self.input_fusion = nn.Linear(
            2 * d_model,
            d_model,
            bias=True,
        )
        # ---------------- Shared Transformer ----------------
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.shared_blocks = nn.ModuleList([
            SingleHopTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                vqw_init_scale=vqw_init_scale,
                hop=self.hop,
            )
            for _ in range(n_layers)
        ])

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
            vq_in,
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

        bpe_x = self.tok_emb(tok_in)

        vqw_valid = vq_in.ne(self.vq_pad_id)

        if self.use_vqw:
            hop_center = self.center_embedding(vq_in)
            vqw_x = self.hop_projection(hop_center)

            vqw_x = (
                    vqw_x
                    * vqw_valid.unsqueeze(-1).to(
                vqw_x.dtype
            )
            )
        else:
            vqw_x = torch.zeros_like(bpe_x)
            vqw_valid = torch.zeros_like(
                vqw_valid
            )

        # 最初のhiddenはBPEのみ
        shared_h = bpe_x + self.pos_emb(pos)

        for block in self.shared_blocks:
            shared_h = block(
                x=shared_h,
                vqw_features=vqw_x,
                vqw_valid=vqw_valid,
                key_padding_mask=key_padding_mask,
            )

        shared_h = self.shared_norm(shared_h)

        # [B,L,V]
        bpe_logits = (
                self.bpe_head(shared_h)
                + self.bpe_bias
        )

        return {
            "bpe_logits": bpe_logits,
            "hidden": shared_h,
            "bpe_input": bpe_x,
            "vqw_input": vqw_x,
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

    print("[architecture] input CAT(BPE, single-HOP VQ) + shared Transformer + BPE head only")
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
            "architecture": "bpe_singlehop_distance_input_cat",
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
