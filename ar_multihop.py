#!/usr/bin/env python3
"""
NN-only-to-BPE autoregressive language model.

Inputs:
    learned projection of BPE[t] embedding
    normalized frozen Single-HOP VQ center (no learned input projection)

Fusion:
    BPE branch and raw VQW-center branch
    -> distance-aware attention
    -> shared causal Transformer
    -> BPE[t+1] head only

Frozen:
    - selected HOP codebook centers

Trainable:
    - BPE embedding
    - BPE input projection
    - VQW attention scale
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


class HeadSplitDistanceAwareAttention(nn.Module):
    """
    Head 0 .. n_bpe_heads-1:
        BPE K/V, normal causal attention

    Remaining heads:
        VQW K/V, only q-k >= hop

    Queryは全headともBPE/shared hiddenから作る。
    これによりquery位置自身のVQWからのリークを防ぐ。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        hop=50,
        n_vqw_heads=None,
        vqw_init_scale=0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.hop = int(hop)

        # 8 headsならデフォルト4 BPE + 4 VQW
        if n_vqw_heads is None:
            n_vqw_heads = n_heads // 2

        self.n_vqw_heads = int(n_vqw_heads)
        self.n_bpe_heads = n_heads - self.n_vqw_heads

        if self.n_bpe_heads <= 0 or self.n_vqw_heads <= 0:
            raise ValueError("Need at least one BPE head and one VQW head")

        bpe_dim = self.n_bpe_heads * self.head_dim
        vqw_dim = self.n_vqw_heads * self.head_dim

        # 8 head, d_model=256
        #
        # head_dim = 32
        # BPE 4 heads = 128
        # VQW 4 heads = 128

        # BPE heads
        self.q_bpe_proj = nn.Linear(
            d_model,
            bpe_dim,
            bias=False,
        )

        self.k_bpe_proj = nn.Linear(
            d_model,
            bpe_dim,
            bias=False,
        )

        self.v_bpe_proj = nn.Linear(
            d_model,
            bpe_dim,
            bias=False,
        )

        # VQW heads
        self.q_vqw_proj = nn.Linear(
            d_model,
            vqw_dim,
            bias=False,
        )

        self.k_vqw_proj = nn.Linear(
            d_model,
            vqw_dim,
            bias=False,
        )

        self.v_vqw_proj = nn.Linear(
            d_model,
            vqw_dim,
            bias=False,
        )

        self.vqw_scale = nn.Parameter(
            torch.tensor(float(vqw_init_scale))
        )

        # 全headsをCATした後256次元へ
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=True,
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x, n_heads):
        B, L, _ = x.shape

        return (
            x.view(B, L, n_heads, self.head_dim)
            .transpose(1, 2)
        )

    def _merge_heads(self, x):
        B, H, L, HD = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, L, H * HD)
        )

    def forward(
            self,
            x,
            vqw_features,
            vqw_valid,
            key_padding_mask=None,
    ):
        B, L, _ = x.shape
        device = x.device

        # ==========================================
        # CATされた512次元を二つに分ける
        # ==========================================

        pos = torch.arange(
            L,
            device=device,
        )

        qpos = pos[:, None]
        kpos = pos[None, :]

        # =====================================================
        # BPE HEADS
        # =====================================================

        qb = self._split_heads(
            self.q_bpe_proj(x),
            self.n_bpe_heads,
        )

        kb = self._split_heads(
            self.k_bpe_proj(x),
            self.n_bpe_heads,
        )

        vb = self._split_heads(
            self.v_bpe_proj(x),
            self.n_bpe_heads,
        )

        bpe_scores = torch.matmul(
            qb,
            kb.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        # 普通のcausal
        bpe_allowed = (
                kpos <= qpos
        )

        bpe_scores = bpe_scores.masked_fill(
            ~bpe_allowed[None, None, :, :],
            float("-inf"),
        )

        if key_padding_mask is not None:
            bpe_scores = bpe_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        bpe_attn = torch.softmax(
            bpe_scores,
            dim=-1,
        )

        bpe_attn = torch.nan_to_num(
            bpe_attn,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        bpe_attn = self.attn_dropout(
            bpe_attn
        )

        bpe_out = torch.matmul(
            bpe_attn,
            vb,
        )

        # =====================================================
        # VQW HEADS
        # =====================================================

        # VQW headでもQueryはBPE/shared hiddenから作る
        qv = self._split_heads(
            self.q_vqw_proj(x),
            self.n_vqw_heads,
        )

        # K/VだけVQWから作る
        kv = self._split_heads(
            self.k_vqw_proj(vqw_features),
            self.n_vqw_heads,
        )

        vv = self._split_heads(
            self.v_vqw_proj(vqw_features),
            self.n_vqw_heads,
        )

        vqw_scores = torch.matmul(
            qv,
            kv.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        # -----------------------------------------
        # HOP50:
        #
        # target=t
        # query=t-1
        #
        # k=t-51
        # q-k=50
        #
        # → 51個目からVQW使用
        # -----------------------------------------

        vqw_allowed = (
                              qpos - kpos
                      ) >= self.hop

        vqw_scores = vqw_scores.masked_fill(
            ~vqw_allowed[None, None, :, :],
            float("-inf"),
        )

        # VQ IDが存在しない位置
        vqw_scores = vqw_scores.masked_fill(
            ~vqw_valid[:, None, None, :],
            float("-inf"),
        )

        if key_padding_mask is not None:
            vqw_scores = vqw_scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        vqw_attn = torch.softmax(
            vqw_scores,
            dim=-1,
        )

        # VQWを1個も利用できないqueryでは
        # softmax(-inf...)がnanになるので0にする
        vqw_attn = torch.nan_to_num(
            vqw_attn,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        vqw_attn = self.attn_dropout(
            vqw_attn
        )

        vqw_out = torch.matmul(
            vqw_attn,
            vv,
        )

        vqw_out = (
            self.vqw_scale * vqw_out
        )

        # =====================================================
        # HEAD CAT
        # =====================================================

        # [B, BPE_heads, L, HD]
        # [B, VQW_heads, L, HD]
        #
        # ↓ head軸でCAT
        #
        # [B, total_heads, L, HD]

        all_heads = torch.cat(
            [bpe_out, vqw_out],
            dim=1,
        )

        merged = self._merge_heads(
            all_heads
        )

        return self.out_proj(merged)

class SingleHopTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        vqw_init_scale=0.1,
        hop=50,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = HeadSplitDistanceAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            hop=hop,
            n_vqw_heads=n_heads // 2,
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
            vqw_valid,
            key_padding_mask=None,
    ):
        h = self.norm1(x)

        attn_out = self.attention(
            x=h,
            vqw_features=vqw_features,
            vqw_valid=vqw_valid,
            key_padding_mask=key_padding_mask,
        )

        x = x + self.dropout1(attn_out)

        x = x + self.dropout2(
            self.ffn(self.norm2(x))
        )

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

        # VQW codebook center自体は固定。
        # L2正規化したcenterだけ、学習可能なLinearへ通す。
        center_dim = int(centers.size(1))

        self.center_embedding = FrozenCenterEmbedding(centers)

        self.vqw_projection = nn.Linear(
            center_dim,
            d_model,
            bias=False,
        )

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

        if not self.tie_weights:
            self.bpe_head = nn.Linear(
                d_model, self.token_vocab_size, bias=False
            )
        self.bpe_bias = nn.Parameter(
            torch.zeros(self.token_vocab_size)
        )

    def forward(
            self,
            tok_in,
            vq_in,
            key_padding_mask=None,
    ):
        _, seq_len = tok_in.shape

        pos = torch.arange(
            seq_len,
            device=tok_in.device,
        )[None, :]

        # ==========================================
        # 1. BPE -> embeddingのみ（追加NNなし）
        # ==========================================
        bpe_x = self.tok_emb(tok_in)

        # ==========================================
        # 2. VQW -> frozen center -> learned NN
        # ==========================================
        vqw_valid = vq_in.ne(self.vq_pad_id)

        if self.use_vqw:
            # center_embedding内でL2正規化済み
            vqw_center = self.center_embedding(vq_in)

            # VQW側だけ学習可能なLinearを通す
            vqw_x = self.vqw_projection(vqw_center)

            # bias=Falseなのでpaddingは元々0のままだが、
            # 無効位置を明示的に0にしておく
            vqw_x = (
                    vqw_x
                    * vqw_valid.unsqueeze(-1).to(vqw_x.dtype)
            )
        else:
            vqw_x = torch.zeros_like(bpe_x)
            vqw_valid = torch.zeros_like(vqw_valid)

        # Transformer本体への入力はBPE + positional embedding。
        # VQWは別枝として各attention blockへ渡す。
        shared_h = (
                bpe_x
                + self.pos_emb(pos)
        )

        for block in self.shared_blocks:
            shared_h = block(
                x=shared_h,
                vqw_features=vqw_x,
                vqw_valid=vqw_valid,
                key_padding_mask=key_padding_mask,
            )

        shared_h = self.shared_norm(shared_h)

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

    print(
        "[architecture] "
        "BPE->learned embedding only, "
        "VQW->normalized frozen center->Linear, "
        "distance-aware causal Transformer, BPE head"
    )
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
            "architecture": "bpe_projection_raw_vqw_distance_attention_causal",
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

