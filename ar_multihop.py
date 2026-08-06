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

class DistancePairedWindowDataset(Dataset):
    def __init__(
        self,
        samples,
        token_ids_flat,
        hop_vq_ids_flat,
        tok_pad_id,
        vq_pad_id,
        max_context=11,
    ):
        self.max_context = int(max_context)

        token_ids_flat = token_ids_flat.long().reshape(-1)
        # HOP10のID列だけを使用
        hop10_vq_ids = hop_vq_ids_flat[10].long().reshape(-1)
        all_tok = []
        all_vq = []
        all_hop = []
        all_y = []
        all_mask = []

        distances = torch.arange(
            max_context,
            0,
            -1,
            dtype=torch.long,
        )
        # max_context=255なら [255,254,...,1]

        # targetから11個以上離れた位置だけHOP10を使う
        use_hop10_by_distance = distances.ge(11)
        # [255]の場合:
        # distance 255..11 -> True
        # distance 10..1   -> False

        for sample in samples:
            start = int(sample["start"])
            end = int(sample["end"])

            if end - start <= 1:
                continue

            target_pos = torch.arange(
                start + 1,
                end,
                dtype=torch.long,
            )
            # [T]

            source_pos = (
                target_pos[:, None]
                - distances[None, :]
            )
            # [T, 11]

            valid = source_pos.ge(start)

            safe_pos = source_pos.clamp_min(start)

            tok = token_ids_flat[safe_pos]

            # 各source位置に対応するHOP10 VQ ID
            vq = hop10_vq_ids[safe_pos]

            # [T, max_context]
            hop10_valid = (
                    valid
                    & use_hop10_by_distance[None, :]
            )

            # HOP10を使う位置は10、それ以外は-1
            hops = torch.full_like(
                source_pos,
                -1,
            )

            hops = hops.masked_fill(
                hop10_valid,
                10,
            )

            # HOP10を使わない位置のVQ IDはpadding
            vq = vq.masked_fill(
                ~hop10_valid,
                vq_pad_id,
            )

            tok = tok.masked_fill(
                ~valid,
                tok_pad_id,
            )

            all_tok.append(tok)
            all_vq.append(vq)
            all_hop.append(hops)
            all_y.append(token_ids_flat[target_pos])
            all_mask.append(valid)

        self.tok_in = torch.cat(all_tok, dim=0)
        self.vq_in = torch.cat(all_vq, dim=0)
        self.hop_ids = torch.cat(all_hop, dim=0)
        self.tok_y = torch.cat(all_y, dim=0)
        self.attention_mask = torch.cat(all_mask, dim=0)

    def __len__(self):
        return self.tok_y.size(0)

    def __getitem__(self, index):
        return (
            self.tok_in[index],
            self.vq_in[index],
            self.hop_ids[index],
            self.tok_y[index],
            self.attention_mask[index],
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

class DistanceAwareVQAttention(nn.Module):
    """
    Causal self-attention with query-key-distance-aware VQW values.

    For prediction row q:

        key q     -> HOP0 VQW
        key q-1   -> HOP1 VQW
        ...
        key q-10  -> HOP10 VQW

    BPE values participate at every causal key position.
    VQW values participate only for distances 0..10.
    """

    def __init__(
            self,
            d_model,
            n_heads,
            dropout=0.1,
            num_hops=11,
            vqw_init_scale=0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by "
                f"n_heads={n_heads}"
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.num_hops = int(num_hops)

        # Learnable VQW contribution scale.
        # Start smaller than 1.0 so that BPE attention remains dominant initially.
        self.vqw_scale = nn.Parameter(
            torch.tensor(float(vqw_init_scale))
        )

        # Standard BPE-hidden-state attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.bpe_v_proj = nn.Linear(d_model, d_model, bias=False)

        # VQW contribution for each relative distance/HOP
        self.vqw_v_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(self.num_hops)
        ])

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        [B, L, D] -> [B, H, L, Dh]
        """
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
        """
        [B, H, L, Dh] -> [B, L, D]
        """
        batch_size, _, seq_len, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

    def forward(
        self,
        x,
        vqw_features,
        hop_valid,
        use_vqw=True,
        key_padding_mask=None,
    ):
        """
        x:
            [B, L, D]

        vqw_features:
            list of 11 tensors.
            vqw_features[h] has shape [B, L, D].

            At prediction row q, vqw_features[h][:, q]
            already represents the VQW from source position q-h.

        hop_valid:
            [B, L, 11]

        key_padding_mask:
            [B, L], True means padding.
        """

        batch_size, seq_len, _ = x.shape

        # -----------------------------------------------------
        # Standard attention Q, K and BPE Value
        # -----------------------------------------------------

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        bpe_v = self._split_heads(self.bpe_v_proj(x))

        # [B, H, Q, K]
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        # Standard causal mask:
        # query q may see key k only when k <= q.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            float("-inf"),
        )

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        attn_weights = torch.softmax(scores, dim=-1)

        # 全keyがmaskされたpadding queryで発生するNaNを除去
        attn_weights = torch.nan_to_num(
            attn_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # padding queryの出力自体も0にする
        if key_padding_mask is not None:
            query_valid = (~key_padding_mask)[:, None, :, None]
            attn_weights = attn_weights * query_valid.to(attn_weights.dtype)

        attn_weights = self.attn_dropout(attn_weights)
        # -----------------------------------------------------
        # Ordinary BPE Value contribution
        # -----------------------------------------------------

        # [B, H, Q, Dh]
        output = torch.matmul(attn_weights, bpe_v)

        # -----------------------------------------------------
        # Exact distance-dependent VQW contribution
        # -----------------------------------------------------

        if use_vqw:
            vqw_output = torch.zeros_like(output)

            for hop in range(self.num_hops):
                # Valid query positions for this HOP:
                #
                # hop=0:
                #   query 0..L-1, key=query
                #
                # hop=1:
                #   query 1..L-1, key=query-1
                #
                # hop=10:
                #   query 10..L-1, key=query-10

                if hop >= seq_len:
                    break

                query_index = torch.arange(
                    hop,
                    seq_len,
                    device=x.device,
                )

                key_index = query_index - hop

                # Attention weight for exactly this distance:
                #
                # [B, H, L-hop]
                distance_weight = attn_weights[
                    :,
                    :,
                    query_index,
                    key_index,
                ]

                # vqw_features[hop][:, q] is already the
                # source q-hop HOP-hop representation.
                vqw_h = self.vqw_v_projections[hop](
                    vqw_features[hop]
                )

                # Keep only valid query rows for this HOP.
                vqw_h = vqw_h[:, query_index, :]

                # [B, H, L-hop, Dh]
                vqw_h = self._split_heads(vqw_h)

                valid_h = hop_valid[
                    :,
                    query_index,
                    hop,
                ]

                # [B, 1, L-hop, 1]
                valid_h = valid_h[
                    :,
                    None,
                    :,
                    None,
                ].to(vqw_h.dtype)

                contribution = (
                    distance_weight.unsqueeze(-1)
                    * vqw_h
                    * valid_h
                )

                # Add left padding in the query-position dimension.
                #
                # hop=0: no padding
                # hop=1: one empty query row at the beginning
                # hop=2: two empty query rows at the beginning
                contribution = F.pad(
                    contribution,
                    pad=(0, 0, hop, 0),
                )

                vqw_output = vqw_output + contribution

            # output = output + vqw_output
            output = output + self.vqw_scale * vqw_output

        output = self._merge_heads(output)
        output = self.out_proj(output)

        return output

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
            vqw_init_scale=0.1,
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
        # BPE 256次元 + HOP10 VQW 256次元
        # を連結して256次元へ戻す
        self.input_fusion = nn.Linear(
            2 * d_model,
            d_model,
            bias=True,
        )
        # ---------------- Shared Transformer ----------------
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.shared_blocks = nn.ModuleList([
            DistanceAwareVQTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                num_hops=self.num_hops,
                vqw_init_scale=vqw_init_scale,
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
            hop_ids,
            key_padding_mask=None,
    ):
        """
        tok_in:
            [B, L]
            各過去位置のBPE ID

        vq_in:
            [B, L]
            各過去位置とペアになるVQW ID

        hop_ids:
            [B, L]

            target直前位置はHOP0
            2個前はHOP1
            ...
            11個前はHOP10

            padding位置は-1
        """

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

        # --------------------------------------------------
        # BPE embedding
        # --------------------------------------------------

        bpe_x = self.tok_emb(tok_in)

        # --------------------------------------------------
        # 各位置に対応するVQW embedding
        # --------------------------------------------------

        # --------------------------------------------------
        # HOP10だけを使用
        #
        # 11個前の位置:
        #     CAT(BPE, HOP10 VQW)
        #
        # その他の位置:
        #     CAT(BPE, zero)
        # --------------------------------------------------

        vqw_x = torch.zeros_like(bpe_x)

        if self.use_vqw:
            # [B, L]
            # 通常はcolumn 0だけTrue。
            # 文頭付近で11個前が存在しない場合は全てFalse。
            hop10_valid = hop_ids.eq(10)

            # HOP10以外の位置をpadding IDにする
            hop10_ids = vq_in.masked_fill(
                ~hop10_valid,
                self.vq_pad_id,
            )

            # HOP10コードブックだけを参照
            hop10_center = self.center_embeddings[10](
                hop10_ids
            )

            hop10_projected = self.hop_projections[10](
                hop10_center
            )

            # 念のためHOP10以外を厳密にゼロ化
            vqw_x = (
                    hop10_projected
                    * hop10_valid.unsqueeze(-1).to(
                hop10_projected.dtype
            )
            )

        # --------------------------------------------------
        # CAT(BPE, HOP10-VQW) -> Linear -> position追加
        # --------------------------------------------------

        fused_input = torch.cat(
            [bpe_x, vqw_x],
            dim=-1,
        )
        # [B, L, 2 * d_model]

        shared_h = (
                self.input_fusion(fused_input)
                + self.pos_emb(pos)
        )
        # [B, L, d_model]

        # --------------------------------------------------
        # 通常のcausal Transformer
        #
        # VQWはすでに入力で加えたので、
        # attention内部では使わない
        # --------------------------------------------------

        dummy_vqw_features = [
            torch.zeros_like(shared_h)
            for _ in range(self.num_hops)
        ]

        dummy_hop_valid = torch.zeros(
            batch_size,
            seq_len,
            self.num_hops,
            dtype=torch.bool,
            device=shared_h.device,
        )

        for block in self.shared_blocks:
            shared_h = block(
                x=shared_h,
                vqw_features=dummy_vqw_features,
                hop_valid=dummy_hop_valid,
                use_vqw=False,
                key_padding_mask=key_padding_mask,
            )

        shared_h = self.shared_norm(shared_h)

        # 左paddingなので、最後の位置は必ずターゲット直前
        last_hidden = shared_h[:, -1, :]

        # 1サンプルにつき次BPEを1個予測
        bpe_logits = (
                self.bpe_head(last_hidden)
                + self.bpe_bias
        )

        return {
            "bpe_logits": bpe_logits,
            "last_hidden": last_hidden,
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
        hop_ids,
        tok_y,
        attention_mask,
    ) in tqdm(loader, desc="[eval]", leave=False):

        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        hop_ids = hop_ids.to(device)
        tok_y = tok_y.to(device)
        attention_mask = attention_mask.to(device)

        output = model(
            tok_in=tok_in,
            vq_in=vq_in,
            hop_ids=hop_ids,
            key_padding_mask=~attention_mask,
        )

        bpe_logits = output["bpe_logits"]

        bpe_loss = F.cross_entropy(
            bpe_logits,
            tok_y,
            reduction="sum",
        )

        bpe_topk = bpe_logits.topk(
            min(5, bpe_logits.size(-1)),
            dim=-1,
        ).indices

        batch_count = tok_y.numel()

        total_bpe_loss += float(bpe_loss.item())
        total_count += batch_count

        total_bpe_top1 += int(
            bpe_topk[:, 0]
            .eq(tok_y)
            .sum()
            .item()
        )

        total_bpe_top5 += int(
            bpe_topk
            .eq(tok_y[:, None])
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
        nargs=11,
        required=True,
        help="HOP0 ... HOP10 TinyStories VQ ID files",
    )
    ap.add_argument(
        "--vqw_init_scale",
        type=float,
        default=0.1,
        help="Initial learnable scale for the VQW attention contribution.",
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
    print(f"[VQW initial scale] {args.vqw_init_scale}")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    tok_pad_id = token_vocab_size
    vq_pad_id = vq_vocab_size

    train_ds = DistancePairedWindowDataset(
        samples=train_samples,
        token_ids_flat=token_ids_flat,
        hop_vq_ids_flat=hop_vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_context=args.max_len,
    )

    valid_ds = DistancePairedWindowDataset(
        samples=valid_samples,
        token_ids_flat=token_ids_flat,
        hop_vq_ids_flat=hop_vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_context=args.max_len,
    )

    test_ds = DistancePairedWindowDataset(
        samples=test_samples,
        token_ids_flat=token_ids_flat,
        hop_vq_ids_flat=hop_vq_ids_flat,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_context=args.max_len,
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
        hop_centers=hop_centers,
        token_vocab_size=token_vocab_size,
        target_vq_vocab_size=vq_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,

        # 入力は最大11位置
        max_len=args.max_len,
        
        tie_weights=args.tie_weights,
        use_vqw=bool(args.use_vqw),
        vqw_init_scale=args.vqw_init_scale,
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
                hop_ids,
                tok_y,
                attention_mask,
        ) in pbar:
            tok_in = tok_in.to(device)
            vq_in = vq_in.to(device)
            hop_ids = hop_ids.to(device)
            tok_y = tok_y.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(
                tok_in=tok_in,
                vq_in=vq_in,
                hop_ids=hop_ids,
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
            "architecture": "bpe_hop10_distance11_input_cat",
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
