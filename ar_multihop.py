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

    各入力位置には11個の距離bucketを保持する。
    bucket 0..9はHOP 1..10、bucket 10は距離11以上で
    HOP10を再利用する。
    """

    def __init__(
        self,
        samples,
        token_ids_flat,
        vq_ids_flat_by_hop,
        tok_pad_id,
        vq_pad_id,
        max_len=255,
    ):
        self.max_len = int(max_len)
        self.tok_pad_id = int(tok_pad_id)
        self.vq_pad_id = int(vq_pad_id)

        token_ids_flat = token_ids_flat.long().reshape(-1)
        if len(vq_ids_flat_by_hop) != 11:
            raise ValueError("Expected 11 VQ distance buckets")
        vq_ids_flat_by_hop = [
            ids.long().reshape(-1) for ids in vq_ids_flat_by_hop
        ]
        if any(ids.numel() != token_ids_flat.numel()
               for ids in vq_ids_flat_by_hop):
            raise ValueError("Token/VQ flattened lengths do not match")

        self.token_ids_flat = token_ids_flat
        self.vq_ids_flat_by_hop = vq_ids_flat_by_hop

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

                length = chunk_end - chunk_start - 1
                if length < 1:
                    continue
                self.examples.append((chunk_start, length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        chunk_start, length = self.examples[index]
        input_end = chunk_start + length

        padded_tok_in = torch.full(
            (self.max_len,), self.tok_pad_id, dtype=torch.long
        )
        padded_vq_in = torch.full(
            (11, self.max_len), self.vq_pad_id, dtype=torch.long
        )
        padded_tok_y = torch.full(
            (self.max_len,), -100, dtype=torch.long
        )
        attention_mask = torch.zeros(self.max_len, dtype=torch.bool)

        padded_tok_in[:length] = self.token_ids_flat[
            chunk_start:input_end
        ]
        padded_tok_y[:length] = self.token_ids_flat[
            chunk_start + 1:input_end + 1
        ]
        for hop_index, ids in enumerate(self.vq_ids_flat_by_hop):
            padded_vq_in[hop_index, :length] = ids[
                chunk_start:input_end
            ]
        attention_mask[:length] = True

        return (
            padded_tok_in,
            padded_vq_in,
            padded_tok_y,
            attention_mask,
        )

class FrozenCenterEmbedding(nn.Module):
    def __init__(self, centers):
        super().__init__()

        centers = centers.float()

        self.padding_idx = int(centers.size(0))
        zero = torch.zeros(
            1,
            centers.size(1),
            dtype=centers.dtype,
        )

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
        VQW K/V, with distance-dependent HOP:
        distance 1..10 -> HOP 1..10, distance >=11 -> HOP10

    Queryは全headともBPE/shared hiddenから作る。
    これによりquery位置自身のVQWからのリークを防ぐ。
    """

    def __init__(
            self,
            d_model,
            n_heads,
            dropout=0.1,
            n_vqw_heads=None,
            vqw_init_scale=0.1,
            disable_second_branch=False,
            samidare_hop=True,
            distant_hop=10,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.disable_second_branch = bool(disable_second_branch)
        self.samidare_hop = bool(samidare_hop)
        self.distant_hop = int(distant_hop)
        if not 1 <= self.distant_hop <= 10:
            raise ValueError(
                f"distant_hop must be in 1..10: {self.distant_hop}"
            )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads

        # 8 headsならデフォルト4 BPE + 4 VQW
        if n_vqw_heads is None:
            n_vqw_heads = n_heads // 2

        self.n_vqw_heads = int(n_vqw_heads)
        self.n_bpe_heads = n_heads - self.n_vqw_heads

        if self.n_bpe_heads <= 0:
            raise ValueError("Need at least one BPE head")

        if self.n_vqw_heads < 0:
            raise ValueError("n_vqw_heads must be non-negative")
        bpe_dim = self.n_bpe_heads * self.head_dim
        vqw_dim = self.n_vqw_heads * self.head_dim

        # 8 head, d_model=256
        #
        # head_dim = 32
        # BPE 4 heads = 128
        # VQW 4 heads = 128

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

        # VQW headsはuse_vqw=1の場合のみ作成
        if self.n_vqw_heads > 0:
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
        else:
            self.q_vqw_proj = None
            self.k_vqw_proj = None
            self.v_vqw_proj = None
            self.register_parameter("vqw_scale", None)

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
        # VQW HEADS / HEAD CAT
        # =====================================================

        if self.n_vqw_heads > 0 and not self.disable_second_branch:
            # VQW headでもQueryはBPE/shared hiddenから作る
            qv = self._split_heads(
                self.q_vqw_proj(x),
                self.n_vqw_heads,
            )

            # vqw_features: [B, 11, L, D]
            # K/V候補をHOPごとに作り、query-key距離で一つを選ぶ。
            kv = self._split_heads(
                self.k_vqw_proj(vqw_features).reshape(B * 11, L, -1),
                self.n_vqw_heads,
            ).reshape(B, 11, self.n_vqw_heads, L, self.head_dim)

            vv = self._split_heads(
                self.v_vqw_proj(vqw_features).reshape(B * 11, L, -1),
                self.n_vqw_heads,
            ).reshape(B, 11, self.n_vqw_heads, L, self.head_dim)

            distance = qpos - kpos
            vqw_scores = torch.full(
                (B, self.n_vqw_heads, L, L),
                float("-inf"),
                device=device,
                dtype=qv.dtype,
            )
            selected_values = []

            if self.samidare_hop:
                # distance 1..10 -> HOP1..10, distance 11+ -> HOP10
                # Each selected HOP reaches at most the current query token,
                # never the next-token prediction target.
                for bucket_index in range(11):
                    if bucket_index < 10:
                        pair_mask = distance.eq(bucket_index + 1)
                    else:
                        pair_mask = distance.ge(11)

                    hop_scores = torch.matmul(
                        qv,
                        kv[:, bucket_index].transpose(-2, -1),
                    ) / math.sqrt(self.head_dim)
                    allowed = (
                        pair_mask[None, None, :, :]
                        & vqw_valid[:, bucket_index, None, None, :]
                    )
                    vqw_scores = torch.where(
                        allowed,
                        hop_scores,
                        vqw_scores,
                    )
                    selected_values.append(
                        (pair_mask, vv[:, bucket_index])
                    )
            else:
                # Fixed-HOP mode. For HOP h, keys with attention distance
                # smaller than h are masked because their bilateral context
                # could include the next-token prediction target. With HOP10,
                # distance>=10 corresponds to the target's 11th previous
                # token or earlier.
                bucket_index = self.distant_hop - 1
                pair_mask = distance.ge(self.distant_hop)
                hop_scores = torch.matmul(
                    qv,
                    kv[:, bucket_index].transpose(-2, -1),
                ) / math.sqrt(self.head_dim)
                allowed = (
                    pair_mask[None, None, :, :]
                    & vqw_valid[:, bucket_index, None, None, :]
                )
                vqw_scores = torch.where(
                    allowed,
                    hop_scores,
                    vqw_scores,
                )
                selected_values.append(
                    (pair_mask, vv[:, bucket_index])
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

            vqw_attn = torch.nan_to_num(
                vqw_attn,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            vqw_attn = self.attn_dropout(
                vqw_attn
            )

            vqw_out = torch.zeros_like(qv)
            for pair_mask, selected_v in selected_values:
                hop_attn = vqw_attn * pair_mask[None, None, :, :]
                vqw_out = vqw_out + torch.matmul(
                    hop_attn,
                    selected_v,
                )

            vqw_out = (
                    self.vqw_scale * vqw_out
            )

            # 4 BPE heads + 4 VQW heads
            all_heads = torch.cat(
                [bpe_out, vqw_out],
                dim=1,
            )
        elif self.n_vqw_heads > 0:
            # Pure BPE:
            # 4 BPE headsは使用し、残り4 headsはゼロ固定
            unused_out = bpe_out.new_zeros(
                B,
                self.n_vqw_heads,
                L,
                self.head_dim,
            )
            all_heads = torch.cat(
                [bpe_out, unused_out],
                dim=1,
            )
        else:
            all_heads = bpe_out

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
            n_vqw_heads=None,
            disable_second_branch=False,
            samidare_hop=True,
            distant_hop=10,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = HeadSplitDistanceAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            n_vqw_heads=n_vqw_heads,
            vqw_init_scale=vqw_init_scale,
            disable_second_branch=disable_second_branch,
            samidare_hop=samidare_hop,
            distant_hop=distant_hop,
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
    BPE causal Transformer with a distance-dependent VQW K/V branch.

    For query q and key k, distance 1..10 selects HOP 1..10;
    distance 11 or larger selects HOP10. Distance 0 is never allowed,
    so the VQW branch cannot leak the current position.
    """

    def __init__(
            self,
            centers_by_hop,
            token_vocab_size,
            target_vq_vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=255,
            tie_weights=False,
            use_vqw=False,
            pure_bpe_mode=False,
            samidare_hop=True,
            distant_hop=10,
            use_hop_embedding=False,
            vqw_init_scale=0.1,
    ):
        super().__init__()
        self.use_hop_embedding = bool(
            use_hop_embedding
        )
        self.use_vqw = bool(use_vqw)
        self.pure_bpe_mode = bool(pure_bpe_mode)
        self.samidare_hop = bool(samidare_hop)
        self.distant_hop = int(distant_hop)
        if not 1 <= self.distant_hop <= 10:
            raise ValueError(
                f"distant_hop must be in 1..10: {self.distant_hop}"
            )
        if self.pure_bpe_mode and self.use_vqw:
            raise ValueError("pure_bpe_mode=1 requires use_vqw=0")
        if self.use_hop_embedding and not self.use_vqw:
            raise ValueError(
                "use_hop_embedding=1 requires use_vqw=1"
            )
        self.token_vocab_size = int(token_vocab_size)
        self.vq_vocab_size = int(target_vq_vocab_size)
        self.tok_pad_id = self.token_vocab_size
        self.vq_pad_id = self.vq_vocab_size
        self.d_model = int(d_model)
        self.tie_weights = bool(tie_weights)

        self.tok_emb = nn.Embedding(
            self.token_vocab_size + 1,
            d_model,
            padding_idx=self.tok_pad_id,
        )

        # BPE側の学習可能なNN
        # 通常BPE attention用
        self.bpe_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # HOP制限付きBPE control枝用
        self.distant_bpe_projection = None
        if not self.pure_bpe_mode:
            self.distant_bpe_projection = nn.Linear(
                d_model,
                d_model,
                bias=False,
            )
        # 11 distance buckets: HOP1..10 plus HOP10 reuse for 11+.
        # VQW codebook center自体は固定。
        # L2正規化したcenterだけ、学習可能なLinearへ通す。
        if len(centers_by_hop) != 11:
            raise ValueError("Expected 11 codebook distance buckets")
        center_dim = int(centers_by_hop[0].size(1))
        if any(int(c.size(1)) != center_dim for c in centers_by_hop):
            raise ValueError("All HOP codebooks must have the same center dimension")

        if not self.pure_bpe_mode:
            self.center_embeddings = nn.ModuleList([
                FrozenCenterEmbedding(centers)
                for centers in centers_by_hop
            ])

            self.vqw_projection = nn.Linear(
                center_dim,
                d_model,
                bias=False,
            )
            # 実際のHOPはHOP1〜HOP10の10種類。
            # 11番目のbucketはHOP10の再利用なので、
            # HOP embeddingはHOP10と共有する。
            self.n_physical_hops = 10

            if self.use_hop_embedding:
                self.hop_embedding = nn.Embedding(
                    self.n_physical_hops,
                    d_model,
                )

                # 従来モデルと同じ初期状態から開始するためゼロ初期化。
                # 学習後はHOPごとに別々のベクトルへ更新される。
                nn.init.zeros_(
                    self.hop_embedding.weight
                )
            else:
                self.hop_embedding = None

        else:
            self.center_embeddings = nn.ModuleList()
            self.vqw_projection = None

            self.n_physical_hops = 10
            self.hop_embedding = None

        self.pos_emb = nn.Embedding(max_len, d_model)
        # Pure BPE: all heads use ordinary causal BPE K/V.
        # Other modes retain the split-head comparison design.
        # 常に4 BPE heads + 4 second-branch headsの構成を保つ
        attention_vqw_heads = n_heads // 2
        self.shared_blocks = nn.ModuleList([
            SingleHopTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                vqw_init_scale=vqw_init_scale,
                n_vqw_heads=attention_vqw_heads,
                disable_second_branch=self.pure_bpe_mode,
                samidare_hop=self.samidare_hop,
                distant_hop=self.distant_hop,
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
        # BPE embedding -> learned NN
        # 元のBPE embedding
        tok_e = self.tok_emb(tok_in)

        # 通常BPE headへ渡す特徴
        bpe_x = self.bpe_projection(tok_e)

        # ==========================================
        # 2. HOP制限付き第2枝
        # ==========================================

        if self.pure_bpe_mode:
            # The attention blocks have zero second-branch heads, so no
            # distance-limited BPE or VQW features enter the model.
            second_x = None
            second_valid = None
        elif self.use_vqw:
            # vq_in shape:
            # [batch, 11 buckets, sequence_length]
            #
            # bucket 0  = HOP1
            # bucket 1  = HOP2
            # ...
            # bucket 9  = HOP10
            # bucket 10 = 距離11以上用に再利用するHOP10

            second_valid = vq_in.ne(self.vq_pad_id)

            # ==============================================
            # 1. frozen centerを取得する
            # ==============================================

            vqw_center = torch.stack([
                self.center_embeddings[bucket_index](
                    vq_in[:, bucket_index]
                )
                for bucket_index in range(11)
            ], dim=1)

            # shape:
            # [B, 11, L, center_dim]

            # ==============================================
            # 2. frozen centerを共通d_model空間へ射影
            # ==============================================

            center_feature = self.vqw_projection(
                vqw_center
            )

            # shape:
            # [B, 11, L, d_model]

            # ==============================================
            # 3. 必要ならglobal ID embeddingを追加
            # ==============================================
            if self.use_hop_embedding:
                # bucketと物理HOPの対応：
                #
                # bucket 0  → HOP1  → embedding index 0
                # bucket 1  → HOP2  → embedding index 1
                # ...
                # bucket 9  → HOP10 → embedding index 9
                # bucket 10 → HOP10 → embedding index 9
                hop_indices = torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
                    dtype=torch.long,
                    device=vq_in.device,
                )

                hop_feature = self.hop_embedding(
                    hop_indices
                )

                # hop_feature:
                # [11, d_model]
                #
                # [1, 11, 1, d_model]へ変形し、
                # batch方向と系列長方向へbroadcastする。
                second_x = (
                        center_feature
                        + hop_feature[None, :, None, :]
                )
            else:
                second_x = center_feature
            # ==============================================
            # 4. padding位置を最終的にゼロ化
            # ==============================================
            second_x = (
                    second_x
                    * second_valid.unsqueeze(-1).to(
                second_x.dtype
            )
            )

        else:
            # BPE control条件：
            # BPE embedding -> dedicated Linear
            second_valid = tok_in.ne(self.tok_pad_id)

            second_x = self.distant_bpe_projection(
                tok_e
            )

            second_x = (
                    second_x
                    * second_valid.unsqueeze(-1).to(second_x.dtype)
            )

            # attention側の共通インターフェース [B, 11, L, D]
            second_x = second_x[:, None, :, :].expand(-1, 11, -1, -1)
            second_valid = second_valid[:, None, :].expand(-1, 11, -1)

        # Transformer本体への入力はBPE + positional embedding。
        # VQWは別枝として各attention blockへ渡す。
        shared_h = (
                bpe_x
                + self.pos_emb(pos)
        )

        for block in self.shared_blocks:
            shared_h = block(
                x=shared_h,
                vqw_features=second_x,
                vqw_valid=second_valid,
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
            "second_input": second_x,

            # 古い解析コードとの互換性用
            "vqw_input": second_x,
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
        "--hop_data_pattern",
        required=True,
        help=(
            "VQ ID path pattern for HOP 1..10. "
            "Use {hop:02d}, e.g. /data/...bilateral{hop:02d}..._ids.pt"
        ),
    )
    ap.add_argument(
        "--samidare_hop",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Use distance-dependent Samidare HOP assignment. "
            "1: distance 1..10 -> HOP1..10 and 11+ -> HOP10; "
            "0: use --distant_hop only at distances >= that HOP."
        ),
    )
    ap.add_argument(
        "--distant_hop",
        type=int,
        default=10,
        choices=range(1, 11),
        help=(
            "Fixed HOP used when --samidare_hop=0. Keys with attention "
            "distance smaller than this value are masked."
        ),
    )
    ap.add_argument(
        "--hop_codebook_pattern",
        required=True,
        help=(
            "Codebook path pattern for HOP 1..10. "
            "Use {hop:02d}, e.g. /data/...bilateral{hop:02d}....pt"
        ),
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
    ap.add_argument(
        "--pure_bpe_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Use four causal BPE heads and leave the four second-branch "
            "head slots unused (1=yes). Requires --use_vqw 0."
        ),
    )
    ap.add_argument(
        "--use_hop_embedding",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Add a trainable HOP1..HOP10 embedding "
            "to the projected frozen VQW center."
        ),
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

    if args.pure_bpe_mode and args.use_vqw:
        ap.error("--pure_bpe_mode 1 requires --use_vqw 0")
    if args.use_hop_embedding and not args.use_vqw:
        ap.error(
            "--use_hop_embedding 1 requires --use_vqw 1"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[device] {device}")

    # Available files are HOP1..HOP10. Attention needs 11 distance
    # buckets, so the last bucket (distance >= 11) reuses HOP10.
    source_hops = list(range(1, 11)) + [10]
    hop_data_paths = [
        args.hop_data_pattern.format(hop=hop) for hop in source_hops
    ]
    hop_codebook_paths = [
        args.hop_codebook_pattern.format(hop=hop) for hop in source_hops
    ]

    references = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in hop_data_paths
    ]
    reference = references[10]
    samples = list(reference["samples"])
    token_ids_flat = reference["token_ids_flat"].long().reshape(-1)
    vq_ids_flat_by_hop = []
    centers_by_hop = []
    vq_vocab_sizes = []

    reference_sample_bounds = [
        (int(s["start"]), int(s["end"])) for s in samples
    ]
    for bucket_index, (source_hop, hop_reference, codebook_path) in enumerate(
        zip(source_hops, references, hop_codebook_paths)
    ):
        recorded_hop = int(hop_reference.get("hop", -1))
        if recorded_hop != source_hop:
            raise ValueError(
                f"Bucket {bucket_index}: expected HOP{source_hop} data, "
                f"metadata says HOP{recorded_hop}"
            )
        hop_tokens = hop_reference["token_ids_flat"].long().reshape(-1)
        if not torch.equal(hop_tokens, token_ids_flat):
            raise ValueError(
                f"Bucket {bucket_index}/HOP{source_hop}: "
                "token_ids_flat differs from HOP10"
            )
        hop_bounds = [
            (int(s["start"]), int(s["end"]))
            for s in hop_reference["samples"]
        ]
        if hop_bounds != reference_sample_bounds:
            raise ValueError(
                f"Bucket {bucket_index}/HOP{source_hop}: "
                "sample boundaries differ from HOP10"
            )

        raw = torch.load(
            codebook_path, map_location="cpu", weights_only=False
        )
        codebook_hop = int(raw.get("args", {}).get("hop", -1))
        if codebook_hop != source_hop:
            raise ValueError(
                f"Bucket {bucket_index}: expected HOP{source_hop} "
                f"codebook, metadata says HOP{codebook_hop}"
            )
        centers = raw["global_centers"].float()
        ids = hop_reference["vq_ids_flat"].long().reshape(-1)
        vocab_size = int(centers.size(0))
        vq_min = int(ids.min().item())
        vq_max = int(ids.max().item())
        if vq_min < 0 or vq_max >= vocab_size:
            raise ValueError(
                f"Bucket {bucket_index}/HOP{source_hop}: VQ IDs out of range: "
                f"{vq_min}..{vq_max}, vocab={vocab_size}"
            )
        vq_ids_flat_by_hop.append(ids)
        centers_by_hop.append(centers)
        vq_vocab_sizes.append(vocab_size)

    if len(set(vq_vocab_sizes)) != 1:
        raise ValueError(f"HOP codebook sizes differ: {vq_vocab_sizes}")
    vq_vocab_size = vq_vocab_sizes[0]

    token_vocab_size = int(
        reference.get("token_vocab_size", 50257)
    )

    if args.pure_bpe_mode:
        architecture_name = "pure_bpe_4heads_plus_4unused"
        architecture_description = (
            "pure BPE: 4 causal BPE heads + 4 zero/unused head slots"
        )
    elif args.use_vqw:
        if args.samidare_hop:
            architecture_name = "bpe_vqw_samidare_distance_attention"
            architecture_description = (
                "4 causal BPE heads + 4 Samidare VQW heads"
            )
        else:
            architecture_name = "bpe_vqw_fixed_distant_hop_attention"
            architecture_description = (
                f"4 causal BPE heads + 4 fixed-HOP{args.distant_hop} "
                f"VQW heads at distance>={args.distant_hop}"
            )
    else:
        if args.samidare_hop:
            architecture_name = "bpe_samidare_distance_control_attention"
            architecture_description = (
                "4 causal BPE heads + 4 Samidare-distance BPE control heads"
            )
        else:
            architecture_name = "bpe_fixed_distant_control_attention"
            architecture_description = (
                f"4 causal BPE heads + 4 BPE control heads at "
                f"distance>={args.distant_hop}"
            )
    print(
        f"[architecture] {architecture_description}; "
        f"samidare_hop={args.samidare_hop}; "
        f"distant_hop={args.distant_hop}"
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
        vq_ids_flat_by_hop=vq_ids_flat_by_hop,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_len=args.max_len,
    )

    valid_ds = SequenceARDataset(
        samples=valid_samples,
        token_ids_flat=token_ids_flat,
        vq_ids_flat_by_hop=vq_ids_flat_by_hop,
        tok_pad_id=tok_pad_id,
        vq_pad_id=vq_pad_id,
        max_len=args.max_len,
    )

    test_ds = SequenceARDataset(
        samples=test_samples,
        token_ids_flat=token_ids_flat,
        vq_ids_flat_by_hop=vq_ids_flat_by_hop,
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
        centers_by_hop=centers_by_hop,
        token_vocab_size=token_vocab_size,
        target_vq_vocab_size=vq_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
        tie_weights=args.tie_weights,
        use_vqw=bool(args.use_vqw),
        pure_bpe_mode=bool(args.pure_bpe_mode),
        samidare_hop=bool(args.samidare_hop),
        distant_hop=args.distant_hop,
        use_hop_embedding=bool(
            args.use_hop_embedding
        ),
        vqw_init_scale=args.vqw_init_scale,
    ).to(device)

    print(
        "[HOP embedding] "
        f"enabled={args.use_hop_embedding} "
        f"physical_hops=10 "
        f"dim={args.d_model} "
        f"parameters={10 * args.d_model}"
    )
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
        if args.use_vqw and args.samidare_hop:
            if args.use_hop_embedding:
                architecture_name = (
                    "bpe_vqw_samidare_hop_embedding"
                )
                architecture_description = (
                    "4 causal BPE heads + 4 Samidare VQW heads "
                    "+ lightweight HOP embedding"
                )
            else:
                architecture_name = (
                    "bpe_vqw_samidare_distance_attention"
                )
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "architecture": architecture_name,
            "history": history,
            "token_vocab_size": token_vocab_size,
            "vq_vocab_size": vq_vocab_size,
            "tok_pad_id": tok_pad_id,
            "vq_pad_id": vq_pad_id,
            "hop_distance_mapping": (
                {
                    "mode": "samidare",
                    "distance_1_to_10": "HOP1_to_HOP10",
                    "distance_11_plus": "HOP10",
                }
                if args.samidare_hop
                else {
                    "mode": "fixed_distant_hop",
                    "fixed_hop": args.distant_hop,
                    "minimum_attention_distance": args.distant_hop,
                    "prediction_relative_first_key": args.distant_hop + 1,
                    "closer_distances": "masked",
                }
            ),
            "hop_data_sources": hop_data_paths,
            "hop_codebook_sources": hop_codebook_paths,
            "vq_centers_frozen": True,
            "vq_used_as_input_only": True,
            "last_valid": valid_metrics,
            "last_test": test_metrics,
            "samidare_hop": bool(args.samidare_hop),
            "distant_hop": args.distant_hop,
            "pure_bpe_mode": bool(args.pure_bpe_mode),
            "use_vqw": bool(args.use_vqw),
            "use_hop_embedding": bool(
                args.use_hop_embedding
            ),
            "hop_embedding_count": (
                10
                if args.use_hop_embedding
                else 0
            ),
            "hop_embedding_dim": (
                args.d_model
                if args.use_hop_embedding
                else 0
            ),
            "hop_embedding_parameters": (
                10 * args.d_model
                if args.use_hop_embedding
                else 0
            ),
            "hop_embedding_init": (
                "zeros"
                if args.use_hop_embedding
                else None
            ),
        }

        torch.save(checkpoint, args.out)

        if valid_metrics["bpe_ppl"] < best_valid:
            best_valid = valid_metrics["bpe_ppl"]
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")

    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()
