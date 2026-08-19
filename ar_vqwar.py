#!/usr/bin/env python3
"""Two leak-free HOP10 AR modes.

local_bpe_direct:
  BPE stream sees through BPE[t-1].
  VQW is available through local-VQW[t-11].
  BPE hidden + learned-alpha * VQW-only residual -> BPE[t]

global_vqwar:
  pair-global-ID(BPE, local-VQW)[t-11] + BPE[t-10:t-1]
  -> pair-global-ID(BPE, local-VQW)[t]
  -> exact BPE recovery from the predicted pair ID
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


class TwoStreamLocalDataset(Dataset):
    """Aligned BPE(t-1) and leak-free local-VQW(t-11) streams."""
    def __init__(self, samples, token_ids, local_vq_ids, vq_pad_id,
                 gap=11, max_len=255):
        self.token_ids = token_ids.long().reshape(-1)
        self.local_vq_ids = local_vq_ids.long().reshape(-1)
        self.vq_pad_id = int(vq_pad_id)
        self.gap = int(gap)
        self.max_len = int(max_len)
        self.bpe_input_len = self.max_len + self.gap - 1
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

        # For target t, this stream ends at BPE[t-1].  The first gap-1
        # positions warm up the causal BPE backbone for every chunk.
        bpe_in = torch.zeros(self.bpe_input_len, dtype=torch.long)
        bpe_valid = torch.zeros(self.bpe_input_len, dtype=torch.bool)
        bpe_length = length + self.gap - 1
        bpe_in[:bpe_length] = self.token_ids[
            source_start:source_start + bpe_length
        ]
        bpe_valid[:bpe_length] = True

        # At target t, source position is t-gap=t-11.  Each VQW input also
        # carries its own BPE identity because local cluster labels are only
        # meaningful within a BPE partition.
        vqw_bpe = torch.zeros(self.max_len, dtype=torch.long)
        local_vq = torch.full(
            (self.max_len,), self.vq_pad_id, dtype=torch.long
        )
        vqw_bpe[:length] = self.token_ids[
            source_start:source_start + length
        ]
        local_vq[:length] = self.local_vq_ids[
            source_start:source_start + length
        ]

        bpe_y = torch.full((self.max_len,), -100, dtype=torch.long)
        bpe_y[:length] = self.token_ids[
            target_start:target_start + length
        ]
        valid = torch.zeros(self.max_len, dtype=torch.bool)
        valid[:length] = True
        return bpe_in, vqw_bpe, local_vq, bpe_y, bpe_valid, valid


class SharedMaskedLocalDataset(Dataset):
    """Contiguous sequences for shared BPE/VQW masked attention."""
    def __init__(self, samples, token_ids, local_vq_ids, vq_pad_id,
                 max_len=255):
        self.token_ids = token_ids.long().reshape(-1)
        self.local_vq_ids = local_vq_ids.long().reshape(-1)
        self.vq_pad_id = int(vq_pad_id)
        self.max_len = int(max_len)
        self.examples = []
        step = self.max_len - 1
        for sample in samples:
            start, end = int(sample["start"]), int(sample["end"])
            for offset in range(0, max(end - start - 1, 0), step):
                length = min(self.max_len, end - start - offset)
                if length >= 2:
                    self.examples.append((start + offset, length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        start, length = self.examples[index]
        bpe = torch.zeros(self.max_len, dtype=torch.long)
        local = torch.full((self.max_len,), self.vq_pad_id, dtype=torch.long)
        valid = torch.zeros(self.max_len, dtype=torch.bool)
        bpe[:length] = self.token_ids[start:start + length]
        local[:length] = self.local_vq_ids[start:start + length]
        valid[:length] = True
        return bpe, local, valid


class FeatureCatLocalDataset(Dataset):
    """One target window: same-position BPE/local-VQW, recent VQW masked."""
    def __init__(self, samples, token_ids, local_vq_ids, vq_pad_id,
                 hop=10, max_len=255, train=False, eval_targets=8):
        self.token_ids = token_ids.long().reshape(-1)
        self.local_vq_ids = local_vq_ids.long().reshape(-1)
        self.vq_pad_id, self.hop, self.max_len = int(vq_pad_id), int(hop), int(max_len)
        self.train = bool(train)
        self.examples = []
        for sample in samples:
            start, end = int(sample["start"]), int(sample["end"])
            if end - start < 2:
                continue
            if self.train:
                self.examples.append((start, end, None))
            else:
                count = min(eval_targets, end - start - 1)
                for n in range(count):
                    target = start + 1 + ((n + 1) * (end - start - 1) - 1) // count
                    self.examples.append((start, end, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sample_start, sample_end, target = self.examples[index]
        if target is None:
            target = random.randrange(sample_start + 1, sample_end)
        context_start = max(sample_start, target - self.max_len)
        length = target - context_start
        offset = 0
        bpe = torch.zeros(self.max_len, dtype=torch.long)
        local = torch.full((self.max_len,), self.vq_pad_id, dtype=torch.long)
        valid = torch.zeros(self.max_len, dtype=torch.bool)
        vqw_available = torch.zeros(self.max_len, dtype=torch.bool)
        bpe[:length] = self.token_ids[context_start:target]
        local[:length] = self.local_vq_ids[context_start:target]
        valid[:length] = True
        positions = torch.arange(context_start, target)
        vqw_available[:length] = positions <= target - self.hop - 1
        return bpe, local, valid, vqw_available, self.token_ids[target]


class RecentBPEWindow(nn.Module):
    """Process distant representation + ten recent BPEs as 11 tokens."""
    def __init__(self, token_vocab_size, local_bpe_tokens, d_model,
                 n_heads, dropout, n_layers=2):
        super().__init__()
        self.local_bpe_tokens = int(local_bpe_tokens)
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.relative_pos = nn.Embedding(local_bpe_tokens + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, distant_h, bpe_gap):
        batch, length, _ = distant_h.shape
        recent = self.bpe_embedding(bpe_gap)
        window = torch.cat([distant_h.unsqueeze(2), recent], dim=2)
        rel = torch.arange(
            self.local_bpe_tokens + 1, device=bpe_gap.device
        )
        window = window + self.relative_pos(rel)[None, None]
        window = window.reshape(
            batch * length, self.local_bpe_tokens + 1, -1
        )
        window = self.transformer(window)
        return self.norm(window[:, 0]).reshape(batch, length, -1)


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


class ResidualLocalAR(nn.Module):
    def __init__(self, token_vocab_size, local_vq_vocab_size,
                 d_model=256, n_layers=6, n_heads=8, dropout=0.1,
                 max_len=255, use_vqw=True, alpha_init=0.5):
        super().__init__()
        self.use_vqw = bool(use_vqw)
        self.local_pad_id = int(local_vq_vocab_size)
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.local_embedding = nn.Embedding(
            local_vq_vocab_size + 1, d_model, padding_idx=self.local_pad_id
        )
        self.bpe_projection = nn.Linear(d_model, d_model)
        self.vqw_projection = nn.Linear(d_model, d_model)
        self.vqw_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # A bounded, interpretable scalar gate.  The matched BPE baseline keeps
        # the same module but forces the effective alpha to exactly zero.
        eps = 1e-6
        alpha_init = min(max(float(alpha_init), eps), 1.0 - eps)
        self.alpha_logit = nn.Parameter(torch.tensor(
            math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32
        ))
        self.input_norm = nn.LayerNorm(d_model)
        self.backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len
        )
        self.bpe_head = nn.Linear(d_model, token_vocab_size)

    def effective_alpha(self):
        if not self.use_vqw:
            return self.alpha_logit.new_zeros(())
        return torch.sigmoid(self.alpha_logit)

    def forward(self, bpe, local_vq, valid, vqw_available):
        bpe_h = self.bpe_projection(self.bpe_embedding(bpe))
        vqw_h = self.vqw_projection(self.local_embedding(local_vq))
        available = vqw_available.unsqueeze(-1).to(vqw_h.dtype)
        vqw_h = vqw_h * available

        # Keep BPE exclusively on the main path.  The residual branch receives
        # only local-VQW information, so an improvement cannot come from a
        # second nonlinear BPE path.  Mask the correction itself as well,
        # because Linear biases would otherwise make unavailable positions
        # nonzero again.
        delta = self.vqw_adapter(vqw_h)
        delta = delta * available
        h = self.input_norm(bpe_h + self.effective_alpha() * delta)
        h = self.backbone(h, ~valid)
        last = valid.long().sum(dim=1).sub(1)
        batch = torch.arange(h.size(0), device=h.device)
        return self.bpe_head(h[batch, last])


class SharedMaskedLocalAR(nn.Module):
    """Shared Transformer with BPE causal and HOP-delayed VQW attention."""
    def __init__(self, token_vocab_size, local_vq_vocab_size, hop=10,
                 d_model=256, n_layers=6, n_heads=8, dropout=0.1,
                 max_len=255):
        super().__init__()
        self.hop = int(hop)
        self.local_pad_id = int(local_vq_vocab_size)
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.local_embedding = nn.Embedding(
            local_vq_vocab_size + 1, d_model,
            padding_idx=self.local_pad_id,
        )
        self.position = nn.Embedding(max_len, d_model)
        self.stream_type = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.bpe_head = nn.Linear(d_model, token_vocab_size)

    def forward(self, bpe, local_vq, valid):
        length = bpe.size(1)
        pos = torch.arange(length, device=bpe.device)
        bpe_h = self.bpe_embedding(bpe) + self.position(pos)[None]
        vqw_h = self.local_embedding(local_vq) + self.position(pos)[None]
        bpe_h = bpe_h + self.stream_type.weight[0][None, None]
        vqw_h = vqw_h + self.stream_type.weight[1][None, None]
        h = torch.cat([bpe_h, vqw_h], dim=1)

        # BPE query q sees BPE keys <=q and VQW keys <=q-hop.
        # VQW query q sees only same-or-earlier BPE/VQW positions.
        mask = torch.ones(2 * length, 2 * length,
                          dtype=torch.bool, device=bpe.device)
        q = torch.arange(length, device=bpe.device)[:, None]
        k = torch.arange(length, device=bpe.device)[None, :]
        mask[:length, :length] = k > q
        mask[:length, length:] = k > (q - self.hop)
        mask[length:, :length] = k > q
        mask[length:, length:] = k > q
        padding = torch.cat([~valid, ~valid], dim=1)
        h = self.norm(self.transformer(
            h, mask=mask, src_key_padding_mask=padding
        ))
        return self.bpe_head(h[:, :length])


class LocalBPEDirectAR(nn.Module):
    def __init__(self, token_vocab_size, pair_vq_vocab_size,
                 local_bpe_tokens=10, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=255, window_layers=2):
        super().__init__()
        self.pair_vq_vocab_size = int(pair_vq_vocab_size)
        self.vq_pad_id = self.pair_vq_vocab_size
        self.gap = int(local_bpe_tokens) + 1
        self.bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.vqw_bpe_embedding = nn.Embedding(token_vocab_size, d_model)
        self.pair_vq_embedding = nn.Embedding(
            self.pair_vq_vocab_size + 1, d_model,
            padding_idx=self.vq_pad_id,
        )
        self.vqw_input_fusion = nn.Linear(2 * d_model, d_model)
        self.vqw_input_norm = nn.LayerNorm(d_model)
        self.bpe_backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len + self.gap - 1
        )
        self.vqw_backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len
        )
        self.stream_fusion = nn.Linear(2 * d_model, d_model)
        self.stream_norm = nn.LayerNorm(d_model)
        self.bpe_head = nn.Linear(d_model, token_vocab_size)

    def forward(self, bpe_in, vqw_bpe, pair_vq,
                bpe_key_padding_mask=None, vqw_key_padding_mask=None):
        h_bpe_all = self.bpe_backbone(
            self.bpe_embedding(bpe_in), bpe_key_padding_mask
        )
        # Output aligned to targets: position gap-1 ends at BPE[t-1].
        h_bpe = h_bpe_all[:, self.gap - 1:self.gap - 1 + pair_vq.size(1)]

        vqw_pair = self.vqw_input_norm(F.gelu(self.vqw_input_fusion(
            torch.cat([
                self.vqw_bpe_embedding(vqw_bpe),
                self.pair_vq_embedding(pair_vq),
            ], dim=-1)
        )))
        h_vqw = self.vqw_backbone(vqw_pair, vqw_key_padding_mask)

        h = self.stream_norm(F.gelu(self.stream_fusion(
            torch.cat([h_bpe, h_vqw], dim=-1)
        )))
        return self.bpe_head(h)


class GlobalVQWAR(nn.Module):
    def __init__(self, vq_vocab_size, token_vocab_size, local_bpe_tokens=10,
                 d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=255, window_layers=2):
        super().__init__()
        self.vq_vocab_size = int(vq_vocab_size)
        self.vq_pad_id = self.vq_vocab_size
        self.vq_embedding = nn.Embedding(
            self.vq_vocab_size + 1, d_model, padding_idx=self.vq_pad_id
        )
        self.backbone = ARBackbone(
            d_model, n_layers, n_heads, dropout, max_len
        )
        self.window = RecentBPEWindow(
            token_vocab_size, local_bpe_tokens, d_model,
            n_heads, dropout, window_layers
        )
        self.vq_head = nn.Linear(d_model, self.vq_vocab_size)

    def forward(self, vq_in, bpe_gap, key_padding_mask=None):
        h = self.backbone(self.vq_embedding(vq_in), key_padding_mask)
        h = self.window(h, bpe_gap)
        return self.vq_head(h)


def make_pair_global_ids(token_ids, local_vq_ids, local_vq_vocab_size):
    """Map every observed (BPE, local-VQW) pair to a compact global ID."""
    if int(local_vq_ids.min()) < 0:
        raise ValueError("negative local VQ ID")
    key = token_ids.long() * int(local_vq_vocab_size) + local_vq_ids.long()
    unique_keys, global_ids = torch.unique(
        key, sorted=True, return_inverse=True
    )
    global_to_bpe = torch.div(
        unique_keys, int(local_vq_vocab_size), rounding_mode="floor"
    )
    global_to_local = unique_keys.remainder(int(local_vq_vocab_size))
    return global_ids.long(), global_to_bpe.long(), global_to_local.long()


def accumulate_topk(logits, targets, k=5):
    top = logits.topk(min(k, logits.size(-1)), dim=-1).indices
    return int(top[:, 0].eq(targets).sum()), int(
        top.eq(targets[:, None]).any(dim=1).sum()
    )


def grouped_bpe_nll(pair_logits, bpe_targets, bpe_to_global):
    """NLL after marginalizing pair-global IDs that share the target BPE."""
    candidate_ids = bpe_to_global[bpe_targets]
    candidate_mask = candidate_ids.ge(0)
    safe_ids = candidate_ids.clamp_min(0)
    candidate_logits = pair_logits.gather(1, safe_ids)
    candidate_logits = candidate_logits.masked_fill(
        ~candidate_mask, float("-inf")
    )
    correct_bpe_log_prob = (
        torch.logsumexp(candidate_logits, dim=1)
        - torch.logsumexp(pair_logits, dim=1)
    )
    return -correct_bpe_log_prob.mean()


@torch.no_grad()
def evaluate_local(model, loader, device):
    model.eval()
    total = dict(count=0, loss=0.0, top1=0, top5=0)
    for bpe, local_vq, valid, vqw_available, target in tqdm(
        loader, desc="[eval local]", leave=False
    ):
        bpe, local_vq = bpe.to(device), local_vq.to(device)
        valid, vqw_available = valid.to(device), vqw_available.to(device)
        target = target.to(device)
        pred = model(bpe, local_vq, valid, vqw_available)
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
def evaluate_global(model, global_to_bpe, bpe_to_global, loader, device):
    model.eval()
    global_to_bpe = global_to_bpe.to(device)
    bpe_to_global = bpe_to_global.to(device)
    total = dict(count=0, vq_loss=0.0, vq_top1=0, vq_top5=0,
                 bpe_loss=0.0, bpe_top1=0)
    for _distant_bpe, vq_in, bpe_gap, vq_y, bpe_y, valid in tqdm(
        loader, desc="[eval global]", leave=False
    ):
        vq_in, vq_y = vq_in.to(device), vq_y.to(device)
        bpe_gap, bpe_y, valid = bpe_gap.to(device), bpe_y.to(device), valid.to(device)
        logits = model(vq_in, bpe_gap, ~valid)
        mask = vq_y.ne(-100)
        vl, vt, bt = logits[mask], vq_y[mask], bpe_y[mask]
        n = int(vt.numel()); total["count"] += n

        total["vq_loss"] += float(F.cross_entropy(vl, vt, reduction="sum"))
        a, b = accumulate_topk(vl, vt)
        total["vq_top1"] += a; total["vq_top5"] += b

        # Exact probability of the correct BPE: sum probabilities of all
        # pair-global IDs whose first component is that BPE.
        candidate_ids = bpe_to_global[bt]
        candidate_mask = candidate_ids.ge(0)
        safe_ids = candidate_ids.clamp_min(0)
        candidate_logits = vl.gather(1, safe_ids)
        candidate_logits = candidate_logits.masked_fill(
            ~candidate_mask, float("-inf")
        )
        bpe_log_prob = (
            torch.logsumexp(candidate_logits, dim=1)
            - torch.logsumexp(vl, dim=1)
        )
        total["bpe_loss"] += float(-bpe_log_prob.sum())
        predicted_bpe = global_to_bpe[vl.argmax(-1)]
        total["bpe_top1"] += int(predicted_bpe.eq(bt).sum())
    n = max(total["count"], 1)
    vq_ce = total["vq_loss"] / n
    bpe_ce = total["bpe_loss"] / n
    return dict(
        count=total["count"],
        vq_loss=vq_ce, vq_ppl=math.exp(min(vq_ce, 20.0)),
        vq_top1=total["vq_top1"] / n, vq_top5=total["vq_top5"] / n,
        bpe_loss=bpe_ce, bpe_ppl=math.exp(min(bpe_ce, 20.0)),
        bpe_top1=total["bpe_top1"] / n,
        oracle_bpe_ppl=1.0, oracle_bpe_top1=1.0,
    )


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
    ap.add_argument("--window_layers", type=int, default=2)
    ap.add_argument("--bpe_aux_weight", type=float, default=1.0)
    ap.add_argument("--disable_vqw", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=255)
    ap.add_argument("--vqw_alpha_init", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.local_bpe_tokens != args.gap - 1:
        ap.error("--local_bpe_tokens must equal --gap - 1")
    if not 0.0 < args.vqw_alpha_init < 1.0:
        ap.error("--vqw_alpha_init must be strictly between 0 and 1")

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

    global_to_bpe = None
    global_to_local = None
    bpe_to_global = None
    if args.mode == "local_bpe_direct":
        if codebook.get("partition_type") != "bpe_local_kmeans":
            raise ValueError("local_bpe_direct requires bpe_local_kmeans codebook")
        local_vq_vocab_size = int(data.get(
            "vq_vocab_size", codebook.get("max_local_clusters", -1)
        ))
        if local_vq_vocab_size < 1:
            raise ValueError("invalid local vq_vocab_size")
        if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= local_vq_vocab_size:
            raise ValueError("local VQ ID out of range")
        vq_vocab_size = local_vq_vocab_size
        model = ResidualLocalAR(
            token_vocab_size, vq_vocab_size, args.d_model, args.n_layers,
            args.n_heads, args.dropout, args.max_len,
            use_vqw=not args.disable_vqw,
            alpha_init=args.vqw_alpha_init,
        ).to(device)
        selection_metric = "bpe_ppl"
        architecture = (
            "matched_residual_bpe_baseline_alpha_zero"
            if args.disable_vqw else
            "bpe_main_plus_vqw_only_residual_adapter_learned_alpha"
        )
    else:
        if codebook.get("partition_type") != "bpe_local_kmeans":
            raise ValueError("global_vqwar requires bpe_local_kmeans codebook")
        local_vq_vocab_size = int(data.get(
            "vq_vocab_size", codebook.get("max_local_clusters", -1)
        ))
        if local_vq_vocab_size < 1:
            raise ValueError("invalid local vq_vocab_size")
        if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= local_vq_vocab_size:
            raise ValueError("local VQ ID out of range")
        vq_ids, global_to_bpe, global_to_local = make_pair_global_ids(
            token_ids, vq_ids, local_vq_vocab_size
        )
        vq_vocab_size = int(global_to_bpe.numel())

        # Reverse table used to sum all pair-ID probabilities belonging to BPE.
        counts = torch.bincount(global_to_bpe, minlength=token_vocab_size)
        max_pairs_per_bpe = int(counts.max())
        bpe_to_global = torch.full(
            (token_vocab_size, max_pairs_per_bpe), -1, dtype=torch.long
        )
        cursor = torch.zeros(token_vocab_size, dtype=torch.long)
        for gid, bpe in enumerate(global_to_bpe.tolist()):
            column = int(cursor[bpe])
            bpe_to_global[bpe, column] = gid
            cursor[bpe] += 1

        model = GlobalVQWAR(
            vq_vocab_size, token_vocab_size, args.local_bpe_tokens,
            args.d_model, args.n_layers, args.n_heads, args.dropout,
            args.max_len, args.window_layers
        ).to(device)
        selection_metric = "bpe_ppl"
        architecture = "pair_global_vqwar_11token_window_bpe_aux"

    samples = list(data["samples"])
    random.shuffle(samples)
    n = len(samples); n_train = int(0.8 * n); n_valid = int(0.1 * n)
    splits = (samples[:n_train], samples[n_train:n_train+n_valid], samples[n_train+n_valid:])
    if args.mode == "local_bpe_direct":
        datasets = [FeatureCatLocalDataset(
            split, token_ids, vq_ids, vq_vocab_size,
            args.local_bpe_tokens, args.max_len, train=(i == 0)
        ) for i, split in enumerate(splits)]
    else:
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
    train_bpe_to_global = (
        bpe_to_global.to(device) if bpe_to_global is not None else None
    )
    history, best_valid = [], float("inf")
    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
    print(f"[mode] {args.mode}")
    print(f"[architecture] {architecture}")
    print(f"[alignment] distant=t-{args.gap}; recent_bpe={args.local_bpe_tokens}")
    print(f"[vocab] bpe={token_vocab_size} vqw={vq_vocab_size}")
    if args.mode == "local_bpe_direct":
        print("[input] BPE main path + local-VQW-only residual")
        print("[mask] predicting t: VQW residual[t-10:t-1] zeroed")
        print(f"[ablation] use_vqw={not args.disable_vqw}")
        print(f"[alpha] init={args.vqw_alpha_init:g}; learned sigmoid scalar")
    else:
        print(f"[window] layers={args.window_layers}; no recent-BPE pooling")
    if args.mode == "global_vqwar":
        print(f"[loss] pair_ce + {args.bpe_aux_weight:g} * bpe_marginal_ce")

    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; total_count = 0
        pbar = tqdm(loaders[0], desc=f"[train] epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "local_bpe_direct":
                bpe, local_vq, valid, vqw_available, target = batch
                bpe, local_vq = bpe.to(device), local_vq.to(device)
                valid, vqw_available = valid.to(device), vqw_available.to(device)
                target = target.to(device)
                logits = model(bpe, local_vq, valid, vqw_available)
            else:
                distant_bpe, vq_in, bpe_gap, vq_y, bpe_y, valid = batch
                vq_in, bpe_gap = vq_in.to(device), bpe_gap.to(device)
                valid = valid.to(device)
                target = vq_y.to(device)
                bpe_target = bpe_y.to(device)
                logits = model(vq_in, bpe_gap, ~valid)
            pair_or_bpe_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=-100
            )
            if args.mode == "global_vqwar":
                train_mask = target.ne(-100)
                bpe_aux_loss = grouped_bpe_nll(
                    logits[train_mask], bpe_target[train_mask],
                    train_bpe_to_global
                )
                loss = pair_or_bpe_loss + args.bpe_aux_weight * bpe_aux_loss
            else:
                loss = pair_or_bpe_loss
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
                model, global_to_bpe, bpe_to_global, loaders[1], device
            )
            test_metrics = evaluate_global(
                model, global_to_bpe, bpe_to_global, loaders[2], device
            )
        alpha = (
            float(model.effective_alpha().detach())
            if args.mode == "local_bpe_direct" else None
        )
        alpha_text = f" alpha={alpha:.6f}" if alpha is not None else ""
        print(f"[epoch {epoch}]{alpha_text} "
              f"{format_metrics('valid', valid_metrics)} "
              f"{format_metrics('test', test_metrics)}")
        history.append({"epoch": epoch,
                        "train_loss": total_loss/max(total_count, 1),
                        "alpha": alpha,
                        "valid": valid_metrics, "test": test_metrics})
        checkpoint = {
            "model": model.state_dict(), "args": vars(args), "history": history,
            "mode": args.mode, "architecture": architecture,
            "vq_vocab_size": vq_vocab_size, "token_vocab_size": token_vocab_size,
            "pair_global_id": args.mode == "global_vqwar",
            "pair_global_role": (
                "none_local_vqw_input" if args.mode == "local_bpe_direct"
                else "ar_input_and_target"
            ),
            "global_to_bpe": global_to_bpe,
            "global_to_local": global_to_local,
            "bpe_to_global": bpe_to_global,
            "window_layers": args.window_layers,
            "bpe_aux_weight": args.bpe_aux_weight,
            "learned_vqw_alpha": alpha,
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