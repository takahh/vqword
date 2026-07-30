#!/usr/bin/env python3
"""
VQW autoregressive language model.

Pipeline:
    VQW input -> causal Transformer -> next-VQW logits
              -> predicted VQW ID -> fixed trained decoder -> BPE logits

The VQ tokenizer, cluster centers, and VQ->BPE decoder are frozen.
Only the autoregressive VQ model is trained.
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


class VQARDataset(Dataset):
    def __init__(self, samples, token_ids_flat, vq_ids_flat, max_len=512):
        self.samples = []
        self.token_ids_flat = token_ids_flat
        self.vq_ids_flat = vq_ids_flat

        for i, sample in enumerate(samples):
            start = int(sample["start"])
            end = int(sample["end"])
            length = end - start
            if 4 <= length <= max_len + 1:
                self.samples.append((i, start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_index, start, end = self.samples[index]
        tok = self.token_ids_flat[start:end].long()
        vq = self.vq_ids_flat[start:end].long()

        if tok.numel() != vq.numel():
            raise RuntimeError(
                f"Length mismatch at sample={sample_index}: "
                f"token={tok.numel()} vq={vq.numel()}"
            )

        return {
            "vq_in": vq[:-1],
            "tok_in": tok[:-1],
            "vq_y": vq[1:],
            "tok_y": tok[1:],
        }

def normalize_data_format(data):
    raw_samples = data["samples"]
    if not raw_samples:
        raise ValueError("No samples found")

    first = raw_samples[0]

    if "start" in first and "end" in first:
        samples = raw_samples
        token_ids_flat = data["token_ids_flat"].long().reshape(-1)
        vq_ids_flat = data["vq_ids_flat"].long().reshape(-1)
        return samples, token_ids_flat, vq_ids_flat

    if "token_ids" in first and "vqword_ids" in first:
        samples = []
        token_parts = []
        vq_parts = []
        offset = 0

        for sample_index, sample in enumerate(raw_samples):
            tok = sample["token_ids"].long().reshape(-1)
            vq = sample["vqword_ids"].long().reshape(-1)
            if tok.numel() != vq.numel():
                raise ValueError(
                    f"Length mismatch at sample={sample_index}: "
                    f"token={tok.numel()} vq={vq.numel()}"
                )

            start = offset
            end = start + tok.numel()
            samples.append({
                "sample_idx": int(sample.get("sample_idx", sample_index)),
                "start": start,
                "end": end,
                "length": int(tok.numel()),
            })
            token_parts.append(tok)
            vq_parts.append(vq)
            offset = end

        return samples, torch.cat(token_parts), torch.cat(vq_parts)

    raise ValueError(f"Unsupported sample format: {list(first.keys())}")


def collate(batch, vq_pad_id, tok_pad_id):
    max_len = max(item["vq_in"].numel() for item in batch)
    batch_size = len(batch)

    vq_in = torch.full(
        (batch_size, max_len),
        vq_pad_id,
        dtype=torch.long,
    )

    tok_in = torch.full(
        (batch_size, max_len),
        tok_pad_id,
        dtype=torch.long,
    )

    vq_y = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )

    tok_y = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )

    for i, item in enumerate(batch):
        n = item["vq_in"].numel()

        vq_in[i, :n] = item["vq_in"]
        tok_in[i, :n] = item["tok_in"]
        vq_y[i, :n] = item["vq_y"]
        tok_y[i, :n] = item["tok_y"]

    attention_mask = vq_in.ne(vq_pad_id)

    return (
        vq_in,
        tok_in,
        vq_y,
        tok_y,
        attention_mask,
    )

class VQAutoregressiveLM(nn.Module):
    def __init__(
            self,
            vq_vocab_size,
            token_vocab_size,
            bpe_input_weight=0.01,
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            max_len=512,
    ):
        super().__init__()

        self.vq_vocab_size = int(vq_vocab_size)
        self.token_vocab_size = int(token_vocab_size)

        self.vq_pad_id = self.vq_vocab_size
        self.tok_pad_id = self.token_vocab_size

        self.bpe_input_weight = float(bpe_input_weight)

        self.vq_emb = nn.Embedding(
            self.vq_vocab_size + 1,
            d_model,
            padding_idx=self.vq_pad_id,
        )
        self.tok_emb = nn.Embedding(
            self.token_vocab_size + 1,
            d_model,
            padding_idx=self.tok_pad_id,
        )

        self.fusion = nn.Linear(
            d_model * 2,
            d_model,
        )
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :d_model].copy_(torch.eye(d_model))
            self.fusion.bias.zero_()
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
        self.vq_head = nn.Linear(d_model, self.vq_vocab_size)

        self.fusion = nn.Linear(d_model * 2, d_model)
        self.alpha = 0.01

    def forward(
            self,
            vq_in,
            tok_in,
            key_padding_mask=None,
    ):
        if vq_in.shape != tok_in.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"vq_in={tuple(vq_in.shape)}, "
                f"tok_in={tuple(tok_in.shape)}"
            )

        batch_size, seq_len = vq_in.shape

        pos = torch.arange(
            seq_len,
            device=vq_in.device,
        )[None, :]

        # [B, T, D]
        vq_h = self.vq_emb(vq_in)

        # [B, T, D]
        tok_h = self.tok_emb(tok_in)

        # BPEを小さい係数で補助情報として使う
        tok_h = self.bpe_input_weight * tok_h

        # [B, T, 2D]
        fused = torch.cat(
            [vq_h, tok_h],
            dim=-1,
        )

        # [B, T, D]
        h = self.fusion(fused)

        # 位置埋め込み
        h = h + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=vq_in.device,
            ),
            diagonal=1,
        )

        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.norm(h)

        return self.vq_head(h)


def load_fixed_decoder(dictionary_path, d_model, token_vocab_size, device):
    raw = torch.load(dictionary_path, map_location="cpu", weights_only=False)
    state = raw.get("decoder_state_dict")
    if state is None:
        raise KeyError(
            "dictionary does not contain decoder_state_dict. "
            "Create it with train_vqword_decoder_only.py."
        )

    decoder = nn.Linear(d_model, token_vocab_size)
    decoder.load_state_dict(state, strict=True)
    decoder.to(device)
    decoder.eval()
    decoder.requires_grad_(False)
    return decoder, raw


def load_centers(codebook_path, device):
    raw = torch.load(codebook_path, map_location="cpu", weights_only=False)
    if "global_centers" not in raw:
        raise KeyError("codebook checkpoint does not contain global_centers")
    centers = raw["global_centers"].float().to(device)
    return F.normalize(centers, dim=-1), raw


@torch.no_grad()
def decode_vq_ids(vq_ids, centers, decoder):
    """Decode arbitrary-shaped VQ IDs to BPE logits."""
    flat_ids = vq_ids.reshape(-1)
    unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
    unique_logits = decoder(centers[unique_ids])
    return unique_logits[inverse].reshape(*vq_ids.shape, -1)

@torch.no_grad()
def topk_marginal_bpe_nll(
    vq_logits,
    tok_y,
    centers,
    decoder,
    topk=32,
    max_tokens=2048,
):
    """
    Approximate:
        P(BPE=y | context)
        = sum_v P(v | context) P(y | v)

    using only the top-k VQ candidates.

    Returns:
        summed NLL
        token count
    """
    flat_vq_logits = vq_logits.reshape(-1, vq_logits.size(-1))
    flat_tok_y = tok_y.reshape(-1)

    valid = flat_tok_y.ne(-100)
    flat_vq_logits = flat_vq_logits[valid]
    flat_tok_y = flat_tok_y[valid]

    if flat_tok_y.numel() == 0:
        return 0.0, 0

    if max_tokens > 0 and flat_tok_y.numel() > max_tokens:
        chosen = torch.randperm(
            flat_tok_y.numel(),
            device=flat_tok_y.device,
        )[:max_tokens]

        flat_vq_logits = flat_vq_logits[chosen]
        flat_tok_y = flat_tok_y[chosen]

    k = min(int(topk), flat_vq_logits.size(-1))

    # 全VQ語彙に対する正規化を維持したままtop-kを取る
    all_log_p_vq = F.log_softmax(flat_vq_logits, dim=-1)
    top_log_p_vq, top_ids = all_log_p_vq.topk(k, dim=-1)

    # [N, K, D]
    selected_centers = centers[top_ids]

    # [N, K, token_vocab]
    bpe_logits = decoder(selected_centers)
    log_p_bpe = F.log_softmax(bpe_logits, dim=-1)

    # 各VQ候補について、正解BPEだけ取り出す
    target_index = flat_tok_y[:, None, None].expand(-1, k, 1)

    target_log_p_bpe = log_p_bpe.gather(
        dim=-1,
        index=target_index,
    ).squeeze(-1)

    # log sum_v P(v|context) P(y|v)
    target_log_prob = torch.logsumexp(
        top_log_p_vq + target_log_p_bpe,
        dim=-1,
    )

    nll_sum = -target_log_prob.sum()

    return float(nll_sum.item()), int(flat_tok_y.numel())

def topk_pipeline_bpe_loss(
    vq_logits,
    tok_y,
    centers,
    decoder,
    topk,
    max_tokens,
):
    """
    Optional differentiable approximation:
      top-k P(VQ|context) -> weighted center -> fixed decoder -> BPE CE.

    This is auxiliary only. The default weight is zero.
    """
    flat_vq_logits = vq_logits.reshape(-1, vq_logits.size(-1))
    flat_tok_y = tok_y.reshape(-1)
    valid = flat_tok_y.ne(-100)

    flat_vq_logits = flat_vq_logits[valid]
    flat_tok_y = flat_tok_y[valid]

    if flat_tok_y.numel() == 0:
        return vq_logits.sum() * 0.0

    if max_tokens > 0 and flat_tok_y.numel() > max_tokens:
        chosen = torch.randperm(
            flat_tok_y.numel(),
            device=flat_tok_y.device,
        )[:max_tokens]
        flat_vq_logits = flat_vq_logits[chosen]
        flat_tok_y = flat_tok_y[chosen]

    k = min(int(topk), flat_vq_logits.size(-1))
    values, ids = flat_vq_logits.topk(k, dim=-1)
    weights = F.softmax(values, dim=-1)
    mixed_center = (
        centers[ids] * weights.unsqueeze(-1)
    ).sum(dim=1)

    bpe_logits = decoder(mixed_center)
    return F.cross_entropy(bpe_logits, flat_tok_y)


@torch.no_grad()
def evaluate(
        model,
        loader,
        centers,
        decoder,
        device,
        marginal_topks=(1, 8, 32, 128),
        marginal_max_tokens=0,
):
    model.eval()

    total_vq_loss = 0.0
    total_count = 0
    total_vq_correct = 0

    total_pipe_bpe_loss = 0.0
    total_pipe_top1 = 0
    total_pipe_top5 = 0

    total_marginal_bpe_loss = {
        int(k): 0.0 for k in marginal_topks
    }
    total_marginal_count = {
        int(k): 0 for k in marginal_topks
    }

    total_oracle_bpe_loss = 0.0

    total_oracle_top1 = 0
    total_oracle_top5 = 0

    for (
            vq_in,
            tok_in,
            vq_y,
            tok_y,
            attention_mask,
    ) in tqdm(
        loader,
        desc="[eval]",
        leave=False,
    ):
        vq_in = vq_in.to(device)
        vq_y = vq_y.to(device)
        tok_y = tok_y.to(device)
        tok_in = tok_in.to(device)
        attention_mask = attention_mask.to(device)

        vq_logits = model(
            vq_in,
            tok_in,
            key_padding_mask=~attention_mask,
        )
        vq_loss = F.cross_entropy(
            vq_logits.reshape(-1, vq_logits.size(-1)),
            vq_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid = vq_y.ne(-100)
        n = int(valid.sum().item())
        pred_vq = vq_logits.argmax(dim=-1)

        total_vq_loss += float(vq_loss.item())
        total_count += n
        total_vq_correct += int(pred_vq[valid].eq(vq_y[valid]).sum().item())

        pred_bpe_logits = decode_vq_ids(pred_vq[valid], centers, decoder)
        true_bpe = tok_y[valid]
        pipe_loss = F.cross_entropy(
            pred_bpe_logits,
            true_bpe,
            reduction="sum",
        )

        pipe_topk = pred_bpe_logits.topk(
            min(5, pred_bpe_logits.size(-1)),
            dim=-1,
        ).indices

        total_pipe_bpe_loss += float(pipe_loss.item())
        total_pipe_top1 += int(pipe_topk[:, 0].eq(true_bpe).sum().item())
        total_pipe_top5 += int(
            pipe_topk.eq(true_bpe[:, None]).any(dim=1).sum().item()
        )
        for marginal_k in marginal_topks:
            marginal_loss_sum, marginal_count = topk_marginal_bpe_nll(
                vq_logits=vq_logits,
                tok_y=tok_y,
                centers=centers,
                decoder=decoder,
                topk=marginal_k,
                max_tokens=marginal_max_tokens,
            )

            total_marginal_bpe_loss[int(marginal_k)] += marginal_loss_sum
            total_marginal_count[int(marginal_k)] += marginal_count

        oracle_bpe_logits = decode_vq_ids(vq_y[valid], centers, decoder)
        oracle_loss = F.cross_entropy(
            oracle_bpe_logits,
            true_bpe,
            reduction="sum",
        )
        oracle_topk = oracle_bpe_logits.topk(
            min(5, oracle_bpe_logits.size(-1)),
            dim=-1,
        ).indices

        total_oracle_bpe_loss += float(oracle_loss.item())
        total_oracle_top1 += int(
            oracle_topk[:, 0].eq(true_bpe).sum().item()
        )
        total_oracle_top5 += int(
            oracle_topk.eq(true_bpe[:, None]).any(dim=1).sum().item()
        )

    vq_ce = total_vq_loss / max(total_count, 1)
    pipe_ce = total_pipe_bpe_loss / max(total_count, 1)
    marginal_ce = {
        int(k): (
                total_marginal_bpe_loss[int(k)]
                / max(total_marginal_count[int(k)], 1)
        )
        for k in marginal_topks
    }
    oracle_ce = total_oracle_bpe_loss / max(total_count, 1)

    return {
        "vq_loss": vq_ce,
        "vq_ppl": math.exp(min(vq_ce, 20.0)),
        "vq_acc": total_vq_correct / max(total_count, 1),
        # This is hard argmax-VQ pipeline CE, not a fully marginalized PPL.
        "pipeline_bpe_hard_loss": pipe_ce,
        "pipeline_bpe_hard_ppl": math.exp(min(pipe_ce, 20.0)),
        "pipeline_bpe_top1": total_pipe_top1 / max(total_count, 1),
        "pipeline_bpe_top5": total_pipe_top5 / max(total_count, 1),
        "marginal_bpe_loss": marginal_ce,

        "marginal_bpe_ppl": {
            int(k): math.exp(min(marginal_ce[int(k)], 20.0))
            for k in marginal_topks
        },

        "marginal_bpe_count": total_marginal_count,
        # Upper bound imposed by tokenizer+decoder reconstruction quality.
        "oracle_bpe_loss": oracle_ce,
        "oracle_bpe_ppl": math.exp(min(oracle_ce, 20.0)),
        "oracle_bpe_top1": total_oracle_top1 / max(total_count, 1),
        "oracle_bpe_top5": total_oracle_top5 / max(total_count, 1),
        "count": total_count,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--dictionary", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--out", default="ar_vqw_to_vqw_to_bpe.pt")
    ap.add_argument(
        "--bpe_input_weight",
        type=float,
        default=0.01,
        help="scale applied to BPE embedding before concatenation",
    )
    ap.add_argument(
        "--marginal_topks",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="top-k values used for marginalized BPE evaluation",
    )

    ap.add_argument(
        "--marginal_max_tokens",
        type=int,
        default=0,
        help="maximum evaluated positions per batch; 0 means all valid positions",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--pipeline_bpe_loss_weight",
        type=float,
        default=0.0,
        help="optional auxiliary loss through top-k VQ mixture and fixed decoder",
    )
    ap.add_argument("--pipeline_topk", type=int, default=8)
    ap.add_argument(
        "--pipeline_bpe_max_tokens",
        type=int,
        default=512,
        help="maximum valid positions used by auxiliary BPE loss per batch",
    )

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    samples, token_ids_flat, vq_ids_flat = normalize_data_format(data)

    dictionary_preview = torch.load(
        args.dictionary,
        map_location="cpu",
        weights_only=False,
    )

    decoder_state = dictionary_preview.get("decoder_state_dict")
    if decoder_state is None:
        raise KeyError("dictionary does not contain decoder_state_dict")

    token_vocab_size = int(decoder_state["weight"].shape[0])

    print(f"[token vocab size] {token_vocab_size}")

    codebook_raw = torch.load(
        args.codebook,
        map_location="cpu",
        weights_only=False,
    )
    centers_cpu = codebook_raw["global_centers"].float()
    vq_vocab_size = int(centers_cpu.size(0))
    center_dim = int(centers_cpu.size(1))

    if args.d_model != center_dim:
        print(
            f"[note] AR d_model={args.d_model}, "
            f"decoder center dim={center_dim}; these may differ."
        )

    centers = F.normalize(centers_cpu, dim=-1).to(device)
    decoder, dictionary_raw = load_fixed_decoder(
        args.dictionary,
        d_model=center_dim,
        token_vocab_size=token_vocab_size,
        device=device,
    )

    dictionary_vq_size = int(
        dictionary_raw.get("vq_vocab_size", vq_vocab_size)
    )
    if dictionary_vq_size != vq_vocab_size:
        raise ValueError(
            f"VQ size mismatch: dictionary={dictionary_vq_size}, "
            f"codebook={vq_vocab_size}"
        )

    vq_min = int(vq_ids_flat.min().item())
    vq_max = int(vq_ids_flat.max().item())
    if vq_min < 0 or vq_max >= vq_vocab_size:
        raise ValueError(
            f"VQ IDs out of range: min={vq_min}, max={vq_max}, "
            f"vocab={vq_vocab_size}"
        )

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    train_ds = VQARDataset(
        train_samples,
        token_ids_flat,
        vq_ids_flat,
        max_len=args.max_len,
    )
    valid_ds = VQARDataset(
        valid_samples,
        token_ids_flat,
        vq_ids_flat,
        max_len=args.max_len,
    )
    test_ds = VQARDataset(
        test_samples,
        token_ids_flat,
        vq_ids_flat,
        max_len=args.max_len,
    )

    vq_pad_id = vq_vocab_size
    tok_pad_id = token_vocab_size

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate(
                b,
                vq_pad_id,
                tok_pad_id,
            ),
        )

    train_loader = make_loader(train_ds, True)
    valid_loader = make_loader(valid_ds, False)
    test_loader = make_loader(test_ds, False)

    model = VQAutoregressiveLM(
        vq_vocab_size=vq_vocab_size,
        token_vocab_size=token_vocab_size,
        bpe_input_weight=args.bpe_input_weight,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_valid = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_vq = 0.0
        running_aux = 0.0
        running_n = 0

        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}/{args.epochs}")

        for (
                vq_in,
                tok_in,
                vq_y,
                tok_y,
                attention_mask,
        ) in pbar:
            vq_in = vq_in.to(device)
            tok_in = tok_in.to(device)
            vq_y = vq_y.to(device)
            tok_y = tok_y.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            vq_logits = model(
                vq_in,
                tok_in,
                key_padding_mask=~attention_mask,
            )
            vq_loss = F.cross_entropy(
                vq_logits.reshape(-1, vq_logits.size(-1)),
                vq_y.reshape(-1),
                ignore_index=-100,
            )

            if args.pipeline_bpe_loss_weight > 0:
                aux_loss = topk_pipeline_bpe_loss(
                    vq_logits=vq_logits,
                    tok_y=tok_y,
                    centers=centers,
                    decoder=decoder,
                    topk=args.pipeline_topk,
                    max_tokens=args.pipeline_bpe_max_tokens,
                )
            else:
                aux_loss = vq_loss.new_zeros(())

            loss = vq_loss + args.pipeline_bpe_loss_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_valid_tokens = int(vq_y.ne(-100).sum().item())
            running_vq += float(vq_loss.item()) * n_valid_tokens
            running_aux += float(aux_loss.item()) * n_valid_tokens
            running_n += n_valid_tokens

            pbar.set_postfix(
                vq=f"{running_vq / max(running_n, 1):.4f}",
                aux=f"{running_aux / max(running_n, 1):.4f}",
            )

        valid_metrics = evaluate(
            model,
            valid_loader,
            centers,
            decoder,
            device,
            marginal_topks=args.marginal_topks,
            marginal_max_tokens=args.marginal_max_tokens,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            centers,
            decoder,
            device,
            marginal_topks=args.marginal_topks,
            marginal_max_tokens=args.marginal_max_tokens,
        )

        row = {
            "epoch": epoch,
            "valid": valid_metrics,
            "test": test_metrics,
        }
        history.append(row)

        print(
            f"[epoch {epoch}] "
            f"valid_vq_ppl={valid_metrics['vq_ppl']:.4f} "
            f"valid_vq_acc={valid_metrics['vq_acc']:.4f} "
            f"valid_hard_bpe_ppl={valid_metrics['pipeline_bpe_hard_ppl']:.4f} "
            f"valid_marginal_bpe_ppl={valid_metrics['marginal_bpe_ppl']} "
            f"valid_bpe_top1={valid_metrics['pipeline_bpe_top1']:.4f} "
            f"test_vq_ppl={test_metrics['vq_ppl']:.4f} "
            f"test_hard_bpe_ppl={test_metrics['pipeline_bpe_hard_ppl']:.4f} "
            f"test_marginal_bpe_ppl={test_metrics['marginal_bpe_ppl']:.4f} "
            f"test_bpe_top1={test_metrics['pipeline_bpe_top1']:.4f} "
            f"oracle_bpe_ppl={test_metrics['oracle_bpe_ppl']:.4f} "
            f"oracle_bpe_top1={test_metrics['oracle_bpe_top1']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "vq_vocab_size": vq_vocab_size,
            "vq_pad_id": vq_pad_id,
            "token_vocab_size": token_vocab_size,
            "decoder_frozen": True,
            "decoder_source": str(args.dictionary),
            "codebook_source": str(args.codebook),
            "last_valid": valid_metrics,
            "last_test": test_metrics,
        }
        torch.save(checkpoint, args.out)

        if valid_metrics["vq_loss"] < best_valid:
            best_valid = valid_metrics["vq_loss"]
            best_path = str(Path(args.out).with_suffix("")) + "_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")

    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()
