#!/usr/bin/env python3
"""
Joint scale-0 autoregressive model and scale-0-past -> scale-1 converter.

At target position t:
    input       = sc0[0:t] shifted right, i.e. sc0[t-1] at the current row
    main target = sc0[t]
    aux target  = sc1[t]

Pipeline:
    past sc0 IDs -> frozen sc0 centers -> causal Transformer
                 -> sc0 head: next-sc0 prediction (main AR task)
                 -> sc1 head: current-sc1 prediction (converter task)

The sc1 target is never used as input. The two heads share the same causal
hidden state. A fixed sc0 decoder can optionally be used to report BPE metrics.
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


class Sc0Sc1Dataset(Dataset):
    def __init__(
        self,
        samples,
        token_ids_flat,
        sc0_ids_flat,
        sc1_ids_flat,
        max_len=512,
    ):
        self.samples = []
        self.token_ids_flat = token_ids_flat
        self.sc0_ids_flat = sc0_ids_flat
        self.sc1_ids_flat = sc1_ids_flat
        self.max_len = int(max_len)

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

        return {
            # Causal input available before predicting position t.
            "sc0_in": self.sc0_ids_flat[start:end - 1].long(),
            # Main next-token target.
            "sc0_y": self.sc0_ids_flat[start + 1:end].long(),
            # Converter target for the same predicted position.
            "sc1_y": self.sc1_ids_flat[start + 1:end].long(),
            # Only for fixed-decoder evaluation of predicted sc0.
            "tok_y": self.token_ids_flat[start + 1:end].long(),
        }


def collate_scales(batch, sc0_pad_id, tok_pad_id):
    batch_size = len(batch)
    max_len = max(item["sc0_in"].size(0) for item in batch)

    sc0_in = torch.full(
        (batch_size, max_len), sc0_pad_id, dtype=torch.long
    )
    sc0_y = torch.full(
        (batch_size, max_len), -100, dtype=torch.long
    )
    sc1_y = torch.full(
        (batch_size, max_len), -100, dtype=torch.long
    )
    tok_y = torch.full(
        (batch_size, max_len), -100, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.bool
    )

    for batch_index, item in enumerate(batch):
        n = item["sc0_in"].size(0)
        sc0_in[batch_index, :n] = item["sc0_in"]
        sc0_y[batch_index, :n] = item["sc0_y"]
        sc1_y[batch_index, :n] = item["sc1_y"]
        tok_y[batch_index, :n] = item["tok_y"]
        attention_mask[batch_index, :n] = True

    return sc0_in, sc0_y, sc1_y, tok_y, attention_mask


class FrozenCenterEmbedding(nn.Module):
    def __init__(self, centers):
        super().__init__()
        centers = F.normalize(centers.float(), dim=-1)
        self.padding_idx = int(centers.size(0))
        zero = torch.zeros(1, centers.size(1), dtype=centers.dtype)
        weight = torch.cat([centers, zero], dim=0)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, ids):
        return F.embedding(
            ids,
            self.weight,
            padding_idx=self.padding_idx,
        )


class Sc0ARWithSc1Converter(nn.Module):
    def __init__(
        self,
        sc0_centers,
        sc0_vocab_size,
        sc1_vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.sc0_vocab_size = int(sc0_vocab_size)
        self.sc1_vocab_size = int(sc1_vocab_size)
        self.sc0_pad_id = self.sc0_vocab_size

        self.sc0_embedding = FrozenCenterEmbedding(sc0_centers)
        center_dim = int(sc0_centers.size(1))
        self.input_projection = nn.Linear(center_dim, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)
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

        self.sc0_head = nn.Linear(d_model, sc0_vocab_size)
        self.sc1_head = nn.Linear(d_model, sc1_vocab_size)

    def forward(self, sc0_in, key_padding_mask=None):
        batch_size, seq_len = sc0_in.shape
        del batch_size

        h = self.sc0_embedding(sc0_in)
        h = self.input_norm(self.input_projection(h))

        pos = torch.arange(seq_len, device=h.device)[None, :]
        h = h + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=h.device,
            ),
            diagonal=1,
        )

        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.norm(h)
        return self.sc0_head(h), self.sc1_head(h)


def load_codebook(path, expected_scale=None):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "global_centers" not in raw:
        raise KeyError(f"{path}: checkpoint does not contain global_centers")

    if expected_scale is not None:
        args = raw.get("args", {})
        recorded = args.get("scale", args.get("hop", None))
        if recorded is not None and int(recorded) != int(expected_scale):
            raise ValueError(
                f"Expected scale {expected_scale}, but {path} records {recorded}"
            )

    centers = raw["global_centers"].float()
    return centers, raw


def load_fixed_decoder(dictionary_path, center_dim, token_vocab_size, device):
    raw = torch.load(dictionary_path, map_location="cpu", weights_only=False)
    state = raw.get("decoder_state_dict")
    if state is None:
        raise KeyError("dictionary does not contain decoder_state_dict")

    decoder = nn.Linear(center_dim, token_vocab_size)
    decoder.load_state_dict(state, strict=True)
    decoder.to(device)
    decoder.eval()
    decoder.requires_grad_(False)
    return decoder, raw


@torch.no_grad()
def decode_vq_ids(vq_ids, centers, decoder):
    flat_ids = vq_ids.reshape(-1)
    unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
    unique_logits = decoder(centers[unique_ids])
    return unique_logits[inverse].reshape(*vq_ids.shape, -1)


@torch.no_grad()
def evaluate(model, loader, device, sc0_centers=None, decoder=None):
    model.eval()

    sc0_loss_sum = 0.0
    sc1_loss_sum = 0.0
    count = 0
    sc0_correct = 0
    sc1_correct = 0
    sc1_top5_correct = 0

    bpe_loss_sum = 0.0
    bpe_top1 = 0
    bpe_top5 = 0
    oracle_bpe_loss_sum = 0.0
    oracle_bpe_top1 = 0
    oracle_bpe_top5 = 0

    for sc0_in, sc0_y, sc1_y, tok_y, attention_mask in tqdm(
        loader, desc="[eval]", leave=False
    ):
        sc0_in = sc0_in.to(device)
        sc0_y = sc0_y.to(device)
        sc1_y = sc1_y.to(device)
        tok_y = tok_y.to(device)
        attention_mask = attention_mask.to(device)

        sc0_logits, sc1_logits = model(
            sc0_in=sc0_in,
            key_padding_mask=~attention_mask,
        )

        loss0 = F.cross_entropy(
            sc0_logits.reshape(-1, sc0_logits.size(-1)),
            sc0_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss1 = F.cross_entropy(
            sc1_logits.reshape(-1, sc1_logits.size(-1)),
            sc1_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid = sc0_y.ne(-100)
        n = int(valid.sum().item())
        pred0 = sc0_logits.argmax(dim=-1)
        pred1 = sc1_logits.argmax(dim=-1)

        sc0_loss_sum += float(loss0.item())
        sc1_loss_sum += float(loss1.item())
        count += n
        sc0_correct += int(pred0[valid].eq(sc0_y[valid]).sum().item())
        sc1_correct += int(pred1[valid].eq(sc1_y[valid]).sum().item())

        top5_sc1 = sc1_logits[valid].topk(
            min(5, sc1_logits.size(-1)), dim=-1
        ).indices
        sc1_top5_correct += int(
            top5_sc1.eq(sc1_y[valid, None]).any(dim=1).sum().item()
        )

        if decoder is not None and sc0_centers is not None:
            true_bpe = tok_y[valid]

            pred_bpe_logits = decode_vq_ids(
                pred0[valid], sc0_centers, decoder
            )
            bpe_loss_sum += float(
                F.cross_entropy(
                    pred_bpe_logits, true_bpe, reduction="sum"
                ).item()
            )
            pred_bpe_topk = pred_bpe_logits.topk(
                min(5, pred_bpe_logits.size(-1)), dim=-1
            ).indices
            bpe_top1 += int(pred_bpe_topk[:, 0].eq(true_bpe).sum().item())
            bpe_top5 += int(
                pred_bpe_topk.eq(true_bpe[:, None]).any(dim=1).sum().item()
            )

            oracle_bpe_logits = decode_vq_ids(
                sc0_y[valid], sc0_centers, decoder
            )
            oracle_bpe_loss_sum += float(
                F.cross_entropy(
                    oracle_bpe_logits, true_bpe, reduction="sum"
                ).item()
            )
            oracle_topk = oracle_bpe_logits.topk(
                min(5, oracle_bpe_logits.size(-1)), dim=-1
            ).indices
            oracle_bpe_top1 += int(
                oracle_topk[:, 0].eq(true_bpe).sum().item()
            )
            oracle_bpe_top5 += int(
                oracle_topk.eq(true_bpe[:, None]).any(dim=1).sum().item()
            )

    sc0_ce = sc0_loss_sum / max(count, 1)
    sc1_ce = sc1_loss_sum / max(count, 1)
    metrics = {
        "sc0_loss": sc0_ce,
        "sc0_ppl": math.exp(min(sc0_ce, 20.0)),
        "sc0_acc": sc0_correct / max(count, 1),
        "sc1_converter_loss": sc1_ce,
        "sc1_converter_ppl": math.exp(min(sc1_ce, 20.0)),
        "sc1_converter_acc": sc1_correct / max(count, 1),
        "sc1_converter_top5": sc1_top5_correct / max(count, 1),
        "count": count,
    }

    if decoder is not None and sc0_centers is not None:
        bpe_ce = bpe_loss_sum / max(count, 1)
        oracle_ce = oracle_bpe_loss_sum / max(count, 1)
        metrics.update({
            "pipeline_bpe_hard_loss": bpe_ce,
            "pipeline_bpe_hard_ppl": math.exp(min(bpe_ce, 20.0)),
            "pipeline_bpe_top1": bpe_top1 / max(count, 1),
            "pipeline_bpe_top5": bpe_top5 / max(count, 1),
            "oracle_bpe_loss": oracle_ce,
            "oracle_bpe_ppl": math.exp(min(oracle_ce, 20.0)),
            "oracle_bpe_top1": oracle_bpe_top1 / max(count, 1),
            "oracle_bpe_top5": oracle_bpe_top5 / max(count, 1),
        })

    return metrics


def verify_aligned_data(sc0_data, sc1_data):
    token0 = sc0_data["token_ids_flat"].long().reshape(-1)
    token1 = sc1_data["token_ids_flat"].long().reshape(-1)
    if not torch.equal(token0, token1):
        raise ValueError("sc0/sc1 token_ids_flat do not match")

    samples0 = list(sc0_data["samples"])
    samples1 = list(sc1_data["samples"])
    if len(samples0) != len(samples1):
        raise ValueError("sc0/sc1 sample counts do not match")

    for i, (sample0, sample1) in enumerate(zip(samples0, samples1)):
        for key in ("sample_idx", "start", "end", "length"):
            if int(sample0[key]) != int(sample1[key]):
                raise ValueError(
                    f"sc0/sc1 sample metadata mismatch at sample={i}, key={key}"
                )

    sc0_ids = sc0_data["vq_ids_flat"].long().reshape(-1)
    sc1_ids = sc1_data["vq_ids_flat"].long().reshape(-1)
    if sc0_ids.numel() != token0.numel() or sc1_ids.numel() != token0.numel():
        raise ValueError("VQ/token flattened lengths do not match")

    return samples0, token0, sc0_ids, sc1_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc0_data", required=True)
    ap.add_argument("--sc1_data", required=True)
    ap.add_argument("--sc0_codebook", required=True)
    ap.add_argument("--sc1_codebook", required=True)
    ap.add_argument(
        "--sc0_dictionary",
        default=None,
        help="optional fixed sc0->BPE decoder dictionary for BPE evaluation",
    )
    ap.add_argument(
        "--out", default="ar_sc0_to_sc0_with_sc1_converter.pt"
    )
    ap.add_argument(
        "--converter_loss_weight",
        type=float,
        default=1.0,
        help="weight of sc0-past -> sc1 auxiliary cross entropy",
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
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print("============================================================")
    print("[joint training]")
    print("input       = past sc0")
    print("main target = next sc0")
    print("aux target  = aligned sc1")
    print(f"converter loss weight = {args.converter_loss_weight}")
    print("============================================================")

    sc0_data = torch.load(
        args.sc0_data, map_location="cpu", weights_only=False
    )
    sc1_data = torch.load(
        args.sc1_data, map_location="cpu", weights_only=False
    )
    samples, token_ids_flat, sc0_ids_flat, sc1_ids_flat = verify_aligned_data(
        sc0_data, sc1_data
    )

    sc0_centers_cpu, sc0_codebook_raw = load_codebook(
        args.sc0_codebook, expected_scale=0
    )
    sc1_centers_cpu, sc1_codebook_raw = load_codebook(
        args.sc1_codebook, expected_scale=1
    )
    sc0_vocab_size = int(sc0_centers_cpu.size(0))
    sc1_vocab_size = int(sc1_centers_cpu.size(0))

    if int(sc0_ids_flat.min()) < 0 or int(sc0_ids_flat.max()) >= sc0_vocab_size:
        raise ValueError("sc0 IDs are outside the sc0 codebook range")
    if int(sc1_ids_flat.min()) < 0 or int(sc1_ids_flat.max()) >= sc1_vocab_size:
        raise ValueError("sc1 IDs are outside the sc1 codebook range")

    print(f"[sc0 vocab] {sc0_vocab_size:,}")
    print(f"[sc1 vocab] {sc1_vocab_size:,}")
    print(f"[sc0 used]  {torch.unique(sc0_ids_flat).numel():,}")
    print(f"[sc1 used]  {torch.unique(sc1_ids_flat).numel():,}")

    decoder = None
    token_vocab_size = int(token_ids_flat.max().item()) + 1
    sc0_centers_device = F.normalize(sc0_centers_cpu, dim=-1).to(device)

    if args.sc0_dictionary is not None:
        preview = torch.load(
            args.sc0_dictionary, map_location="cpu", weights_only=False
        )
        state = preview.get("decoder_state_dict")
        if state is None:
            raise KeyError("sc0 dictionary does not contain decoder_state_dict")
        token_vocab_size = int(state["weight"].shape[0])
        decoder, dictionary_raw = load_fixed_decoder(
            args.sc0_dictionary,
            center_dim=int(sc0_centers_cpu.size(1)),
            token_vocab_size=token_vocab_size,
            device=device,
        )
        dictionary_vq_size = int(
            dictionary_raw.get("vq_vocab_size", sc0_vocab_size)
        )
        if dictionary_vq_size != sc0_vocab_size:
            raise ValueError(
                f"sc0 dictionary/codebook VQ mismatch: "
                f"dictionary={dictionary_vq_size}, codebook={sc0_vocab_size}"
            )

        with torch.no_grad():
            ids = sc0_ids_flat[:100000].to(device)
            targets = token_ids_flat[:100000].to(device)
            logits = decoder(sc0_centers_device[ids])
            oracle_loss = F.cross_entropy(logits, targets)
            oracle_top1 = logits.argmax(dim=-1).eq(targets).float().mean()
        print(f"[sc0 decoder sanity] ppl={math.exp(oracle_loss.item()):.4f} "
              f"top1={oracle_top1.item():.4f}")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    def make_dataset(split_samples):
        return Sc0Sc1Dataset(
            samples=split_samples,
            token_ids_flat=token_ids_flat,
            sc0_ids_flat=sc0_ids_flat,
            sc1_ids_flat=sc1_ids_flat,
            max_len=args.max_len,
        )

    sc0_pad_id = sc0_vocab_size
    tok_pad_id = token_vocab_size

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_scales(
                batch, sc0_pad_id, tok_pad_id
            ),
        )

    train_loader = make_loader(make_dataset(train_samples), True)
    valid_loader = make_loader(make_dataset(valid_samples), False)
    test_loader = make_loader(make_dataset(test_samples), False)

    model = Sc0ARWithSc1Converter(
        sc0_centers=sc0_centers_cpu,
        sc0_vocab_size=sc0_vocab_size,
        sc1_vocab_size=sc1_vocab_size,
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
    best_valid_sc0 = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_sc0 = 0.0
        running_sc1 = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"[train] epoch {epoch}/{args.epochs}")

        for sc0_in, sc0_y, sc1_y, tok_y, attention_mask in pbar:
            del tok_y
            sc0_in = sc0_in.to(device)
            sc0_y = sc0_y.to(device)
            sc1_y = sc1_y.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            sc0_logits, sc1_logits = model(
                sc0_in=sc0_in,
                key_padding_mask=~attention_mask,
            )
            sc0_loss = F.cross_entropy(
                sc0_logits.reshape(-1, sc0_logits.size(-1)),
                sc0_y.reshape(-1),
                ignore_index=-100,
            )
            sc1_loss = F.cross_entropy(
                sc1_logits.reshape(-1, sc1_logits.size(-1)),
                sc1_y.reshape(-1),
                ignore_index=-100,
            )
            loss = sc0_loss + args.converter_loss_weight * sc1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_valid_tokens = int(sc0_y.ne(-100).sum().item())
            running_sc0 += float(sc0_loss.item()) * n_valid_tokens
            running_sc1 += float(sc1_loss.item()) * n_valid_tokens
            running_n += n_valid_tokens
            pbar.set_postfix(
                sc0=f"{running_sc0 / max(running_n, 1):.4f}",
                sc1=f"{running_sc1 / max(running_n, 1):.4f}",
            )

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            sc0_centers=sc0_centers_device,
            decoder=decoder,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            sc0_centers=sc0_centers_device,
            decoder=decoder,
        )
        history.append({
            "epoch": epoch,
            "valid": valid_metrics,
            "test": test_metrics,
        })

        message = (
            f"[epoch {epoch}] "
            f"valid_sc0_ppl={valid_metrics['sc0_ppl']:.4f} "
            f"valid_sc0_acc={valid_metrics['sc0_acc']:.4f} "
            f"valid_sc1_conv_ppl={valid_metrics['sc1_converter_ppl']:.4f} "
            f"valid_sc1_conv_acc={valid_metrics['sc1_converter_acc']:.4f} "
            f"test_sc0_ppl={test_metrics['sc0_ppl']:.4f} "
            f"test_sc0_acc={test_metrics['sc0_acc']:.4f} "
            f"test_sc1_conv_ppl={test_metrics['sc1_converter_ppl']:.4f} "
            f"test_sc1_conv_acc={test_metrics['sc1_converter_acc']:.4f} "
            f"test_sc1_conv_top5={test_metrics['sc1_converter_top5']:.4f}"
        )
        if decoder is not None:
            message += (
                f" test_hard_bpe_ppl={test_metrics['pipeline_bpe_hard_ppl']:.4f}"
                f" test_bpe_top1={test_metrics['pipeline_bpe_top1']:.4f}"
                f" oracle_bpe_ppl={test_metrics['oracle_bpe_ppl']:.4f}"
                f" oracle_bpe_top1={test_metrics['oracle_bpe_top1']:.4f}"
            )
        print(message)

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "sc0_vocab_size": sc0_vocab_size,
            "sc1_vocab_size": sc1_vocab_size,
            "sc0_pad_id": sc0_pad_id,
            "token_vocab_size": token_vocab_size,
            "sc0_codebook_source": str(args.sc0_codebook),
            "sc1_codebook_source": str(args.sc1_codebook),
            "sc0_dictionary_source": (
                None if args.sc0_dictionary is None
                else str(args.sc0_dictionary)
            ),
            "last_valid": valid_metrics,
            "last_test": test_metrics,
        }
        torch.save(checkpoint, args.out)

        # Keep the main AR objective as the model-selection criterion.
        if valid_metrics["sc0_loss"] < best_valid_sc0:
            best_valid_sc0 = valid_metrics["sc0_loss"]
            best_path = str(Path(args.out).with_suffix("")) + "_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[save best] {best_path}")

    print(f"[save final] {args.out}")


if __name__ == "__main__":
    main()
