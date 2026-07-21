#!/usr/bin/env python3
import argparse, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VQWordARDataset(Dataset):
    def __init__(
        self,
        samples,
        token_ids_flat,
        vq_ids_flat,
        max_len=1024,
    ):
        self.samples = []

        # 保存ファイルではint32だが、ここでは元Tensorをそのまま保持する
        self.token_ids_flat = token_ids_flat
        self.vq_ids_flat = vq_ids_flat

        for s in samples:
            start = int(s["start"])
            end = int(s["end"])

            length = end - start

            if length > max_len + 1:
                continue

            if length >= 4:
                self.samples.append({
                    "sample_idx": int(s["sample_idx"]),
                    "start": start,
                    "end": end,
                    "length": int(s["length"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]

        start = s["start"]
        end = s["end"]

        # バッチとして使うサンプルだけlongへ変換する
        tok = self.token_ids_flat[start:end].long()
        vq = self.vq_ids_flat[start:end].long()

        if len(tok) != len(vq):
            raise RuntimeError(
                f"Length mismatch: tok={len(tok)}, vq={len(vq)}, "
                f"start={start}, end={end}"
            )

        return {
            "tok_in": tok[:-1],
            "vq_in": vq[:-1],
            "tok_y": tok[1:],
            "vq_y": vq[1:],
        }


@torch.no_grad()
def debug_token_prob_sum(
    model,
    loader,
    device,
    word2vq_prob,
    max_print=20,
):
    model.eval()
    printed = 0

    for tok_in, vq_in, tok_y, vq_y, attn_mask in loader:
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        vq_y = vq_y.to(device)
        attn_mask = attn_mask.to(device)

        h, tok_logits, vq_logits = model(
            tok_in,
            vq_in,
            ~attn_mask,
        )

        log_vq_prob = F.log_softmax(
            vq_logits,
            dim=-1,
        )

        flat_log_vq = log_vq_prob.reshape(
            -1,
            log_vq_prob.size(-1),
        )
        flat_tok = tok_y.reshape(-1)
        flat_vq = vq_y.reshape(-1)

        valid = flat_tok.ne(-100)
        flat_log_vq = flat_log_vq[valid]
        flat_tok = flat_tok[valid]
        flat_vq = flat_vq[valid]

        for i in range(flat_tok.size(0)):
            wid = int(flat_tok[i])
            true_vq = int(flat_vq[i])

            item = word2vq_prob.get(wid, None)

            if item is None:
                continue

            vq_ids, p_word_given_vq = item

            p_true_vq = (
                flat_log_vq[i, true_vq]
                .exp()
                .item()
            )

            log_p_token = torch.logsumexp(
                flat_log_vq[i, vq_ids]
                + torch.log(
                    p_word_given_vq.clamp_min(1e-12)
                ),
                dim=0,
            )

            p_token = log_p_token.exp().item()

            # 正解VQ内部でのP(word|true VQ)
            match = vq_ids.eq(true_vq)

            if match.any():
                true_word_given_vq = (
                    p_word_given_vq[match][0].item()
                )
            else:
                true_word_given_vq = 0.0

            print(
                "wid", wid,
                "n_vq", vq_ids.numel(),
                "p_true_vq", p_true_vq,
                "p_word_given_true_vq",
                true_word_given_vq,
                "p_token", p_token,
            )

            printed += 1

            if printed >= max_print:
                return

def collate(batch, pad_token_id, vq_pad_id):
    maxlen = max(len(x["tok_in"]) for x in batch)

    def pad(x, key, pad_id):
        y = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            v = b[key]
            y[i, :len(v)] = v
        return y

    tok_in = pad(batch, "tok_in", pad_token_id)
    vq_in = pad(batch, "vq_in", vq_pad_id)
    tok_y = pad(batch, "tok_y", -100)
    vq_y = pad(batch, "vq_y", -100)

    attn_mask = tok_in.ne(pad_token_id)
    return tok_in, vq_in, tok_y, vq_y, attn_mask


class ARVQWordLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        vq_vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        max_len=512,
        use_token_input=True,
        use_vq_input=True,
        concat_inputs=False,
        input_vq_weight=1.0,
        init_vq_loss_weight=0.05,
    ):
        super().__init__()

        self.use_token_input = use_token_input
        self.use_vq_input = use_vq_input
        self.concat_inputs = (
            concat_inputs and use_token_input and use_vq_input
        )
        if input_vq_weight < 0:
            raise ValueError(
                "input_vq_weight must be greater than or equal to 0"
            )

        self.input_vq_weight = float(input_vq_weight)
        if init_vq_loss_weight <= 0:
            raise ValueError(
                "init_vq_loss_weight must be greater than 0"
            )

        # softplus(raw) = init_vq_loss_weight になるよう初期化
        raw_init = math.log(math.expm1(init_vq_loss_weight))

        self.raw_vq_loss_weight = nn.Parameter(
            torch.tensor(raw_init, dtype=torch.float32)
        )

        self.vq_emb = (
            nn.Embedding(vq_vocab_size, d_model)
            if use_vq_input else None
        )
        self.tok_emb = (
            nn.Embedding(vocab_size, d_model)
            if use_token_input else None
        )

        # Fine-tuning fusion layer.  Its initialization exactly reproduces
        # the pretrained VQ-only input: W_tok=0 and W_vq=I.
        self.input_fusion = (
            nn.Linear(2 * d_model, d_model)
            if self.concat_inputs else None
        )
        if self.input_fusion is not None:
            with torch.no_grad():
                self.input_fusion.weight.zero_()
                self.input_fusion.bias.zero_()
                self.input_fusion.weight[:, d_model:].copy_(
                    torch.eye(d_model)
                )

        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.tr = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.tok_head = nn.Linear(d_model, vocab_size)
        self.vq_head = nn.Linear(d_model, vq_vocab_size)

    def get_vq_loss_weight(self):
        return F.softplus(self.raw_vq_loss_weight)

    def forward(self, tok_in, vq_in, key_padding_mask=None):
        B, L = tok_in.shape
        pos = torch.arange(
            L,
            device=tok_in.device,
        )[None, :]

        h = self.pos_emb(pos).expand(B, L, -1)

        if self.concat_inputs:
            tok_h = self.tok_emb(tok_in)
            vq_h = (
                self.input_vq_weight
                * self.vq_emb(vq_in)
            )

            fused_h = self.input_fusion(
                torch.cat([tok_h, vq_h], dim=-1)
            )
            h = h + fused_h
        else:
            if self.use_vq_input:
                h = (
                    h
                    + self.input_vq_weight
                    * self.vq_emb(vq_in)
                )

            if self.use_token_input:
                h = h + self.tok_emb(tok_in)

        causal = torch.triu(
            torch.ones(
                L,
                L,
                device=tok_in.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.tr(
            h,
            mask=causal,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.norm(h)

        tok_logits = self.tok_head(h)
        vq_logits = self.vq_head(h)

        return h, tok_logits, vq_logits

def candidate_token_ce_from_hidden(
    model,
    h,
    tok_y,
    vq_logits,
    vq2word_ids,
    topk=32,
):
    B, T, D = h.shape
    pred_vq = vq_logits.argmax(dim=-1)
    flat_h = h.reshape(B * T, D)
    flat_tok_y = tok_y.reshape(B * T)
    flat_vq = pred_vq.reshape(B * T)

    valid = flat_tok_y.ne(-100)
    flat_h = flat_h[valid]
    flat_tok_y = flat_tok_y[valid]
    flat_vq = flat_vq[valid]

    losses = []
    total = 0

    W = model.tok_head.weight
    b = model.tok_head.bias

    for vq_id in flat_vq.unique().tolist():
        base = vq2word_ids.get(int(vq_id), None)
        if base is None or base.numel() == 0:
            continue

        cand_ids = base[:topk]

        idx = flat_vq.eq(vq_id).nonzero(as_tuple=True)[0]
        h_i = flat_h[idx]
        true_tok = flat_tok_y[idx]

        # 辞書候補に正解がある位置だけ使う
        hit = cand_ids[None, :].eq(true_tok[:, None])
        keep = hit.any(dim=1)

        if keep.sum().item() == 0:
            continue

        h_i = h_i[keep]
        true_tok = true_tok[keep]
        hit = hit[keep]

        target = hit.float().argmax(dim=1).long()

        W_c = W[cand_ids]
        b_c = b[cand_ids]
        small_logits = h_i @ W_c.T + b_c

        losses.append(F.cross_entropy(small_logits, target, reduction="sum"))
        total += h_i.size(0)

    if total == 0:
        return torch.tensor(0.0, device=h.device, requires_grad=True)

    return torch.stack(losses).sum() / total

def candidate_token_ce_from_hidden_fast(
    model,
    h,
    tok_y,
    vq_logits,
    cand_table,
    cand_mask,
):
    B, T, D = h.shape

    pred_vq = vq_logits.argmax(dim=-1)

    flat_h = h.reshape(B * T, D)
    flat_tok_y = tok_y.reshape(B * T)
    flat_vq = pred_vq.reshape(B * T)

    valid = flat_tok_y.ne(-100)

    flat_h = flat_h[valid]
    flat_tok_y = flat_tok_y[valid]
    flat_vq = flat_vq[valid]

    cand_ids = cand_table[flat_vq]      # [N, K]
    mask = cand_mask[flat_vq]           # [N, K]

    hit = cand_ids.eq(flat_tok_y[:, None]) & mask
    keep = hit.any(dim=1)

    if keep.sum().item() == 0:
        return torch.tensor(0.0, device=h.device, requires_grad=True)

    flat_h = flat_h[keep]
    cand_ids = cand_ids[keep]
    mask = mask[keep]
    hit = hit[keep]

    target = hit.float().argmax(dim=1).long()

    W = model.tok_head.weight
    b = model.tok_head.bias

    W_c = W[cand_ids]          # [N, K, D]
    b_c = b[cand_ids]          # [N, K]

    logits = (W_c * flat_h[:, None, :]).sum(dim=-1) + b_c
    logits = logits.masked_fill(~mask, -1e9)

    return F.cross_entropy(logits, target, reduction="mean")


def load_vq_dictionary(path):
    raw = torch.load(path, map_location="cpu")

    vq2word_ids = {}
    for vq_id, entries in raw.items():
        vq2word_ids[int(vq_id)] = [int(wid) for wid, word, count in entries]

    return vq2word_ids

@torch.no_grad()
def evaluate_pred_vq_to_word(model, loader, device, vq2word_ids, topk=16):
    model.eval()

    total = 0
    covered = 0
    correct_vq = 0
    correct_word = 0

    W = model.tok_head.weight
    b = model.tok_head.bias

    for tok_in, vq_in, tok_y, vq_y, attn_mask in tqdm(
            loader,
            desc="[pipe]",
            leave=False,
    ):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        vq_y = vq_y.to(device)
        attn_mask = attn_mask.to(device)

        key_padding_mask = ~attn_mask
        h, tok_logits, vq_logits = model(tok_in, vq_in, key_padding_mask)

        pred_vq = vq_logits.argmax(dim=-1)

        flat_h = h.reshape(-1, h.size(-1))
        flat_pred_vq = pred_vq.reshape(-1)
        flat_true_vq = vq_y.reshape(-1)
        flat_true_tok = tok_y.reshape(-1)

        valid = flat_true_tok.ne(-100)

        flat_h = flat_h[valid]
        flat_pred_vq = flat_pred_vq[valid]
        flat_true_vq = flat_true_vq[valid]
        flat_true_tok = flat_true_tok[valid]

        total += flat_true_tok.numel()
        correct_vq += flat_pred_vq.eq(flat_true_vq).sum().item()

        for vq_id in flat_pred_vq.unique().tolist():
            cand = vq2word_ids.get(int(vq_id), None)
            if cand is None or cand.numel() == 0:
                continue

            cand = cand[:topk]

            idx = flat_pred_vq.eq(vq_id).nonzero(as_tuple=True)[0]
            h_i = flat_h[idx]
            true_tok_i = flat_true_tok[idx]

            hit = cand[None, :].eq(true_tok_i[:, None])
            covered += hit.any(dim=1).sum().item()

            W_c = W[cand]
            b_c = b[cand]
            logits = h_i @ W_c.T + b_c

            pred_word = cand[logits.argmax(dim=-1)]
            correct_word += pred_word.eq(true_tok_i).sum().item()

    return {
        "vq_acc": correct_vq / max(total, 1),
        "pred_dict_coverage": covered / max(total, 1),
        "word_acc": correct_word / max(total, 1),
    }

def build_vq_candidate_table(raw_dict, vq_vocab_size, topk, device, pad_value=0):
    cand = torch.full(
        (vq_vocab_size, topk),
        pad_value,
        device=device,
        dtype=torch.long,
    )
    cand_mask = torch.zeros(
        (vq_vocab_size, topk),
        device=device,
        dtype=torch.bool,
    )

    for vq_id, entries in raw_dict.items():
        ids = [int(wid) for wid, word, count in entries[:topk]]
        if len(ids) == 0:
            continue

        ids = torch.tensor(ids, device=device, dtype=torch.long)
        n = min(len(ids), topk)

        cand[int(vq_id), :n] = ids[:n]
        cand_mask[int(vq_id), :n] = True

    return cand, cand_mask


def build_word2vq_prob(raw_dict, token_vocab_size, device, alpha=1e-6):
    word2vq = {}

    for vq_id, entries in raw_dict.items():
        counts = torch.tensor(
            [float(cnt) for wid, word, cnt in entries],
            device=device,
            dtype=torch.float32,
        )
        probs = counts + alpha
        probs = probs / probs.sum()

        for j, (wid, word, cnt) in enumerate(entries):
            wid = int(wid)
            word2vq.setdefault(wid, []).append((int(vq_id), probs[j].item()))

    return {
        wid: (
            torch.tensor([x[0] for x in pairs], device=device, dtype=torch.long),
            torch.tensor([x[1] for x in pairs], device=device, dtype=torch.float32),
        )
        for wid, pairs in word2vq.items()
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    aux_lambda,
    main_target,
    vq2word_ids=None,
    dict_loss=False,
    vq2word_prob=None,
    cand_table=None,
    cand_mask=None,
    word2vq_prob=None,
    vq_to_tok=None,
    check_align=False
):
    model.eval()
    total_tok_loss = 0.0
    total_vq_loss = 0.0
    total_main_loss = 0.0
    total_tok = 0
    total_tok_full_loss = 0.0
    total_dict_loss = 0
    total_dict_tok = 0

    for tok_in, vq_in, tok_y, vq_y, attn_mask in tqdm(
            loader,
            desc="[eval]",
            leave=False,
    ):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        vq_y = vq_y.to(device)
        attn_mask = attn_mask.to(device)
        if word2vq_prob is not None:
            flat_tok = tok_y.reshape(-1)
            flat_vq = vq_y.reshape(-1)
            valid_align = flat_tok.ne(-100) & flat_vq.ne(-100)

            flat_tok = flat_tok[valid_align]
            flat_vq = flat_vq[valid_align]

        key_padding_mask = ~attn_mask
        h, tok_logits, vq_logits = model(tok_in, vq_in, key_padding_mask)

        if word2vq_prob is not None:
            dloss, n = dict_word_ce_fast(
                vq_logits,
                tok_y,
                word2vq_prob,
            )

            total_dict_loss += dloss.item()
            total_dict_tok += n

        if dict_loss and cand_table is not None:
            tok_loss = candidate_token_ce_from_hidden_fast(
                model=model,
                h=h,
                tok_y=tok_y,
                vq_logits=vq_logits,
                cand_table=cand_table,
                cand_mask=cand_mask,
            ) * tok_y.ne(-100).sum()
        else:
            if tok_logits is None:
                tok_logits = model.tok_head(h)

            tok_loss = F.cross_entropy(
                tok_logits.reshape(-1, tok_logits.size(-1)),
                tok_y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        vq_loss = F.cross_entropy(
            vq_logits.reshape(-1, vq_logits.size(-1)),
            vq_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        n = tok_y.ne(-100).sum().item()

        if main_target == "tok":
            main_loss = tok_loss + aux_lambda * vq_loss if aux_lambda > 0 else tok_loss

        elif main_target == "vq":
            main_loss = vq_loss + aux_lambda * tok_loss if aux_lambda > 0 else vq_loss

        elif main_target == "both":
            main_loss = tok_loss + aux_lambda * vq_loss

        total_tok_loss += tok_loss.item()
        total_vq_loss += vq_loss.item()
        total_main_loss += main_loss.item()
        total_tok += n

    tok_ce = total_tok_loss / max(total_tok, 1)
    vq_ce = total_vq_loss / max(total_tok, 1)
    main_ce = total_main_loss / max(total_tok, 1)
    dict_ce = total_dict_loss / max(total_dict_tok, 1)

    return {
        "main_loss": main_ce,
        "main_ppl": math.exp(min(main_ce, 20)),
        "tok_loss": tok_ce,
        "tok_ppl": math.exp(min(tok_ce, 20)),
        "vq_loss": vq_ce,
        "vq_ppl": math.exp(min(vq_ce, 20)),
        "dict_word_loss": dict_ce,
        "dict_word_ppl": math.exp(min(dict_ce, 20)),
    }

@torch.no_grad()
def evaluate_vq_only(model, loader, device):
    model.eval()

    total_vq_loss = 0.0
    total_tok = 0

    for tok_in, vq_in, tok_y, vq_y, attn_mask in tqdm(
            loader,
            desc="[eval-vq]",
            leave=False,
    ):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        vq_y = vq_y.to(device)
        attn_mask = attn_mask.to(device)

        key_padding_mask = ~attn_mask
        h, tok_logits, vq_logits = model(tok_in, vq_in, key_padding_mask)

        vq_loss = F.cross_entropy(
            vq_logits.reshape(-1, vq_logits.size(-1)),
            vq_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        n = vq_y.ne(-100).sum().item()
        total_vq_loss += vq_loss.item()
        total_tok += n

    vq_ce = total_vq_loss / max(total_tok, 1)

    return {
        "main_loss": vq_ce,
        "main_ppl": math.exp(min(vq_ce, 20)),
        "vq_loss": vq_ce,
        "vq_ppl": math.exp(min(vq_ce, 20)),

        # history 保存で落ちないようにダミーも入れる
        "tok_loss": 0.0,
        "tok_ppl": 1.0,
        "tok_full_loss": 0.0,
        "tok_full_ppl": 1.0,
        "dict_word_loss": 0.0,
        "dict_word_ppl": 1.0,
    }

def build_vq_prob_table(raw_dict, vq_vocab_size, topk, device, pad_value=0, alpha=1e-6):
    cand = torch.full((vq_vocab_size, topk), pad_value, device=device, dtype=torch.long)
    prob = torch.zeros((vq_vocab_size, topk), device=device, dtype=torch.float32)
    mask = torch.zeros((vq_vocab_size, topk), device=device, dtype=torch.bool)

    for vq_id, entries in raw_dict.items():
        ids = [int(wid) for wid, word, count in entries[:topk]]
        counts = torch.tensor([float(count) for wid, word, count in entries[:topk]], device=device)

        if len(ids) == 0:
            continue

        p = counts + alpha
        p = p / p.sum()

        n = min(len(ids), topk)
        cand[int(vq_id), :n] = torch.tensor(ids[:n], device=device)
        prob[int(vq_id), :n] = p[:n]
        mask[int(vq_id), :n] = True

    return cand, prob, mask


@torch.no_grad()
def evaluate_argmax_vq_dict_ppl_fast(model, loader, device, cand_table, cand_prob, cand_mask):
    model.eval()
    total_loss = 0.0
    total_tok = 0
    covered = 0

    for tok_in, vq_in, tok_y, vq_y, attn_mask in tqdm(loader, desc="[eval-argmax-dict-fast]", leave=False):
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        attn_mask = attn_mask.to(device)

        h, tok_logits, vq_logits = model(tok_in, vq_in, ~attn_mask)

        pred_vq = vq_logits.argmax(dim=-1).reshape(-1)
        true_tok = tok_y.reshape(-1)

        valid = true_tok.ne(-100)
        pred_vq = pred_vq[valid]
        true_tok = true_tok[valid]

        cand_ids = cand_table[pred_vq]
        probs = cand_prob[pred_vq]
        mask = cand_mask[pred_vq]

        hit = cand_ids.eq(true_tok[:, None]) & mask

        p = torch.zeros(true_tok.size(0), device=device)
        keep = hit.any(dim=1)

        if keep.any():
            pos = hit[keep].float().argmax(dim=1)
            p[keep] = probs[keep, pos]
            covered += keep.sum().item()

        p = p.clamp_min(1e-12)

        total_loss += -torch.log(p).sum().item()
        total_tok += true_tok.numel()

    ce = total_loss / max(total_tok, 1)

    return {
        "argmax_dict_word_loss": ce,
        "argmax_dict_word_ppl": math.exp(min(ce, 20)),
        "argmax_dict_coverage": covered / max(total_tok, 1),
    }

@torch.no_grad()
def dict_word_ce_fast(vq_logits, tok_y, word2vq_prob):
    """
    P(word | context)
      = sum_vq P(vq | context) * P(word | vq)

    を使ってword-level cross entropyを計算する。
    """
    log_vq_prob = F.log_softmax(vq_logits, dim=-1)

    flat_log_vq = log_vq_prob.reshape(
        -1,
        log_vq_prob.size(-1),
    )
    flat_tok_y = tok_y.reshape(-1)

    valid = flat_tok_y.ne(-100)
    flat_log_vq = flat_log_vq[valid]
    flat_tok_y = flat_tok_y[valid]

    total_loss = 0.0
    total_tok = int(flat_tok_y.numel())

    log_floor = math.log(1e-12)

    for wid in flat_tok_y.unique().tolist():
        item = word2vq_prob.get(int(wid), None)
        idx = flat_tok_y.eq(wid)
        n = int(idx.sum().item())

        if item is None:
            total_loss += -log_floor * n
            continue

        vq_ids, p_word_given_vq = item

        # log P(vq | context)
        selected_log_vq = flat_log_vq[idx][:, vq_ids]

        # log P(word | vq)
        log_p_word_given_vq = torch.log(
            p_word_given_vq.clamp_min(1e-12)
        )

        # log sum_v P(v|context) P(word|v)
        logp_word = torch.logsumexp(
            selected_log_vq
            + log_p_word_given_vq.unsqueeze(0),
            dim=1,
        )

        total_loss += (-logp_word).sum().item()

    return (
        torch.tensor(
            total_loss,
            device=vq_logits.device,
        ),
        total_tok,
    )

def build_word2vq_weighted(raw_dict, device):
    """
    辞書カウントから P(BPE | VQ) を作り、
    BPEごとに対応するVQ IDと確率をまとめる。

    return:
        word2vq[wid] = (
            vq_ids,       # [N]
            word_probs,   # [N], 各VQにおける P(wid | vq)
        )
    """
    word2vq = {}

    n_used_vq = 0
    n_empty_vq = 0
    n_relations = 0

    for vq_id, entries in raw_dict.items():
        if entries is None or len(entries) == 0:
            n_empty_vq += 1
            continue

        total = sum(float(cnt) for wid, word, cnt in entries)

        if total <= 0:
            n_empty_vq += 1
            continue

        for wid, word, cnt in entries:
            wid = int(wid)
            probability = float(cnt) / total

            word2vq.setdefault(wid, []).append(
                (int(vq_id), probability)
            )
            n_relations += 1

        n_used_vq += 1

    result = {}

    for wid, pairs in word2vq.items():
        result[wid] = (
            torch.tensor(
                [vq_id for vq_id, probability in pairs],
                device=device,
                dtype=torch.long,
            ),
            torch.tensor(
                [probability for vq_id, probability in pairs],
                device=device,
                dtype=torch.float32,
            ),
        )

    print(
        f"[build_word2vq_weighted] "
        f"used_vq={n_used_vq:,} "
        f"empty_vq={n_empty_vq:,} "
        f"token_entries={len(result):,} "
        f"relations={n_relations:,}"
    )

    return result

def build_word2vq_unique(raw_dict, device):
    """
    各VQ IDを、その辞書内で最頻のBPE IDへ対応付ける。

    未使用VQ IDは entries=[] なのでスキップする。
    """
    word2vq = {}

    n_empty = 0
    n_used = 0

    for vq_id, entries in raw_dict.items():
        # 未使用クラスタは辞書候補が空
        if entries is None or len(entries) == 0:
            n_empty += 1
            continue

        # entriesは出現回数順に並んでいるので、
        # 先頭がそのVQに対応する最多BPE
        wid, word, cnt = entries[0]

        wid = int(wid)
        vq_id = int(vq_id)

        word2vq.setdefault(wid, []).append(vq_id)
        n_used += 1

    print(
        f"[build_word2vq_unique] "
        f"used_vq={n_used:,} "
        f"empty_vq={n_empty:,} "
        f"token_entries={len(word2vq):,}"
    )

    return {
        wid: torch.tensor(
            vqs,
            device=device,
            dtype=torch.long,
        )
        for wid, vqs in word2vq.items()
    }

def build_vq_to_tok(raw_dict, vq_vocab_size, device):
    """
    各VQ IDを、そのVQで最多のBPE IDへ対応付ける。

    未使用VQは辞書エントリが空なので、
    対応値を -1 のまま残す。
    """
    vq_to_tok = torch.full(
        (vq_vocab_size,),
        -1,
        device=device,
        dtype=torch.long,
    )

    n_used = 0
    n_empty = 0

    for vq_id, entries in raw_dict.items():
        # 未使用VQは空リスト
        if entries is None or len(entries) == 0:
            n_empty += 1
            continue

        wid, word, cnt = entries[0]

        vq_id = int(vq_id)
        wid = int(wid)

        if vq_id < 0 or vq_id >= vq_vocab_size:
            raise ValueError(
                f"VQ ID out of range: "
                f"vq_id={vq_id}, "
                f"vq_vocab_size={vq_vocab_size}"
            )

        vq_to_tok[vq_id] = wid
        n_used += 1

    print(
        f"[build_vq_to_tok] "
        f"used_vq={n_used:,} "
        f"empty_vq={n_empty:,}"
    )

    return vq_to_tok

def masked_token_ce_by_true_vq(tok_logits, tok_y, vq_y, vq2word_ids):
    B, T, V = tok_logits.shape
    flat_logits = tok_logits.reshape(B * T, V)
    flat_tok_y = tok_y.reshape(B * T)
    flat_vq_y = vq_y.reshape(B * T)

    valid = flat_tok_y.ne(-100)
    flat_logits = flat_logits[valid]
    flat_tok_y = flat_tok_y[valid]
    flat_vq_y = flat_vq_y[valid]

    losses = []

    # 同じ vq_id ごとにまとめる
    for vq_id in flat_vq_y.unique().tolist():
        mask = flat_vq_y.eq(vq_id)
        idx = mask.nonzero(as_tuple=True)[0]

        true_tok = flat_tok_y[idx]  # [N]

        cand_ids = list(vq2word_ids.get(int(vq_id), []))[:32]

        # このbatch内の正解tokenをまとめて候補に追加
        cand_ids = sorted(set(cand_ids) | set(true_tok.detach().cpu().tolist()))

        cand = torch.tensor(
            cand_ids,
            device=tok_logits.device,
            dtype=torch.long,
        )

        small_logits = flat_logits[idx][:, cand]  # [N, C]

        # true_tok -> candidate index
        pos = {int(w): j for j, w in enumerate(cand_ids)}
        target = torch.tensor(
            [pos[int(t)] for t in true_tok.detach().cpu().tolist()],
            device=tok_logits.device,
            dtype=torch.long,
        )

        losses.append(
            F.cross_entropy(small_logits, target, reduction="sum")
        )

    return torch.stack(losses).sum() / flat_tok_y.numel()

def load_dictionary_entries(path):
    """
    旧形式とglobal IVF形式の両方の辞書を読み込み、
    以下の共通形式へ変換する。

        dict_entries[vq_id] = [
            (bpe_id, token_text, count),
            ...
        ]

    global IVF辞書にはtoken_textが入っていないため、
    token_textには空文字列を入れる。
    """
    raw = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # ------------------------------------------
    # 新しいglobal IVF辞書
    # ------------------------------------------
    if "vq_to_bpe_ids" in raw:
        source_entries = raw["vq_to_bpe_ids"]

        print("[dictionary format] global vq_to_bpe_ids")

    # ------------------------------------------
    # 以前の辞書形式
    # 数字キーがトップレベルにある
    # ------------------------------------------
    else:
        source_entries = {
            int(k): v
            for k, v in raw.items()
            if isinstance(k, int)
            or (isinstance(k, str) and k.isdigit())
        }

        print("[dictionary format] legacy numeric keys")

    dict_entries = {}

    for raw_vq_id, entries in source_entries.items():
        vq_id = int(raw_vq_id)
        normalized = []

        for entry in entries:
            # 新global形式:
            # (bpe_id, count)
            if len(entry) == 2:
                wid, count = entry
                word = ""

            # 旧形式:
            # (bpe_id, token文字列, count)
            elif len(entry) == 3:
                wid, word, count = entry

            else:
                raise ValueError(
                    f"Unexpected dictionary entry: "
                    f"vq_id={vq_id}, entry={entry}"
                )

            normalized.append(
                (
                    int(wid),
                    str(word),
                    int(count),
                )
            )

        dict_entries[vq_id] = normalized

    return raw, dict_entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to VQWord AR training data .pt")
    ap.add_argument("--tokenizer", default=None, help="local tokenizer dir or HF tokenizer name")
    ap.add_argument("--token_vocab_size", type=int, default=None)
    ap.add_argument("--vq_vocab_size", type=int, default=None)
    ap.add_argument("--out", default="ar_vqword_lm.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--main_target", choices=["tok", "vq", "both"], default="tok")
    ap.add_argument("--freeze_vq_backbone", action="store_true")
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux_lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vq_only", action="store_true")
    ap.add_argument("--token_only", action="store_true")
    ap.add_argument("--init_from", default=None, help="pretrained AR checkpoint .pt")
    ap.add_argument(
        "--init_source",
        choices=["vq", "bpe"],
        default="vq",
        help=(
            "checkpoint type used by --init_from: "
            "'vq' for VQ-pretrained model, "
            "'bpe' for token-only BPE baseline"
        ),
    )
    ap.add_argument("--reset_heads", action="store_true")
    ap.add_argument("--dictionary", default=None)
    ap.add_argument("--dict_loss", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    ap.add_argument(
        "--learn_vq_loss_weight",
        action="store_true",
        help="learn the positive weight applied to vq_loss",
    )

    ap.add_argument(
        "--init_vq_loss_weight",
        type=float,
        default=0.05,
        help="initial value of the learnable VQ loss weight",
    )
    args = ap.parse_args()

    history = {
        "epoch": [],
        "valid_loss": [],
        "valid_ppl": [],
        "valid_tok_ppl": [],
        "valid_vq_ppl": [],
        "test_loss": [],
        "test_ppl": [],
        "test_tok_ppl": [],
        "test_vq_ppl": [],
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.vq_only and args.token_only:
        raise ValueError("Choose at most one of --vq_only or --token_only")

    use_token_input = not args.vq_only
    use_vq_input = not args.token_only

    data = torch.load(args.data, map_location="cpu")

    raw_samples = data["samples"]

    if len(raw_samples) == 0:
        raise ValueError("No samples found in data file")

    first_sample = raw_samples[0]

    # ============================================================
    # データ形式を統一する
    #
    # 新形式:
    #   samples = [{"start": ..., "end": ...}, ...]
    #   token_ids_flat / vq_ids_flat を参照
    #
    # 旧形式:
    #   samples = [{
    #       "token_ids": Tensor,
    #       "vqword_ids": Tensor,
    #       "length": ...
    #   }, ...]
    #
    # どちらも内部では新形式へ統一する。
    # ============================================================

    if "start" in first_sample and "end" in first_sample:
        print("[data format] flat start/end format")

        samples = raw_samples
        token_ids_flat = data["token_ids_flat"].long().reshape(-1)
        vq_ids_flat = data["vq_ids_flat"].long().reshape(-1)

    elif "token_ids" in first_sample and "vqword_ids" in first_sample:
        print("[data format] embedded sample format -> converting in memory")

        samples = []
        token_parts = []
        vq_parts = []
        offset = 0

        for sample_idx, sample in enumerate(raw_samples):
            token_ids = sample["token_ids"].long().reshape(-1)
            vq_ids = sample["vqword_ids"].long().reshape(-1)

            if token_ids.numel() != vq_ids.numel():
                raise ValueError(
                    f"Sample length mismatch at sample {sample_idx}: "
                    f"token_ids={token_ids.numel()}, "
                    f"vqword_ids={vq_ids.numel()}"
                )

            length = token_ids.numel()
            start = offset
            end = start + length

            samples.append({
                "sample_idx": int(
                    sample.get("sample_idx", sample_idx)
                ),
                "start": start,
                "end": end,
                "length": length,
            })

            token_parts.append(token_ids)
            vq_parts.append(vq_ids)
            offset = end

        token_ids_flat = torch.cat(token_parts)
        vq_ids_flat = torch.cat(vq_parts)

        print(
            f"[converted] samples={len(samples):,} "
            f"tokens={token_ids_flat.numel():,}"
        )

    else:
        raise ValueError(
            "Unsupported sample format. "
            f"First sample keys: {list(first_sample.keys())}"
        )

    if len(token_ids_flat) != len(vq_ids_flat):
        raise ValueError(
            f"Flat length mismatch: "
            f"token_ids={len(token_ids_flat):,}, "
            f"vq_ids={len(vq_ids_flat):,}"
        )

    for sample_idx, sample in enumerate(samples):
        start = int(sample["start"])
        end = int(sample["end"])

        if not (0 <= start <= end <= len(token_ids_flat)):
            raise ValueError(
                f"Invalid sample range at sample {sample_idx}: "
                f"start={start}, end={end}, "
                f"flat_len={len(token_ids_flat)}"
            )

    print("[token_ids_flat]", token_ids_flat.dtype, token_ids_flat.shape)
    print("[vq_ids_flat]", vq_ids_flat.dtype, vq_ids_flat.shape)

    for s in samples[:5]:
        start = int(s["start"])
        end = int(s["end"])

        tok = token_ids_flat[start:end]
        vq = vq_ids_flat[start:end]

        print("len", len(tok), len(vq))
        print("tok[:20]", tok[:20].tolist())
        print("vq [:20]", vq[:20].tolist())
        print()

    dict_vq_vocab_size = None

    if args.dictionary is not None:
        raw_dict, dict_entries = load_dictionary_entries(
            args.dictionary
        )

        print(
            f"[dictionary] loaded entries="
            f"{len(dict_entries):,}"
        )

        # キー0が未使用でも落ちないように、
        # 最初の非空エントリを探して表示する
        sample_found = False

        for sample_vq_id in sorted(dict_entries.keys()):
            entries = dict_entries[sample_vq_id]

            if len(entries) == 0:
                continue

            print(
                f"[dictionary sample] "
                f"vq_id={sample_vq_id}"
            )
            print(entries[:10])

            sample_found = True
            break

        if not sample_found:
            raise RuntimeError(
                "Dictionary contains no non-empty VQ entries"
            )

    else:
        raw_dict = None
        dict_entries = {}

    if len(dict_entries) > 0:
        dict_vq_vocab_size = int(
            raw_dict.get("vq_vocab_size", max(dict_entries.keys()) + 1)
        )
        print(f"[dict_vq_vocab_size] {dict_vq_vocab_size}")

        DICT_TOPK = 32
        vq2word_ids = {
            int(vq_id): torch.tensor(
                [int(wid) for wid, word, count in entries[:DICT_TOPK]],
                device=device,
                dtype=torch.long,
            )
            for vq_id, entries in dict_entries.items()
        }

        print(f"[dictionary] loaded {len(vq2word_ids)} VQ entries")

    else:
        raw_dict = None
        vq2word_ids = None

    tokenizer_name = args.tokenizer or data.get("tokenizer", None)

    print(f"[data] {args.data}")
    print(f"[tokenizer metadata] {tokenizer_name}")

    # このAR学習では、データファイル内に保存済みの
    # token_ids_flatを直接BPE IDとして使用する。
    #
    # そのため、Tokenizer本体をロードする必要はない。
    # tokenizer_nameはcheckpointへ保存するメタデータとしてのみ保持する。

    if args.token_vocab_size is not None:
        token_vocab_size = int(args.token_vocab_size)

    elif "token_vocab_size" in data:
        token_vocab_size = int(data["token_vocab_size"])

    elif "token_ids_flat" in data:
        token_vocab_size = int(data["token_ids_flat"].max().item()) + 1

    elif "word2id" in data and data["word2id"] is not None:
        token_vocab_size = len(data["word2id"])

    else:
        raise ValueError(
            "Cannot infer token_vocab_size. "
            "Specify --token_vocab_size."
        )

    pad_token_id = int(data.get("pad_token_id", 0))

    print(f"[pad_token_id] {pad_token_id}")
    word2vq_prob = None

    if raw_dict is not None:
        word2vq_prob = build_word2vq_weighted(
            dict_entries,
            device,
        )

        print(
            f"[word2vq_prob] loaded "
            f"{len(word2vq_prob):,} token entries"
        )
    if args.vq_vocab_size is not None:
        base_vq_vocab_size = args.vq_vocab_size
    elif "vq_vocab_size" in data:
        base_vq_vocab_size = int(data["vq_vocab_size"])
    else:
        base_vq_vocab_size = int(data["vq_ids_flat"].max().item()) + 1

    if dict_vq_vocab_size is not None:
        base_vq_vocab_size = max(base_vq_vocab_size, dict_vq_vocab_size)

    vq_pad_id = base_vq_vocab_size
    vq_vocab_size_incl_pad = vq_pad_id + 1

    print(f"[token_vocab_size] {token_vocab_size}")
    print(f"[base_vq_vocab_size] {base_vq_vocab_size}")
    print(f"[vq_vocab_size incl pad] {vq_vocab_size_incl_pad}")
    print(f"[vq_pad_id] {vq_pad_id}")
    print("[CHECK] data keys:", data.keys())
    vq_max = int(vq_ids_flat.max().item())
    vq_min = int(vq_ids_flat.min().item())

    print("[CHECK] vq_ids min:", vq_min)
    print("[CHECK] vq_ids max:", vq_max)

    print("[vq_pad_id]", vq_pad_id)

    cand_table = None
    cand_mask = None

    if raw_dict is not None:
        cand_table, cand_mask = build_vq_candidate_table(
            raw_dict=dict_entries,
            vq_vocab_size=base_vq_vocab_size,
            topk=16,
            device=device,
            pad_value=pad_token_id,
        )
        print(f"[candidate table] {cand_table.shape}")

    vq_to_tok = None
    if raw_dict is not None:
        vq_to_tok = build_vq_to_tok(
            dict_entries,
            vq_vocab_size_incl_pad,
            device,
        )

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    train_s = samples[:n_train]
    valid_s = samples[n_train:n_train + n_valid]
    test_s = samples[n_train + n_valid:]

    train_ds = VQWordARDataset(
        train_s,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        max_len=512,
    )

    valid_ds = VQWordARDataset(
        valid_s,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        max_len=512,
    )

    test_ds = VQWordARDataset(
        test_s,
        token_ids_flat=token_ids_flat,
        vq_ids_flat=vq_ids_flat,
        max_len=512,
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate(b, pad_token_id, vq_pad_id),
        )

    train_loader = make_loader(train_ds, True)
    valid_loader = make_loader(valid_ds, False)
    test_loader = make_loader(test_ds, False)

    model = ARVQWordLM(
        vocab_size=token_vocab_size,
        vq_vocab_size=vq_vocab_size_incl_pad,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=512,
        use_token_input=use_token_input,
        use_vq_input=use_vq_input,
        concat_inputs=(args.mode == "finetune"),
        init_vq_loss_weight=args.init_vq_loss_weight,
    ).to(device)
    if not args.learn_vq_loss_weight:
        model.raw_vq_loss_weight.requires_grad = False

    if args.init_from is not None:
        ckpt = torch.load(
            args.init_from,
            map_location="cpu",
            weights_only=False,
        )

        if "model" not in ckpt:
            raise KeyError(
                f"Checkpoint does not contain 'model': "
                f"{args.init_from}"
            )

        # 元checkpointの辞書を壊さないようコピーする
        sd = dict(ckpt["model"])

        print(f"[init_from] {args.init_from}")
        print(f"[init_source] {args.init_source}")

        # ====================================================
        # VQ pretrained checkpointから初期化
        # ====================================================
        if args.init_source == "vq":
            # VQ pretrain時のtok_headは使わない
            sd.pop("tok_head.weight", None)
            sd.pop("tok_head.bias", None)

            if "vq_emb.weight" not in sd:
                raise KeyError(
                    "VQ checkpoint does not contain "
                    "'vq_emb.weight'"
                )

            old_vq_size = sd["vq_emb.weight"].shape[0]
            new_vq_size = model.vq_emb.weight.shape[0]

            if old_vq_size != new_vq_size:
                print(
                    f"[resize ckpt vq] "
                    f"{old_vq_size} -> {new_vq_size}"
                )

                copy_n = min(old_vq_size, new_vq_size)

                # ----------------------------
                # VQ embedding
                # ----------------------------
                old_w = sd["vq_emb.weight"]
                new_w = (
                    model.vq_emb.weight
                    .detach()
                    .cpu()
                    .clone()
                )
                new_w[:copy_n] = old_w[:copy_n]
                sd["vq_emb.weight"] = new_w

                # ----------------------------
                # VQ output head
                # ----------------------------
                if "vq_head.weight" in sd:
                    old_w = sd["vq_head.weight"]
                    new_w = (
                        model.vq_head.weight
                        .detach()
                        .cpu()
                        .clone()
                    )
                    new_w[:copy_n] = old_w[:copy_n]
                    sd["vq_head.weight"] = new_w

                if "vq_head.bias" in sd:
                    old_b = sd["vq_head.bias"]
                    new_b = (
                        model.vq_head.bias
                        .detach()
                        .cpu()
                        .clone()
                    )
                    new_b[:copy_n] = old_b[:copy_n]
                    sd["vq_head.bias"] = new_b

        # ====================================================
        # token-only BPE baselineから初期化
        # ====================================================
        elif args.init_source == "bpe":
            print("[bpe init] loading token-only BPE baseline")

            # BPE baselineから引き継ぐもの
            #
            #   tok_emb
            #   pos_emb
            #   Transformer
            #   norm
            #   tok_head
            #
            # VQ関連とfusionは今回のモデル用に新規初期化する。

            sd.pop("vq_emb.weight", None)
            sd.pop("vq_head.weight", None)
            sd.pop("vq_head.bias", None)

            sd.pop("input_fusion.weight", None)
            sd.pop("input_fusion.bias", None)

            # 今回指定したVQ loss weightを使う
            sd.pop("raw_vq_loss_weight", None)

            current_sd = model.state_dict()

            required_bpe_keys = [
                "tok_emb.weight",
                "pos_emb.weight",
                "tok_head.weight",
                "tok_head.bias",
            ]

            for key in required_bpe_keys:
                if key not in sd:
                    raise KeyError(
                        f"BPE checkpoint is missing required key: "
                        f"{key}"
                    )

                if sd[key].shape != current_sd[key].shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"checkpoint={tuple(sd[key].shape)}, "
                        f"current={tuple(current_sd[key].shape)}"
                    )

            # Transformer層の形も事前確認
            for key, value in sd.items():
                if key not in current_sd:
                    continue

                if value.shape != current_sd[key].shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"checkpoint={tuple(value.shape)}, "
                        f"current={tuple(current_sd[key].shape)}"
                    )

        # ====================================================
        # 共通：checkpointをロード
        # ====================================================
        missing, unexpected = model.load_state_dict(
            sd,
            strict=False,
        )

        print(f"[init] missing={missing}")
        print(f"[init] unexpected={unexpected}")

        # ====================================================
        # BPE初期化時：
        # fusion([BPE, VQW]) = BPE
        #
        # concat順：
        #   [tok_h | vq_h]
        #
        # fusion重み：
        #   [I | 0]
        # ====================================================
        if args.init_source == "bpe":
            if model.input_fusion is None:
                raise RuntimeError(
                    "BPE initialization requires input_fusion. "
                    "Run with --mode finetune."
                )

            d_model = model.tok_emb.embedding_dim

            with torch.no_grad():
                model.input_fusion.weight.zero_()
                model.input_fusion.bias.zero_()

                identity = torch.eye(
                    d_model,
                    device=model.input_fusion.weight.device,
                    dtype=model.input_fusion.weight.dtype,
                )

                # 左半分＝BPE側をIdentityにする
                model.input_fusion.weight[
                    :,
                    :d_model
                ].copy_(identity)

            print(
                "[fusion init] "
                "BPE path = identity, VQW path = zero"
            )

        # reset_headsを明示した場合だけheadを初期化し直す
        if args.reset_heads:
            print("[init] reset tok_head / vq_head")
            model.tok_head.reset_parameters()
            model.vq_head.reset_parameters()

        # ====================================================
        # freeze処理
        # ====================================================
        if args.freeze_vq_backbone:
            for p in model.parameters():
                p.requires_grad = False

            if args.init_source == "vq":
                # 従来：
                # VQ経路を固定し、新しいBPE側を学習
                if model.tok_emb is not None:
                    for p in model.tok_emb.parameters():
                        p.requires_grad = True

                if model.input_fusion is not None:
                    for p in model.input_fusion.parameters():
                        p.requires_grad = True

                for p in model.tok_head.parameters():
                    p.requires_grad = True

                print(
                    "[freeze] "
                    "train tok_emb + input_fusion + tok_head"
                )

            elif args.init_source == "bpe":
                # BPE baselineを固定し、
                # VQ embeddingとfusionだけ学習する場合
                if model.vq_emb is not None:
                    for p in model.vq_emb.parameters():
                        p.requires_grad = True

                if model.input_fusion is not None:
                    for p in model.input_fusion.parameters():
                        p.requires_grad = True

                print(
                    "[freeze] "
                    "train vq_emb + input_fusion"
                )

    # ===========================
    # 評価のみ
    # ===========================
    if args.learn_vq_loss_weight:
        eval_aux_lambda = float(
            model.get_vq_loss_weight().detach().cpu()
        )
    else:
        eval_aux_lambda = args.aux_lambda
    if args.eval_only:
        valid = evaluate(
            model,
            valid_loader,
            device,
            eval_aux_lambda,
            args.main_target,
            vq2word_ids=vq2word_ids,
            dict_loss=False,
            cand_table=cand_table,
            cand_mask=cand_mask,
            word2vq_prob=word2vq_prob,
            vq_to_tok=vq_to_tok,
            check_align=True
        )

        test = evaluate(
            model,
            test_loader,
            device,
            eval_aux_lambda,
            args.main_target,
            vq2word_ids=vq2word_ids,
            dict_loss=False,
            cand_table=cand_table,
            cand_mask=cand_mask,
            word2vq_prob=word2vq_prob,
            vq_to_tok=vq_to_tok,
        )
        debug_token_prob_sum(
            model,
            valid_loader,
            device,
            word2vq_prob,
            max_print=30,
        )
        print(
            f"[eval-only] "
            f"valid_vq_ppl={valid['vq_ppl']:.2f} "
            f"valid_dict_word_ppl={valid['dict_word_ppl']:.2f} "
            f"test_vq_ppl={test['vq_ppl']:.2f} "
            f"test_dict_word_ppl={test['dict_word_ppl']:.2f}"
        )

        return

    regular_params = []
    weight_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name == "raw_vq_loss_weight":
            weight_params.append(parameter)
        else:
            regular_params.append(parameter)

    optimizer_groups = [
        {
            "params": regular_params,
            "weight_decay": 0.01,
        }
    ]

    if weight_params:
        optimizer_groups.append({
            "params": weight_params,
            "weight_decay": 0.0,
        })

    opt = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
    )
    best_valid = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}")

        for tok_in, vq_in, tok_y, vq_y, attn_mask in pbar:
            tok_in = tok_in.to(device)
            vq_in = vq_in.to(device)
            tok_y = tok_y.to(device)
            vq_y = vq_y.to(device)
            attn_mask = attn_mask.to(device)

            key_padding_mask = ~attn_mask
            if model.vq_emb is not None:
                if vq_in.min() < 0 or vq_in.max() >= model.vq_emb.num_embeddings:
                    print("[BAD vq_in]")
                    print("vq_in min:", int(vq_in.min()))
                    print("vq_in max:", int(vq_in.max()))
                    print(
                        "model.vq_emb.num_embeddings:",
                        model.vq_emb.num_embeddings,
                    )
                    raise RuntimeError("vq_in out of range")

            valid_vq_y = vq_y[vq_y.ne(-100)]

            if valid_vq_y.numel() > 0 and (
                    valid_vq_y.min() < 0 or valid_vq_y.max() >= model.vq_head.out_features
            ):
                print("[BAD vq_y]")
                print("valid_vq_y min:", int(valid_vq_y.min()))
                print("valid_vq_y max:", int(valid_vq_y.max()))
                print("model.vq_head.out_features:", model.vq_head.out_features)
                raise RuntimeError("vq_y out of range")

            h, tok_logits, vq_logits = model(tok_in, vq_in, key_padding_mask)

            if args.mode == "pretrain":
                tok_loss = torch.tensor(0.0, device=device)

                vq_loss = F.cross_entropy(
                    vq_logits.reshape(-1, vq_logits.size(-1)),
                    vq_y.reshape(-1),
                    ignore_index=-100,
                )

                loss = vq_loss

            elif args.mode == "finetune":
                if args.dict_loss and vq2word_ids is not None:
                    tok_loss = candidate_token_ce_from_hidden_fast(
                        model=model,
                        h=h,
                        tok_y=tok_y,
                        vq_logits=vq_logits,
                        cand_table=cand_table,
                        cand_mask=cand_mask,
                    )
                else:
                    tok_loss = F.cross_entropy(
                        tok_logits.reshape(-1, tok_logits.size(-1)),
                        tok_y.reshape(-1),
                        ignore_index=-100,
                    )

                vq_loss = F.cross_entropy(
                    vq_logits.reshape(-1, vq_logits.size(-1)),
                    vq_y.reshape(-1),
                    ignore_index=-100,
                )

                if args.learn_vq_loss_weight:
                    vq_loss_weight = model.get_vq_loss_weight()
                else:
                    vq_loss_weight = tok_loss.new_tensor(
                        args.aux_lambda
                    )

                if args.main_target == "tok":
                    loss = tok_loss + vq_loss_weight * vq_loss

                elif args.main_target == "vq":
                    loss = vq_loss + vq_loss_weight * tok_loss

                elif args.main_target == "both":
                    loss = tok_loss + vq_loss_weight * vq_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if args.mode == "pretrain":
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    vq=f"{vq_loss.item():.3f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    tok=f"{tok_loss.item():.3f}",
                    vq=f"{vq_loss.item():.3f}",
                    vqw=f"{float(vq_loss_weight.detach()):.4f}",
                )
        if args.learn_vq_loss_weight:
            eval_aux_lambda = float(
                model.get_vq_loss_weight().detach().cpu()
            )
        else:
            eval_aux_lambda = float(args.aux_lambda)

        if args.mode == "pretrain":
            valid = evaluate_vq_only(model, valid_loader, device)
            test = evaluate_vq_only(model, test_loader, device)
        else:
            valid = evaluate(
                model,
                valid_loader,
                device,
                eval_aux_lambda,
                args.main_target,
                vq2word_ids=vq2word_ids,
                dict_loss=args.dict_loss,
                word2vq_prob=word2vq_prob,
                cand_table=cand_table,
                cand_mask=cand_mask,
            )

            test = evaluate(
                model,
                test_loader,
                device,
                eval_aux_lambda,
                args.main_target,
                vq2word_ids=vq2word_ids,
                dict_loss=args.dict_loss,
                word2vq_prob=word2vq_prob,
                cand_table=cand_table,
                cand_mask=cand_mask,
            )
        if args.learn_vq_loss_weight:
            current_vq_weight = float(
                model.get_vq_loss_weight().detach().cpu()
            )
        else:
            current_vq_weight = float(args.aux_lambda)

        print(
            f"[loss-weight] ep={ep} "
            f"vq_loss_weight={current_vq_weight:.6f}"
        )
        pipe_valid = None
        pipe_test = None

        if args.mode == "finetune" and args.dict_loss and vq2word_ids is not None:
            pipe_valid = evaluate_pred_vq_to_word(
                model, valid_loader, device, vq2word_ids, topk=16
            )
            pipe_test = evaluate_pred_vq_to_word(
                model, test_loader, device, vq2word_ids, topk=16
            )

        if args.mode == "pretrain":
            # VQ pretrainではvalid VQ lossをbest判定に使う
            valid_loss = valid["vq_loss"]
            test_loss = test["vq_loss"]
        else:
            # finetuneではvalid token lossをbest判定に使う
            valid_loss = valid["tok_loss"]
            test_loss = test["tok_loss"]

        if args.mode == "pretrain":
            print(
                f"[eval] ep={ep} "
                f"valid_vq_ppl={valid['vq_ppl']:.2f} "
                f"test_vq_ppl={test['vq_ppl']:.2f}"
            )
        else:
            print(
                f"[eval] ep={ep} "
                f"valid_tok_ppl={valid['tok_ppl']:.2f} "
                f"valid_vq_ppl={valid['vq_ppl']:.2f} "
                f"test_tok_ppl={test['tok_ppl']:.2f} "
                f"test_vq_ppl={test['vq_ppl']:.2f}"
            )

        if pipe_valid is not None:
            print(
                f"[pipe] ep={ep} "
                f"valid_vq_acc={pipe_valid['vq_acc']:.4f} "
                f"valid_pred_dict_cov={pipe_valid['pred_dict_coverage']:.4f} "
                f"valid_word_acc={pipe_valid['word_acc']:.4f} "
                f"test_vq_acc={pipe_test['vq_acc']:.4f} "
                f"test_pred_dict_cov={pipe_test['pred_dict_coverage']:.4f} "
                f"test_word_acc={pipe_test['word_acc']:.4f}"
                f"valid_dict_word_ppl={valid['dict_word_ppl']:.2f} "
                f"test_dict_word_ppl={test['dict_word_ppl']:.2f} "
            )
        if word2vq_prob is not None:
            print(
                f"[dict] "
                f"valid_word_ppl={valid['dict_word_ppl']:.2f} "
                f"test_word_ppl={test['dict_word_ppl']:.2f}"
            )
        history["epoch"].append(ep)

        history["valid_loss"].append(valid["main_loss"])
        history["valid_ppl"].append(valid["main_ppl"])
        history["valid_tok_ppl"].append(valid["tok_ppl"])
        history["valid_vq_ppl"].append(valid["vq_ppl"])

        history["test_loss"].append(test["main_loss"])
        history["test_ppl"].append(test["main_ppl"])
        history["test_tok_ppl"].append(test["tok_ppl"])
        history["test_vq_ppl"].append(test["vq_ppl"])

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "data": args.data,
                "epoch": ep,
                "tokenizer": tokenizer_name,
                "token_vocab_size": token_vocab_size,
                "base_vq_vocab_size": base_vq_vocab_size,
                "vq_vocab_size": base_vq_vocab_size,
                "vq_pad_id": vq_pad_id,
                "pad_token_id": pad_token_id,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "history": history,
            }, args.out)
            print(f"[save] {args.out}")

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "args": vars(args),
            "tokenizer": tokenizer_name,
            "vq_vocab_size": base_vq_vocab_size,
            "vq_pad_id": vq_pad_id,
            "pad_token_id": pad_token_id,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "history": history,
        }, args.out.replace(".pt", "_last.pt"))



if __name__ == "__main__":
    main()
