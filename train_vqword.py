#!/usr/bin/env python3
import argparse, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class VQWord(nn.Module):
    def __init__(self, vocab_size, codebook_size=8192, d_model=256, hop=3, beta=0.25):
        super().__init__()
        self.hop = hop
        self.beta = beta
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2 * hop + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, std=0.02)

        self.decoder = nn.Linear(d_model, vocab_size)

    def encode_context(self, ctx_ids):
        # ctx_ids: [B, 2hop+1]
        b, l = ctx_ids.shape
        pos = torch.arange(l, device=ctx_ids.device).unsqueeze(0).expand(b, l)
        h = self.tok_emb(ctx_ids) + self.pos_emb(pos)
        h = self.encoder(h)
        return F.normalize(h[:, self.hop], dim=-1)

    def quantize(self, z):
        code = F.normalize(self.codebook.weight, dim=-1)
        sim = z @ code.t()
        ids = sim.argmax(dim=-1)
        zq = code[ids]

        commit = F.mse_loss(z, zq.detach())
        codebook = F.mse_loss(zq, z.detach())
        z_st = z + (zq - z).detach()

        return z_st, ids, commit, codebook

    def forward(self, ctx_ids, target_ids):
        z = self.encode_context(ctx_ids)
        zq, vq_ids, commit, codebook = self.quantize(z)
        logits = self.decoder(zq)

        ce = F.cross_entropy(logits, target_ids)
        loss = ce + self.beta * commit + codebook

        return loss, ce.detach(), commit.detach(), codebook.detach(), vq_ids


def make_windows(token_ids, hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (hop, hop), value=pad_id)

    ctx, tgt = [], []
    for i in range(len(ids)):
        ctx.append(padded[i:i + 2 * hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hop", type=int, default=3)
    ap.add_argument("--codebook_size", type=int, default=8192)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="vqword.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(args.dataset, split="train")

    all_ctx, all_tgt = [], []

    print("[data] tokenizing")
    for ex in tqdm(ds.select(range(min(args.max_samples, len(ds))))):
        text = ex[args.text_col]
        ids = tok.encode(text, add_special_tokens=False)[:args.seq_len]
        if len(ids) < 8:
            continue
        ctx, tgt = make_windows(ids, args.hop, tok.pad_token_id)
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print(f"[data] windows={len(tgt):,} vocab={tok.vocab_size}")

    model = VQWord(
        vocab_size=tok.vocab_size,
        codebook_size=args.codebook_size,
        d_model=args.d_model,
        hop=args.hop,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(tgt)

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        total_loss = total_ce = total_commit = total_codebook = 0.0
        used = set()

        model.train()
        pbar = tqdm(range(0, n, args.batch_size), desc=f"epoch {ep}")

        for start in pbar:
            idx = perm[start:start + args.batch_size]
            xb = ctx[idx].to(device)
            yb = tgt[idx].to(device)

            loss, ce, commit, codebook, vq_ids = model(xb, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(idx)
            total_loss += float(loss) * bs
            total_ce += float(ce) * bs
            total_commit += float(commit) * bs
            total_codebook += float(codebook) * bs
            used.update(vq_ids.detach().cpu().tolist())

            pbar.set_postfix(
                loss=total_loss / max(1, start + bs),
                ce=total_ce / max(1, start + bs),
                used=len(used),
            )

        print(
            f"[epoch {ep}] "
            f"loss={total_loss/n:.4f} "
            f"ce={total_ce/n:.4f} "
            f"ppl={math.exp(min(20, total_ce/n)):.2f} "
            f"commit={total_commit/n:.4f} "
            f"codebook={total_codebook/n:.4f} "
            f"used={len(used)}/{args.codebook_size}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "tokenizer": args.tokenizer,
            "pad_token_id": tok.pad_token_id,
        },
        args.out,
    )
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()