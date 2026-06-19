#!/usr/bin/env python3
import argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


def make_adj(seq_len, hop, device):
    pos = torch.arange(seq_len, device=device)
    dist = (pos[:, None] - pos[None, :]).abs()

    adj = (dist <= hop).float()
    adj.fill_diagonal_(1.0)

    # degree normalize
    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adj = adj / deg
    return adj


class AdjGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.nei_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        # h: [B, L, D]
        # adj: [L, L]
        m = torch.einsum("ij,bjd->bid", adj, h)
        out = self.self_lin(h) + self.nei_lin(m)
        out = F.gelu(out)
        return self.norm(h + out)


class VQWordGNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, hop=3, n_layers=3, dropout=0.1):
        super().__init__()
        self.hop = hop
        self.seq_len = 2 * hop + 1

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)

        self.layers = nn.ModuleList([
            AdjGNNLayer(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

    def encode_context(self, ctx_ids):
        # ctx_ids: [B, 2hop+1]
        B, L = ctx_ids.shape
        pos = torch.arange(L, device=ctx_ids.device).unsqueeze(0).expand(B, L)

        h = self.tok_emb(ctx_ids) + self.pos_emb(pos)
        h = self.dropout(h)

        adj = make_adj(L, self.hop, ctx_ids.device)

        for layer in self.layers:
            h = layer(h, adj)

        z = h[:, self.hop]
        return F.normalize(z, dim=-1)

    def forward(self, ctx_ids, target_ids):
        z = self.encode_context(ctx_ids)
        logits = self.decoder(z)
        loss = F.cross_entropy(logits, target_ids)
        return loss, logits, z


def make_windows(token_ids, hop, pad_id):
    ids = torch.tensor(token_ids, dtype=torch.long)
    padded = F.pad(ids, (hop, hop), value=pad_id)

    ctx, tgt = [], []
    for i in range(len(ids)):
        ctx.append(padded[i:i + 2 * hop + 1])
        tgt.append(ids[i])

    return torch.stack(ctx), torch.tensor(tgt, dtype=torch.long)


@torch.no_grad()
def collect_embeddings(model, ctx, batch_size, device):
    model.eval()
    zs = []

    for start in tqdm(range(0, len(ctx), batch_size), desc="[embed]"):
        xb = ctx[start:start + batch_size].to(device)
        z = model.encode_context(xb)
        zs.append(z.cpu())

    return torch.cat(zs, dim=0)


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
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kmeans_batch_size", type=int, default=8192)
    ap.add_argument("--out", default="vqword_gnn_kmeans.pt")
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

        if len(ids) < 2 * args.hop + 2:
            continue

        ctx, tgt = make_windows(ids, args.hop, tok.pad_token_id)
        all_ctx.append(ctx)
        all_tgt.append(tgt)

    ctx = torch.cat(all_ctx, dim=0)
    tgt = torch.cat(all_tgt, dim=0)

    print(f"[data] windows={len(tgt):,} vocab={tok.vocab_size}")

    model = VQWordGNN(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        hop=args.hop,
        n_layers=args.n_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n = len(tgt)

    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        total_ce = 0.0

        model.train()
        pbar = tqdm(range(0, n, args.batch_size), desc=f"[train] epoch {ep}")

        for start in pbar:
            idx = perm[start:start + args.batch_size]
            xb = ctx[idx].to(device)
            yb = tgt[idx].to(device)

            loss, logits, _ = model(xb, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(idx)
            total_loss += float(loss.detach()) * bs
            total_ce += float(loss.detach()) * bs

            denom = max(1, start + bs)
            pbar.set_postfix(
                loss=total_loss / denom,
                ppl=math.exp(min(20, total_ce / denom)),
            )

        print(
            f"[epoch {ep}] "
            f"ce={total_ce/n:.4f} "
            f"ppl={math.exp(min(20, total_ce/n)):.2f}"
        )

    print("[kmeans] collecting embeddings")
    z = collect_embeddings(model, ctx, args.batch_size, device)
    z_np = z.numpy()

    print("[kmeans] fitting")
    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size,
        batch_size=args.kmeans_batch_size,
        random_state=0,
        verbose=1,
        n_init="auto",
    )

    vq_ids = kmeans.fit_predict(z_np)

    used = len(set(vq_ids.tolist()))
    print(f"[kmeans] used={used}/{args.codebook_size}")

    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    vq_ids = torch.tensor(vq_ids, dtype=torch.long)

    torch.save(
        {
            "model": model.state_dict(),
            "centroids": centroids,
            "vq_ids": vq_ids,
            "ctx": ctx,
            "tgt": tgt,
            "args": vars(args),
            "tokenizer": args.tokenizer,
            "pad_token_id": tok.pad_token_id,
        },
        args.out,
    )

    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()