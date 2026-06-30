#!/usr/bin/env python3
import argparse, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class VQWordARDataset(Dataset):
    def __init__(self, samples):
        self.samples = []
        for s in samples:
            tok = s["token_ids"].long()
            vq = s["vqword_ids"].long()
            if len(tok) >= 4:
                self.samples.append((tok, vq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        tok, vq = self.samples[i]
        return {
            "tok_in": tok[:-1],
            "vq_in": vq[:-1],
            "tok_y": tok[1:],
            "vq_y": vq[1:],
        }


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
    ):
        super().__init__()
        self.use_token_input = use_token_input
        self.use_vq_input = use_vq_input

        self.vq_emb = nn.Embedding(vq_vocab_size, d_model) if use_vq_input else None
        self.tok_emb = nn.Embedding(vocab_size, d_model) if use_token_input else None
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.vq_to_tok = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.tok_head = nn.Linear(d_model, vocab_size)
        self.vq_head = nn.Linear(d_model, vq_vocab_size)

    def forward(self, tok_in, vq_in, key_padding_mask=None):
        B, L = tok_in.shape
        pos = torch.arange(L, device=tok_in.device)[None, :]

        h = self.pos_emb(pos).expand(B, L, -1)

        if self.use_vq_input:
            h = h + self.vq_emb(vq_in)

        if self.use_token_input:
            h = h + self.tok_emb(tok_in)
        causal = torch.triu(
            torch.ones(L, L, device=tok_in.device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.tr(
            h,
            mask=causal,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.norm(h)
        # forward内
        tok_logits = self.tok_head(h)
        vq_logits = self.vq_head(h)

        return tok_logits, vq_logits

@torch.no_grad()
def evaluate(model, loader, device, aux_lambda, main_target):
    model.eval()
    total_tok_loss = 0.0
    total_vq_loss = 0.0
    total_main_loss = 0.0
    total_tok = 0

    for tok_in, vq_in, tok_y, vq_y, attn_mask in loader:
        tok_in = tok_in.to(device)
        vq_in = vq_in.to(device)
        tok_y = tok_y.to(device)
        vq_y = vq_y.to(device)
        attn_mask = attn_mask.to(device)

        key_padding_mask = ~attn_mask
        tok_logits, vq_logits, tok_logits_from_vq = model(tok_in, vq_in, key_padding_mask)

        tok_loss = F.cross_entropy(
            tok_logits.reshape(-1, tok_logits.size(-1)),
            tok_y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        tok_from_vq_loss = F.cross_entropy(
            tok_logits_from_vq.reshape(-1, tok_logits_from_vq.size(-1)),
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
            main_loss = tok_from_vq_loss

        total_tok_loss += tok_loss.item()
        total_vq_loss += vq_loss.item()
        total_main_loss += main_loss.item()
        total_tok += n

    tok_ce = total_tok_loss / max(total_tok, 1)
    vq_ce = total_vq_loss / max(total_tok, 1)
    main_ce = total_main_loss / max(total_tok, 1)

    return {
        "main_loss": main_ce,
        "main_ppl": math.exp(min(main_ce, 20)),
        "tok_loss": tok_ce,
        "tok_ppl": math.exp(min(tok_ce, 20)),
        "vq_loss": vq_ce,
        "vq_ppl": math.exp(min(vq_ce, 20)),
    }

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
    ap.add_argument("--reset_heads", action="store_true")
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
    samples = data["samples"]

    tokenizer_name = args.tokenizer or data.get("tokenizer", None)
    if tokenizer_name is None:
        raise ValueError("Tokenizer is not specified. Use --tokenizer or include data['tokenizer'].")

    print(f"[data] {args.data}")
    print(f"[tokenizer] {tokenizer_name}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pad_token_id = tok.pad_token_id

    token_vocab_size = args.token_vocab_size or len(tok)

    if args.vq_vocab_size is not None:
        base_vq_vocab_size = args.vq_vocab_size
    elif "vq_vocab_size" in data:
        base_vq_vocab_size = int(data["vq_vocab_size"])
    else:
        base_vq_vocab_size = int(data["vq_ids_flat"].max().item()) + 1

    vq_pad_id = base_vq_vocab_size
    vq_vocab_size = base_vq_vocab_size + 1

    print(f"[token_vocab_size] {token_vocab_size}")
    print(f"[base_vq_vocab_size] {base_vq_vocab_size}")
    print(f"[vq_pad_id] {vq_pad_id}")
    print(f"[vq_vocab_size incl pad] {vq_vocab_size}")

    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    train_s = samples[:n_train]
    valid_s = samples[n_train:n_train + n_valid]
    test_s = samples[n_train + n_valid:]

    train_ds = VQWordARDataset(train_s)
    valid_ds = VQWordARDataset(valid_s)
    test_ds = VQWordARDataset(test_s)

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
        vq_vocab_size=vq_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_len=1024,
        use_token_input=use_token_input,
        use_vq_input=use_vq_input,
    ).to(device)

    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu")
        sd = ckpt["model"]

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init_from] {args.init_from}")
        print(f"[init] missing={missing}")
        print(f"[init] unexpected={unexpected}")

        if args.reset_heads:
            print("[init] reset tok_head / vq_head")
            model.tok_head.reset_parameters()
            model.vq_head.reset_parameters()

    if args.freeze_vq_backbone:
        for p in model.parameters():
            p.requires_grad = False

        for p in model.tok_head.parameters():
            p.requires_grad = True

        print("[freeze] train only tok_head")

        print("[freeze] train only vq_to_tok")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
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
            tok_logits, vq_logits, tok_logits_from_vq = model(tok_in, vq_in, key_padding_mask)

            tok_loss = F.cross_entropy(
                tok_logits.reshape(-1, tok_logits.size(-1)),
                tok_y.reshape(-1),
                ignore_index=-100,
            )

            tok_from_vq_loss = F.cross_entropy(
                tok_logits_from_vq.reshape(-1, tok_logits_from_vq.size(-1)),
                tok_y.reshape(-1),
                ignore_index=-100,
            )

            vq_loss = F.cross_entropy(
                vq_logits.reshape(-1, vq_logits.size(-1)),
                vq_y.reshape(-1),
                ignore_index=-100,
            )

            if args.main_target == "tok":
                loss = tok_loss + args.aux_lambda * vq_loss

            elif args.main_target == "vq":
                loss = vq_loss + args.aux_lambda * tok_loss

            elif args.main_target == "both":
                loss = tok_loss + args.aux_lambda * vq_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                tok=f"{tok_loss.item():.3f}",
                vq=f"{vq_loss.item():.3f}",
                vq2tok=f"{tok_from_vq_loss.item():.3f}",
            )

        valid = evaluate(model, valid_loader, device, args.aux_lambda, args.main_target)
        test = evaluate(model, test_loader, device, args.aux_lambda, args.main_target)

        valid_loss = valid["main_loss"]
        test_loss = test["main_loss"]

        print(
            f"[eval] ep={ep} "
            f"valid_tok_ppl={valid['tok_ppl']:.2f} valid_vq_ppl={valid['vq_ppl']:.2f} "
            f"valid_main_loss={valid['main_loss']:.4f} "
            f"test_tok_ppl={test['tok_ppl']:.2f} test_vq_ppl={test['vq_ppl']:.2f} "
            f"test_main_loss={test['main_loss']:.4f}"
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
                "tokenizer": tokenizer_name,
                "token_vocab_size": token_vocab_size,
                "base_vq_vocab_size": base_vq_vocab_size,
                "vq_vocab_size": vq_vocab_size,
                "vq_pad_id": vq_pad_id,
                "pad_token_id": pad_token_id,
                "valid_loss": valid_loss,
                "test_loss": test_loss,
                "history": history,
            }, args.out)
            print(f"[save] {args.out}")

        torch.save({
            "model": model.state_dict(),
            "args": vars(args),
            "tokenizer": data["tokenizer"],
            "vq_vocab_size": vq_vocab_size,
            "vq_pad_id": vq_pad_id,
            "pad_token_id": pad_token_id,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "history": history,
        }, args.out.replace(".pt", "_last.pt"))



if __name__ == "__main__":
    main()