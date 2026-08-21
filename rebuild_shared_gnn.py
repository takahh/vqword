#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.nei_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        m = torch.einsum("ij,bjd->bid", adj, h)
        out = self.self_lin(h) + self.nei_lin(m)
        out = F.gelu(out)
        return self.norm(h + out)

class VQWordGNN(nn.Module):
    def __init__(self, vocab_size, d_model=256, hop=10, n_layers=1,
                 dropout=0.1, center_scale=1.0):
        super().__init__()
        if int(n_layers) != 1:
            raise ValueError("tied recurrent HOP requires n_layers=1")
        self.hop = int(hop)
        self.seq_len = 2 * self.hop + 1
        self.center_idx = self.hop
        self.center_scale = float(center_scale)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.seq_len, d_model)
        self.shared_gnn = AdjGNNLayer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(d_model, vocab_size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    part = torch.load(args.hop_file, map_location="cpu", weights_only=False)
    sig = part.get("signature")
    if not isinstance(sig, dict):
        raise ValueError("HOP file has no usable signature")

    required = {
        "hops", "seed", "local_clusters", "d_model",
        "center_scale", "tokenizer_name", "vocab_size",
        "encoder_max_hop",
    }
    missing = required - set(sig.keys())
    if missing:
        raise ValueError(f"signature missing keys: {sorted(missing)}")

    seed = int(sig["seed"])
    vocab_size = int(sig["vocab_size"])
    d_model = int(sig["d_model"])
    max_hop = int(sig["encoder_max_hop"])
    center_scale = float(sig["center_scale"])
    hops = [int(h) for h in sig["hops"]]

    # Original generator calls torch.manual_seed(seed) before constructing
    # this model and does not train the GNN before clustering.
    torch.manual_seed(seed)

    model = VQWordGNN(
        vocab_size=vocab_size,
        d_model=d_model,
        hop=max_hop,
        n_layers=1,
        center_scale=center_scale,
    )
    model.eval()

    out = {
        "signature": sig,
        "model": model.state_dict(),
        "args": {
            "d_model": d_model,
            "n_layers": 1,
            "center_scale": center_scale,
            "tokenizer": sig["tokenizer_name"],
        },
        "tokenizer_name": sig["tokenizer_name"],
        "vocab_type": "byte_bpe",
        "partitioned": True,
        "partition_type": "bpe_local_kmeans",
        "max_local_clusters": int(sig["local_clusters"]),
        "context_type": "bilateral",
        "hop": max_hop,
        "hops": hops,
        "context_width": 2 * max_hop + 1,
        "shared_gnn_across_hops": True,
        "shared_codebook_across_hops": False,
        "hop_embedding": False,
        "id_scheme": "(bpe_id, local_vq_id)",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[reconstructed shared model] {out_path}")
    print(f"  seed={seed} vocab={vocab_size} d_model={d_model} max_hop={max_hop}")
    print(f"  hops={hops} center_scale={center_scale}")

if __name__ == "__main__":
    main()
