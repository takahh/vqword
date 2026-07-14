import torch

d = torch.load(
    "/Users/taka/Downloads/wikitext103_bpe_vqword_selfbpe_left10_center0_global_ivf256_dictionary.pt",
    map_location="cpu",
    weights_only=False,
)

vq_to_bpe = d["vq_to_bpe_ids"]

purity = []

for vq_id, items in vq_to_bpe.items():
    if len(items) == 0:
        continue

    total = sum(cnt for _, cnt in items)
    top = max(cnt for _, cnt in items)

    purity.append(top / total)

purity = torch.tensor(purity)

print("mean :", purity.mean().item())
print("median :", purity.median().item())
print("q90 :", torch.quantile(purity, 0.9).item())
print("min :", purity.min().item())