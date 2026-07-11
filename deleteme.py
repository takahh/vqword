import torch

path = "/Users/taka/Downloads/wikitext103_bpe_vqword_selfbpe_left10_center0_pertok_f03_ids.pt"
d = torch.load(path, map_location="cpu")

print(d.keys())
print()

total = 0

for key, value in d.items():
    if torch.is_tensor(value):
        size_bytes = value.numel() * value.element_size()
        total += size_bytes

        print(
            f"{key:20s}",
            f"dtype={str(value.dtype):12s}",
            f"shape={tuple(value.shape)}",
            f"size={size_bytes / 1024**3:.3f} GiB",
        )
    else:
        print(f"{key:20s} type={type(value).__name__}")

print(f"\nTensor total: {total / 1024**3:.3f} GiB")