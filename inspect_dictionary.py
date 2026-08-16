import torch
from collections import defaultdict, Counter

raw = torch.load("wikitext103_vqword_bpe50257_bilateral10_center1_deconly_dec3_global_ivf256_vqcb25k_seed0_dictionary.pt", map_location="cpu")

dict_entries = {
    int(k): v
    for k, v in raw.items()
    if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
}

tok_to_vqs = defaultdict(list)

for vq_id, entries in dict_entries.items():
    wid, word, cnt = entries[0]
    tok_to_vqs[int(wid)].append(int(vq_id))

sizes = [len(v) for v in tok_to_vqs.values()]

print("num token covered:", len(tok_to_vqs))
print("avg VQ per token:", sum(sizes) / len(sizes))
print("max VQ per token:", max(sizes))
print("size hist top:", Counter(sizes).most_common(20))

for wid, vqs in sorted(tok_to_vqs.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
    print(wid, len(vqs), vqs[:10])