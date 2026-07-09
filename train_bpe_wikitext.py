from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

ds = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train"
)

with open("wikitext103_train.txt", "w") as f:
    for x in ds:
        t = x["text"]
        if t.strip():
            f.write(t.replace("\n", " ") + "\n")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["wikitext103_train.txt"],
    vocab_size=50257,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
)

tokenizer.save_model("bpe_wikitext103_50k")