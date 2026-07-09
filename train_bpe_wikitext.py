from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

ds = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
)

with open("wikitext103_train.txt", "w", encoding="utf-8") as f:
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

out_dir = Path("bpe_wikitext103_50k")
out_dir.mkdir(exist_ok=True)

# Save vocab.json + merges.txt
tokenizer.save_model(str(out_dir))

# Save Hugging Face tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer._tokenizer,
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
)

hf_tokenizer.save_pretrained(str(out_dir))

print("Saved tokenizer to", out_dir)
print("Vocab size:", hf_tokenizer.vocab_size)