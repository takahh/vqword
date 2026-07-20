set -euo pipefail

pip install datasets tokenizers transformers
apt update
apt install -y lftp

cd /

if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi

cd /vqword
git pull

cat > train_bpe_wikitext103_50257.py <<'PY'
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 50257
OUT_DIR = Path("/vqword/bpe_wikitext103_50257")
TEXT_FILE = Path("/vqword/wikitext103_train.txt")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
)

with TEXT_FILE.open("w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"]
        if text.strip():
            f.write(text.replace("\n", " ") + "\n")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[str(TEXT_FILE)],
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer.save_model(str(OUT_DIR))

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer._tokenizer,
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
)

hf_tokenizer.save_pretrained(str(OUT_DIR))

print("Saved tokenizer to:", OUT_DIR)
print("vocab_size:", hf_tokenizer.vocab_size)
print("len(tokenizer):", len(hf_tokenizer))
print("special_tokens_map:", hf_tokenizer.special_tokens_map)
PY

python /vqword/train_bpe_wikitext103_50257.py