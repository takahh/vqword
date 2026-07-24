#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y lftp

pip install \
  torch \
  datasets \
  transformers \
  scikit-learn \
  tqdm \
  numpy

cd /

if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi

cd /vqword
git pull

# ============================================================
# Step4: TinyStoriesにVQWord IDを付与
#
# 入力:
#   1. WikiText-103で作成したBPE tokenizer
#   2. WikiText-103で作成したVQWord tokenizer/checkpoint
#
# 出力:
#   TinyStoriesのBPE ID + VQWord ID
# ============================================================

BPE_VOCAB_LABEL=50257

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 使用方法
#
#   bash assign_tinystories_vqword.sh 25k
#   bash assign_tinystories_vqword.sh 50k
#   bash assign_tinystories_vqword.sh 100k
#   bash assign_tinystories_vqword.sh 200k
#
# 25k / 50k / 100k:
#   checkpoint名に _seed0 が付く
#
# 200k:
#   既存ファイルとの互換性のため _seed0 なし
# ============================================================

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {25k|50k|100k|200k}"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"

case "${VQ_CODEBOOK_LABEL}" in
  25k)
    VQ_CODEBOOK_SIZE=25000
    VQ_SEED=0
    VQ_FILENAME_SUFFIX="_seed${VQ_SEED}"
    ;;
  50k)
    VQ_CODEBOOK_SIZE=50000
    VQ_SEED=0
    VQ_FILENAME_SUFFIX="_seed${VQ_SEED}"
    ;;
  100k)
    VQ_CODEBOOK_SIZE=100000
    VQ_SEED=0
    VQ_FILENAME_SUFFIX="_seed${VQ_SEED}"
    ;;
  200k)
    VQ_CODEBOOK_SIZE=200000
    VQ_SEED=0
    VQ_FILENAME_SUFFIX=""
    ;;
  *)
    echo "[error] Unsupported VQ vocabulary size: ${VQ_CODEBOOK_LABEL}"
    echo "Supported values: 25k, 50k, 100k, 200k"
    exit 1
    ;;
esac

HOP=20
IVF_NLIST=256

MAX_SAMPLES=20000
SEQ_LEN=256
BATCH_SIZE=512
K_BLOCK=4096

# ============================================================
# ファイル名
# ============================================================

BASE_TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center0_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}"

# checkpoint側のタグ
# 25k / 50k / 100k は _seed0 付き
# 200kは既存ファイルに合わせてseed表記なし
VQ_TAG="${BASE_TAG}${VQ_FILENAME_SUFFIX}"

BPE_ARCHIVE="bpe_wikitext103_50257.tar.gz"
TOKENIZER_DIR="/vqword/bpe_wikitext103_50257"

VQ_CKPT="wikitext103_vqword_${VQ_TAG}.pt"
VQ_CKPT_PATH="/vqword/${VQ_CKPT}"

# 出力には必ずVQ vocab sizeを含める
# seedも含めて、どのtokenizerから作ったか明確にする
OUT="tinystories_vqword_${VQ_TAG}_ids.pt"
OUT_PATH="/vqword/${OUT}"

ASSIGN_SCRIPT="/vqword/assign_vqword_ids.py"

# ============================================================
# FTP設定
# ============================================================

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

echo "============================================================"
echo "[configuration]"
echo "BPE tokenizer        = ${BPE_ARCHIVE}"
echo "VQ checkpoint        = ${VQ_CKPT}"
echo "VQ codebook label    = ${VQ_CODEBOOK_LABEL}"
echo "VQ codebook size     = ${VQ_CODEBOOK_SIZE}"
echo "VQ seed              = ${VQ_SEED}"
echo "context              = left ${HOP}"
echo "TinyStories samples  = ${MAX_SAMPLES}"
echo "sequence length      = ${SEQ_LEN}"
echo "batch size           = ${BATCH_SIZE}"
echo "output               = ${OUT}"
echo "============================================================"

# ============================================================
# 古いファイルを削除
# ============================================================

rm -f "/vqword/${BPE_ARCHIVE}"
rm -f "${VQ_CKPT_PATH}"
rm -f "${OUT_PATH}"
rm -rf "${TOKENIZER_DIR}"

# ============================================================
# BPE tokenizerとVQWord checkpointをFTPから取得
# ============================================================

echo "============================================================"
echo "[download files from FTP]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${BPE_ARCHIVE}" \
  -o "/vqword/${BPE_ARCHIVE}"

get "${VQ_CKPT}" \
  -o "${VQ_CKPT_PATH}"

bye
EOF

echo "============================================================"
echo "[downloaded files]"
echo "============================================================"

ls -lh \
  "/vqword/${BPE_ARCHIVE}" \
  "${VQ_CKPT_PATH}"

# ============================================================
# BPE tokenizerを展開
# ============================================================

echo "============================================================"
echo "[extract BPE tokenizer]"
echo "============================================================"

tar -xzf "/vqword/${BPE_ARCHIVE}" -C /vqword

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] Tokenizer directory was not created:"
  echo "        ${TOKENIZER_DIR}"
  echo
  echo "[archive contents]"
  tar -tzf "/vqword/${BPE_ARCHIVE}" | head -50
  exit 1
fi

ls -lh "${TOKENIZER_DIR}"

# ============================================================
# 必要ファイルを確認
# ============================================================

if [ ! -f "${ASSIGN_SCRIPT}" ]; then
  echo "[error] Assignment script was not found:"
  echo "        ${ASSIGN_SCRIPT}"
  exit 1
fi

if [ ! -f "${VQ_CKPT_PATH}" ]; then
  echo "[error] VQWord checkpoint was not found:"
  echo "        ${VQ_CKPT_PATH}"
  exit 1
fi

python - <<PY
import torch
from transformers import AutoTokenizer

tokenizer_path = "${TOKENIZER_DIR}"
checkpoint_path = "${VQ_CKPT_PATH}"

print("============================================================")
print("[input verification]")

tok = AutoTokenizer.from_pretrained(tokenizer_path)

print("tokenizer:", tokenizer_path)
print("vocab_size:", tok.vocab_size)
print("len(tokenizer):", len(tok))
print("pad_token_id:", tok.pad_token_id)
print("unk_token_id:", tok.unk_token_id)

ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

print("checkpoint:", checkpoint_path)
print("checkpoint keys:", list(ckpt.keys()))
print("vq_vocab_size:", ckpt.get("vq_vocab_size"))
print("tokenizer_name:", ckpt.get("tokenizer_name"))

required = {
    "model",
    "ivf_centers",
    "global_centers",
    "global_offsets",
    "vq_vocab_size",
    "args",
}

missing = sorted(required - set(ckpt.keys()))

if missing:
    raise ValueError(
        f"Missing checkpoint keys: {missing}"
    )

expected_vq_vocab_size = int("${VQ_CODEBOOK_SIZE}")
expected_bpe_vocab_size = int("${BPE_VOCAB_LABEL}")

actual_vq_vocab_size = int(ckpt["vq_vocab_size"])

if actual_vq_vocab_size != expected_vq_vocab_size:
    raise ValueError(
        "VQ vocabulary mismatch: "
        f"expected={expected_vq_vocab_size:,}, "
        f"actual={actual_vq_vocab_size:,}"
    )

model_vocab_size = int(
    ckpt["model"]["tok_emb.weight"].shape[0]
)

if model_vocab_size != expected_bpe_vocab_size:
    raise ValueError(
        "BPE vocabulary mismatch: "
        f"expected={expected_bpe_vocab_size:,}, "
        f"actual={model_vocab_size:,}"
    )

print("[check] OK")
print("============================================================")
PY

# ============================================================
# TinyStoriesにVQWord IDを付与
#
# TinyStoriesはPython内部のload_dataset()で取得される
# ============================================================

echo "============================================================"
echo "[assign VQWord IDs to TinyStories]"
echo "============================================================"

python "${ASSIGN_SCRIPT}" \
  --ckpt "${VQ_CKPT_PATH}" \
  --dataset roneneldan/TinyStories \
  --split train \
  --text_col text \
  --max_samples "${MAX_SAMPLES}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --k_block "${K_BLOCK}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --out "${OUT_PATH}"

# ============================================================
# 出力を確認
# ============================================================

if [ ! -f "${OUT_PATH}" ]; then
  echo "[error] Output file was not created:"
  echo "        ${OUT_PATH}"
  exit 1
fi

echo "============================================================"
echo "[generated file]"
echo "============================================================"

ls -lh "${OUT_PATH}"

python - <<PY
import torch

path = "${OUT_PATH}"

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

print("============================================================")
print("[output verification]")
print("path:", path)
print("keys:", list(data.keys()))

required = {
    "samples",
    "vq_ids_flat",
    "token_ids_flat",
    "vq_vocab_size",
    "vq_pad_id",
}

missing = sorted(required - set(data.keys()))

if missing:
    raise ValueError(
        f"Missing output keys: {missing}"
    )

token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)

print("samples:", f"{len(data['samples']):,}")
print("token IDs:", f"{token_ids.numel():,}")
print("VQWord IDs:", f"{vq_ids.numel():,}")
print("token min/max:", int(token_ids.min()), int(token_ids.max()))
print("VQWord min/max:", int(vq_ids.min()), int(vq_ids.max()))
print("vq_vocab_size:", data["vq_vocab_size"])
print("vq_pad_id:", data["vq_pad_id"])
print("hop:", data.get("hop"))
print("tokenizer:", data.get("tokenizer"))
print("checkpoint:", data.get("ckpt"))

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        "Token/VQWord length mismatch: "
        f"token={token_ids.numel():,}, "
        f"vq={vq_ids.numel():,}"
    )

if int(vq_ids.max()) >= int(data["vq_vocab_size"]):
    raise ValueError(
        "VQWord ID is out of range: "
        f"max={int(vq_ids.max()):,}, "
        f"vocab={int(data['vq_vocab_size']):,}"
    )

print("[check] OK")
print("============================================================")
PY

# ============================================================
# FTPへアップロード
# ============================================================

echo "============================================================"
echo "[upload output]"
echo "============================================================"

FILE_SIZE=$(stat -c%s "${OUT_PATH}")

# Lolipopの単一ファイル上限を考慮し、
# 1.8GB未満ならそのまま、超える場合は450MBずつに分割
if [ "${FILE_SIZE}" -lt 1800000000 ]; then

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

put "${OUT_PATH}" \
  -o "${OUT}"

bye
EOF

else

  echo "[note] Output is large; splitting before upload"

  rm -f "${OUT_PATH}.part"*

  split \
    -b 450M \
    -d \
    -a 3 \
    "${OUT_PATH}" \
    "${OUT_PATH}.part"

  ls -lh "${OUT_PATH}.part"*

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

mput "${OUT_PATH}.part"*

bye
EOF

fi

echo "============================================================"
echo "[completed]"
echo "VQ checkpoint = ${VQ_CKPT}"
echo "VQ vocab label = ${VQ_CODEBOOK_LABEL}"
echo "VQ vocab size  = ${VQ_CODEBOOK_SIZE}"
echo "VQ seed        = ${VQ_SEED}"
echo "BPE tokenizer = ${BPE_ARCHIVE}"
echo "TinyStories   = ${MAX_SAMPLES} samples"
echo "output        = ${OUT}"
echo "============================================================"