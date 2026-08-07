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
# 共通設定
# ============================================================
CENTER_SCALE=0
CENTER_LABEL=0

BPE_VOCAB_LABEL=50257
VQ_CODEBOOK_LABEL=100k
VQ_CODEBOOK_SIZE=100000
DISCRETIZATION_SEED=0

IVF_NLIST=256
DECODER_EPOCHS=3
MODEL_VARIANT="deconly"

MAX_SAMPLES=20000
SEQ_LEN=256
BATCH_SIZE=512
K_BLOCK=4096

BPE_ARCHIVE="bpe_wikitext103_50257.tar.gz"
TOKENIZER_DIR="/vqword/bpe_wikitext103_50257"
ASSIGN_SCRIPT="/vqword/assign_vqword_ids.py"

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {hop}"
    echo
    echo "Example:"
    echo "  $0 100"
    exit 1
fi

HOP="$1"

if ! [[ "${HOP}" =~ ^[0-9]+$ ]]; then
    echo "[error] hop must be a non-negative integer: ${HOP}"
    exit 1
fi

# ============================================================
# BPE tokenizerを一度だけ取得・展開
# ============================================================

rm -f "/vqword/${BPE_ARCHIVE}"
rm -rf "${TOKENIZER_DIR}"

echo "============================================================"
echo "[download BPE tokenizer]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${BPE_ARCHIVE}" \
  -o "/vqword/${BPE_ARCHIVE}"

bye
EOF

tar -xzf "/vqword/${BPE_ARCHIVE}" -C /vqword

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] Tokenizer directory was not created:"
  echo "        ${TOKENIZER_DIR}"
  exit 1
fi

if [ ! -f "${ASSIGN_SCRIPT}" ]; then
  echo "[error] Assignment script was not found:"
  echo "        ${ASSIGN_SCRIPT}"
  exit 1
fi

# ============================================================
# HOP 0〜10
# ============================================================

HOP2=$(printf "%02d" "${HOP}")

BASE_TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center${CENTER_LABEL}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"

VQ_CKPT="wikitext103_vqword_${BASE_TAG}.pt"
VQ_CKPT_PATH="/vqword/${VQ_CKPT}"

OUT="tinystories_vqword_${BASE_TAG}_ids.pt"
OUT_PATH="/vqword/${OUT}"

echo
echo "============================================================"
echo "[HOP ${HOP}]"
echo "checkpoint = ${VQ_CKPT}"
echo "output     = ${OUT}"
echo "============================================================"

rm -f "${VQ_CKPT_PATH}"
rm -f "${OUT_PATH}"
rm -f "${OUT_PATH}.part"*

# ----------------------------------------------------------
# checkpoint取得
# ----------------------------------------------------------

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${VQ_CKPT}" \
  -o "${VQ_CKPT_PATH}"

bye
EOF

  if [ ! -f "${VQ_CKPT_PATH}" ]; then
    echo "[error] checkpoint not found: ${VQ_CKPT_PATH}"
    exit 1
  fi

  # ----------------------------------------------------------
  # checkpoint確認
  # ----------------------------------------------------------

  python - <<PY
import torch
expected_center_scale = float("${CENTER_SCALE}")
path = "${VQ_CKPT_PATH}"
expected_hop = int("${HOP}")
expected_vq_vocab = int("${VQ_CODEBOOK_SIZE}")
expected_bpe_vocab = int("${BPE_VOCAB_LABEL}")

ckpt = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

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
    raise ValueError(f"Missing checkpoint keys: {missing}")

args = ckpt["args"]
actual_hop = int(args["hop"])
actual_center_scale = float(
    args.get("center_scale", args.get("center_weight", -1.0))
)
actual_vq_vocab = int(ckpt["vq_vocab_size"])
actual_bpe_vocab = int(
    ckpt["model"]["tok_emb.weight"].shape[0]
)
if abs(actual_center_scale - expected_center_scale) > 1e-8:
    raise ValueError(
        f"Center scale mismatch: expected={expected_center_scale}, "
        f"actual={actual_center_scale}"
    )
if actual_hop != expected_hop:
    raise ValueError(
        f"HOP mismatch: expected={expected_hop}, actual={actual_hop}"
    )

if actual_vq_vocab != expected_vq_vocab:
    raise ValueError(
        f"VQ vocab mismatch: expected={expected_vq_vocab}, "
        f"actual={actual_vq_vocab}"
    )

if actual_bpe_vocab != expected_bpe_vocab:
    raise ValueError(
        f"BPE vocab mismatch: expected={expected_bpe_vocab}, "
        f"actual={actual_bpe_vocab}"
    )

print("[checkpoint check] OK")
print("hop:", actual_hop)
print("vq_vocab_size:", actual_vq_vocab)
print("center_scale:", actual_center_scale)
print("bpe_vocab_size:", actual_bpe_vocab)
PY

  # ----------------------------------------------------------
  # TinyStories ID付与
  # ----------------------------------------------------------

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

  # ----------------------------------------------------------
  # 出力確認
  # ----------------------------------------------------------

  python - <<PY
import torch

path = "${OUT_PATH}"
expected_hop = int("${HOP}")

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

required = {
    "samples",
    "vq_ids_flat",
    "token_ids_flat",
    "vq_vocab_size",
    "vq_pad_id",
    "hop",
}

missing = sorted(required - set(data.keys()))
if missing:
    raise ValueError(f"Missing output keys: {missing}")

token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        f"Length mismatch: token={token_ids.numel()}, "
        f"vq={vq_ids.numel()}"
    )

if int(data["hop"]) != expected_hop:
    raise ValueError(
        f"Output HOP mismatch: expected={expected_hop}, "
        f"actual={data['hop']}"
    )

if vq_ids.numel() and int(vq_ids.max()) >= int(data["vq_vocab_size"]):
    raise ValueError(
        f"VQ ID out of range: max={int(vq_ids.max())}, "
        f"vocab={int(data['vq_vocab_size'])}"
    )

print("[output check] OK")
print("samples:", len(data["samples"]))
print("tokens:", token_ids.numel())
print("hop:", data["hop"])
print("vq min/max:", int(vq_ids.min()), int(vq_ids.max()))
PY

  # ----------------------------------------------------------
  # FTPアップロード
  # ----------------------------------------------------------

  FILE_SIZE=$(stat -c%s "${OUT_PATH}")

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
    split \
      -b 450M \
      -d \
      -a 3 \
      "${OUT_PATH}" \
      "${OUT_PATH}.part"

    lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

mput "${OUT_PATH}.part"*

bye
EOF
  fi

  echo "[completed HOP ${HOP}] ${OUT}"

  # 次のHOP用に巨大checkpointだけ削除
  rm -f "${VQ_CKPT_PATH}"

echo
echo "============================================================"
echo "[HOP 50 completed]"
echo "============================================================"