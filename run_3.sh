#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# WikiText-103 multi-HOP VQWord generation
#
# Generate 10 independent bilateral-context tokenizers:
#   hop=1, 2, ..., 10
#
# Usage:
#   FTP_PASS='...' bash run_vqword_multihop_wikitext.sh 100k 0.0
#
# Arguments:
#   $1: VQW codebook size: 25k|50k|100k|200k|300k
#   $2: center scale
# ============================================================

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

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 {25k|50k|100k|200k|300k} {center_scale} {hop}"
  echo
  echo "Example:"
  echo "  FTP_PASS='your-password' $0 100k 0.0 5"
  exit 1
fi

CB_SIZE="$1"
CENTER_SCALE_RAW="$2"
HOP="$3"
if ! [[ "${HOP}" =~ ^[0-9]+$ ]]; then
  echo "[error] HOP must be an integer: ${HOP}"
  exit 1
fi

if (( HOP < 0 || HOP > 10 )); then
  echo "[error] HOP must be between 0 and 10: ${HOP}"
  exit 1
fi

CENTER_SCALE="$(
  python -c '
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(
        f"[error] center_scale must be numeric: {sys.argv[1]}"
    )

if not math.isfinite(value):
    raise SystemExit(
        f"[error] center_scale must be finite: {value}"
    )

if value < 0:
    raise SystemExit(
        f"[error] center_scale must be >= 0: {value}"
    )

print(f"{value:g}")
' "${CENTER_SCALE_RAW}"
)"

case "${CB_SIZE}" in
  25k)
    VQ_CODEBOOK_LABEL=25k
    VQ_CODEBOOK_SIZE=25000
    ;;
  50k)
    VQ_CODEBOOK_LABEL=50k
    VQ_CODEBOOK_SIZE=50000
    ;;
  100k)
    VQ_CODEBOOK_LABEL=100k
    VQ_CODEBOOK_SIZE=100000
    ;;
  200k)
    VQ_CODEBOOK_LABEL=200k
    VQ_CODEBOOK_SIZE=200000
    ;;
  300k)
    VQ_CODEBOOK_LABEL=300k
    VQ_CODEBOOK_SIZE=300000
    ;;
  *)
    echo "[error] Invalid codebook size: ${CB_SIZE}"
    echo "Usage: $0 {25k|50k|100k|200k|300k} {center_scale}"
    exit 1
    ;;
esac

# ============================================================
# Fixed configuration
# ============================================================
BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257

IVF_NLIST=256
SEED=0
D_MODEL=256
N_LAYERS=3
DECODER_EPOCHS=3

BPE_ARCHIVE="bpe_wikitext103_${BPE_VOCAB_LABEL}.tar.gz"
BPE_ARCHIVE_PATH="/vqword/${BPE_ARCHIVE}"
TOKENIZER_DIR="/vqword/bpe_wikitext103_${BPE_VOCAB_LABEL}"

# This Python file must implement bilateral windows:
#   [i-hop, ..., i-1, i, i+1, ..., i+hop]
TRAIN_SCRIPT="train_vqword_reconstruct.py"
TRAIN_SCRIPT_PATH="/vqword/${TRAIN_SCRIPT}"

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

if [ ! -f "${TRAIN_SCRIPT_PATH}" ]; then
  echo "[error] Bilateral multi-HOP training script was not found:"
  echo "        ${TRAIN_SCRIPT_PATH}"
  echo
  echo "Set TRAIN_SCRIPT to the correct filename, for example:"
  echo "  TRAIN_SCRIPT=vqword_multihop_discretize.py FTP_PASS='...' $0 ${CB_SIZE} ${CENTER_SCALE}"
  exit 1
fi

# ============================================================
# Download and verify BPE tokenizer
# ============================================================
rm -f "${BPE_ARCHIVE_PATH}"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
get "${BPE_ARCHIVE}" -o "${BPE_ARCHIVE_PATH}"
bye
EOF_LFTP

rm -rf "${TOKENIZER_DIR}"
tar -xzf "${BPE_ARCHIVE_PATH}" -C /vqword

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] tokenizer directory was not created:"
  echo "        ${TOKENIZER_DIR}"
  tar -tzf "${BPE_ARCHIVE_PATH}" | head -50
  exit 1
fi

for file in tokenizer.json tokenizer_config.json; do
  if [ ! -f "${TOKENIZER_DIR}/${file}" ]; then
    echo "[error] Missing tokenizer file:"
    echo "        ${TOKENIZER_DIR}/${file}"
    exit 1
  fi
done

python - <<PY
from transformers import AutoTokenizer

path = "${TOKENIZER_DIR}"
expected_vocab_size = ${BPE_VOCAB_SIZE}

tok = AutoTokenizer.from_pretrained(path)

print("============================================================")
print("[tokenizer verification]")
print("path:", path)
print("tok.vocab_size:", tok.vocab_size)
print("len(tok):", len(tok))
print("pad_token:", tok.pad_token)
print("pad_token_id:", tok.pad_token_id)
print("unk_token:", tok.unk_token)
print("unk_token_id:", tok.unk_token_id)
print("bos_token:", tok.bos_token)
print("bos_token_id:", tok.bos_token_id)
print("eos_token:", tok.eos_token)
print("eos_token_id:", tok.eos_token_id)

if tok.vocab_size != expected_vocab_size:
    raise ValueError(
        "BPE vocabulary mismatch: "
        f"expected={expected_vocab_size:,}, "
        f"actual={tok.vocab_size:,}"
    )

print("[check] OK")
print("============================================================")
PY
# ============================================================
# Run one bilateral discretization for the specified HOP
# ============================================================

HOP_PADDED="$(printf '%02d' "${HOP}")"

MANIFEST="/vqword/wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral${HOP_PADDED}_center${CENTER_SCALE}_vqcb${VQ_CODEBOOK_LABEL}_seed${SEED}_manifest.tsv"
printf "hop\tmodel\tdictionary\tids\n" > "${MANIFEST}"

TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP_PADDED}_center${CENTER_SCALE}_deconly_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${SEED}"

OUT="wikitext103_vqword_${TAG}.pt"
DICTIONARY="wikitext103_vqword_${TAG}_dictionary.pt"
IDS="wikitext103_vqword_${TAG}_ids.pt"

OUT_PATH="/vqword/${OUT}"
DICTIONARY_PATH="/vqword/${DICTIONARY}"
IDS_PATH="/vqword/${IDS}"

echo "============================================================"
echo "[train VQWord]"
echo "dataset              = WikiText-103"
echo "tokenizer            = ${TOKENIZER_DIR}"
echo "context              = bilateral"
echo "hop                  = ${HOP} left + ${HOP} right"
echo "context width        = $((2 * HOP + 1))"
echo "center scale         = ${CENTER_SCALE}"
echo "VQW codebook         = ${VQ_CODEBOOK_SIZE}"
echo "output               = ${OUT}"
echo "============================================================"

python "${TRAIN_SCRIPT_PATH}" \
  --dataset Salesforce/wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --text_col text \
  --tokenizer "${TOKENIZER_DIR}" \
  --max_samples 1000000 \
  --seq_len 256 \
  --hop "${HOP}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --center_scale "${CENTER_SCALE}" \
  --decoder_epochs "${DECODER_EPOCHS}" \
  --decoder_lr 1e-3 \
  --decoder_weight_decay 1e-4 \
  --decoder_eval_size 100000 \
  --ivf_nlist "${IVF_NLIST}" \
  --ivf_iters 1 \
  --ivf_batch_size 8192 \
  --global_codebook_size "${VQ_CODEBOOK_SIZE}" \
  --global_kmeans_iters 5 \
  --global_batch_size 8192 \
  --batch_size 1024 \
  --k_block 4096 \
  --seed "${SEED}" \
  --out "${OUT_PATH}"

for path in "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "[error] Expected output was not generated:"
    echo "        ${path}"
    exit 1
  fi
done

ls -lh "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}"

# Upload model and dictionary.
lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
put "${OUT_PATH}" -o "${OUT}"
put "${DICTIONARY_PATH}" -o "${DICTIONARY}"
bye
EOF_LFTP

# Split and upload the potentially large ID file.
rm -f "${IDS_PATH}.part"*
split -b 450M -d -a 3 "${IDS_PATH}" "${IDS_PATH}.part"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
mput "${IDS_PATH}.part"*
bye
EOF_LFTP

printf "%s\t%s\t%s\t%s\n" \
  "${HOP}" \
  "${OUT}" \
  "${DICTIONARY}" \
  "${IDS}" \
  >> "${MANIFEST}"

echo "============================================================"
echo "[completed hop ${HOP}]"
echo "model      = ${OUT}"
echo "dictionary = ${DICTIONARY}"
echo "IDs        = ${IDS}"
echo "============================================================"

MANIFEST_NAME="$(basename "${MANIFEST}")"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
put "${MANIFEST}" -o "${MANIFEST_NAME}"
bye
EOF_LFTP

echo "============================================================"
echo "[single HOP run completed]"
echo "HOP             = ${HOP}"
echo "context width   = $((2 * HOP + 1))"
echo "context         = bilateral"
echo "center scale    = ${CENTER_SCALE}"
echo "VQW codebook    = ${VQ_CODEBOOK_SIZE}"
echo "manifest        = ${MANIFEST}"
echo "============================================================"
