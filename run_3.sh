#!/usr/bin/env bash
set -euo pipefail

# WikiText-103: BPE-local VQWord generation
# Usage: FTP_PASS='...' bash run_vqword_bpe_local_wikitext.sh 5 0.0 10
#   $1: maximum local clusters per BPE
#   $2: center scale
#   $3: bilateral hop

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 {local_clusters} {center_scale} {hop}"
  echo "Example: FTP_PASS='your-password' $0 5 0.0 10"
  exit 1
fi

LOCAL_CLUSTERS="$1"
CENTER_SCALE_RAW="$2"
HOP="$3"

for item in "LOCAL_CLUSTERS:${LOCAL_CLUSTERS}" "HOP:${HOP}"; do
  name="${item%%:*}"
  value="${item#*:}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[error] ${name} must be an integer: ${value}"
    exit 1
  fi
done

if (( LOCAL_CLUSTERS < 1 || LOCAL_CLUSTERS > 32767 )); then
  echo "[error] LOCAL_CLUSTERS must be between 1 and 32767"
  exit 1
fi
if (( HOP < 0 || HOP > 100 )); then
  echo "[error] HOP must be between 0 and 100: ${HOP}"
  exit 1
fi

CENTER_SCALE="$(python3 -c '
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"[error] invalid center_scale: {sys.argv[1]}")
print(f"{value:g}")
' "${CENTER_SCALE_RAW}")"

apt update
apt install -y lftp
pip install torch datasets transformers scikit-learn tqdm numpy

cd /
if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi
cd /vqword
git pull

BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257
SEED=0
D_MODEL=256
N_LAYERS=3

BPE_ARCHIVE="bpe_wikitext103_${BPE_VOCAB_LABEL}.tar.gz"
BPE_ARCHIVE_PATH="/vqword/${BPE_ARCHIVE}"
TOKENIZER_DIR="/vqword/bpe_wikitext103_${BPE_VOCAB_LABEL}"

# Commit vqword_bpe_local.py to /vqword before running this script.
TRAIN_SCRIPT="train_vqword_reconstruct.py"
TRAIN_SCRIPT_PATH="/vqword/${TRAIN_SCRIPT}"

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

if [ ! -f "${TRAIN_SCRIPT_PATH}" ]; then
  echo "[error] training script was not found: ${TRAIN_SCRIPT_PATH}"
  exit 1
fi

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

for file in tokenizer.json tokenizer_config.json; do
  if [ ! -f "${TOKENIZER_DIR}/${file}" ]; then
    echo "[error] missing tokenizer file: ${TOKENIZER_DIR}/${file}"
    exit 1
  fi
done

python3 - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${TOKENIZER_DIR}")
print("[tokenizer] vocab_size=", tok.vocab_size, "len=", len(tok))
if tok.vocab_size != ${BPE_VOCAB_SIZE}:
    raise ValueError(f"BPE vocabulary mismatch: {tok.vocab_size}")
PY

HOP_PADDED="$(printf '%02d' "${HOP}")"
TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP_PADDED}_center${CENTER_SCALE}_localbpe${LOCAL_CLUSTERS}_seed${SEED}"
OUT="wikitext103_vqword_${TAG}.pt"
DICTIONARY="wikitext103_vqword_${TAG}_dictionary.pt"
IDS="wikitext103_vqword_${TAG}_ids.pt"
OUT_PATH="/vqword/${OUT}"
DICTIONARY_PATH="/vqword/${DICTIONARY}"
IDS_PATH="/vqword/${IDS}"
MANIFEST="/vqword/wikitext103_vqword_${TAG}_manifest.tsv"

echo "[train] hop=${HOP} center_scale=${CENTER_SCALE} local_clusters=${LOCAL_CLUSTERS}"
python3 "${TRAIN_SCRIPT_PATH}" \
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
  --local_clusters "${LOCAL_CLUSTERS}" \
  --batch_size 1024 \
  --seed "${SEED}" \
  --out "${OUT_PATH}"

for path in "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}"; do
  if [ ! -f "${path}" ]; then
    echo "[error] expected output was not generated: ${path}"
    exit 1
  fi
done

printf "hop\tlocal_clusters\tmodel\tdictionary\tids\n" > "${MANIFEST}"
printf "%s\t%s\t%s\t%s\t%s\n" \
  "${HOP}" "${LOCAL_CLUSTERS}" "${OUT}" "${DICTIONARY}" "${IDS}" >> "${MANIFEST}"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
put "${OUT_PATH}" -o "${OUT}"
put "${DICTIONARY_PATH}" -o "${DICTIONARY}"
put "${MANIFEST}" -o "$(basename "${MANIFEST}")"
bye
EOF_LFTP

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

ls -lh "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}" "${MANIFEST}"
echo "[completed] ${TAG}"
