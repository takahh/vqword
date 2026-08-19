#!/usr/bin/env bash
set -euo pipefail

# WikiText-103: shared-space, recurrent-HOP BPE-local VQWord generation
#
# One tied GNN cell is applied repeatedly. Intermediate states for
# HOP=MIN_HOP..MAX_HOP are discretized with independent BPE-local codebooks.
# HOP is not embedded in the token.
#
# Usage:
#   FTP_PASS='...' bash run_vqword_bpe_local_sharedhop_wikitext.sh 5 0.0 10
#
# Arguments:
#   $1: maximum local clusters per BPE
#   $2: center scale
#   $3: maximum bilateral HOP
#
# Optional environment variables:
#   MIN_HOP=1                    first saved HOP (set 0 to include HOP0)
#   CLUSTER_SAMPLES_PER_BPE=256 physical positions sampled per BPE for KMeans

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 {local_clusters} {center_scale} {max_hop}"
  echo "Example: FTP_PASS='your-password' $0 5 0.0 10"
  exit 1
fi

LOCAL_CLUSTERS="$1"
CENTER_SCALE_RAW="$2"
MAX_HOP="$3"
MIN_HOP="${MIN_HOP:-1}"
CLUSTER_SAMPLES_PER_BPE="${CLUSTER_SAMPLES_PER_BPE:-256}"

for item in \
  "LOCAL_CLUSTERS:${LOCAL_CLUSTERS}" \
  "MIN_HOP:${MIN_HOP}" \
  "MAX_HOP:${MAX_HOP}" \
  "CLUSTER_SAMPLES_PER_BPE:${CLUSTER_SAMPLES_PER_BPE}"; do
  name="${item%%:*}"
  value="${item#*:}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "[error] ${name} must be a non-negative integer: ${value}"
    exit 1
  fi
done

if (( LOCAL_CLUSTERS < 1 || LOCAL_CLUSTERS > 32767 )); then
  echo "[error] LOCAL_CLUSTERS must be between 1 and 32767"
  exit 1
fi
if (( MAX_HOP > 100 )); then
  echo "[error] MAX_HOP must be between 0 and 100: ${MAX_HOP}"
  exit 1
fi
if (( MIN_HOP > MAX_HOP )); then
  echo "[error] MIN_HOP must be <= MAX_HOP: ${MIN_HOP} > ${MAX_HOP}"
  exit 1
fi
if (( CLUSTER_SAMPLES_PER_BPE < 1 )); then
  echo "[error] CLUSTER_SAMPLES_PER_BPE must be at least 1"
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

# Exact shared recurrent HOP semantics require one tied cell application
# per HOP. The training script rejects any other value.
N_LAYERS=1

BPE_ARCHIVE="bpe_wikitext103_${BPE_VOCAB_LABEL}.tar.gz"
BPE_ARCHIVE_PATH="/vqword/${BPE_ARCHIVE}"
TOKENIZER_DIR="/vqword/bpe_wikitext103_${BPE_VOCAB_LABEL}"

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

MIN_HOP_PADDED="$(printf '%02d' "${MIN_HOP}")"
MAX_HOP_PADDED="$(printf '%02d' "${MAX_HOP}")"
TAG="bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop${MIN_HOP_PADDED}to${MAX_HOP_PADDED}_center${CENTER_SCALE}_localbpe${LOCAL_CLUSTERS}_seed${SEED}"

OUT="wikitext103_vqword_${TAG}.pt"
DICTIONARY="wikitext103_vqword_${TAG}_dictionary.pt"
IDS="wikitext103_vqword_${TAG}_ids.pt"
MANIFEST_NAME="wikitext103_vqword_${TAG}_manifest.tsv"

OUT_PATH="/vqword/${OUT}"
DICTIONARY_PATH="/vqword/${DICTIONARY}"
IDS_PATH="/vqword/${IDS}"
MANIFEST_PATH="/vqword/${MANIFEST_NAME}"

echo "============================================================"
echo "[train] shared recurrent HOP VQWord"
echo "HOP range          = ${MIN_HOP}..${MAX_HOP}"
echo "tied GNN           = yes"
echo "shared codebook    = no (independent per HOP)"
echo "HOP embedding      = no"
echo "center scale       = ${CENTER_SCALE}"
echo "local clusters     = ${LOCAL_CLUSTERS}"
echo "cluster samples/BPE= ${CLUSTER_SAMPLES_PER_BPE}"
echo "output             = ${OUT_PATH}"
echo "============================================================"

python3 "${TRAIN_SCRIPT_PATH}" \
  --dataset Salesforce/wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --text_col text \
  --tokenizer "${TOKENIZER_DIR}" \
  --max_samples 1000000 \
  --seq_len 256 \
  --all_hops \
  --min_hop "${MIN_HOP}" \
  --max_hop "${MAX_HOP}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --center_scale "${CENTER_SCALE}" \
  --local_clusters "${LOCAL_CLUSTERS}" \
  --cluster_samples_per_bpe "${CLUSTER_SAMPLES_PER_BPE}" \
  --batch_size 1024 \
  --seed "${SEED}" \
  --out "${OUT_PATH}"

for path in "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}"; do
  if [ ! -s "${path}" ]; then
    echo "[error] expected output was not generated: ${path}"
    exit 1
  fi
done

# Verify tied GNN plus independent per-HOP codebooks.
python3 - "${OUT_PATH}" "${IDS_PATH}" "${MIN_HOP}" "${MAX_HOP}" <<'PY'
import sys
import torch

model_path, ids_path, min_hop, max_hop = sys.argv[1:]
expected_hops = list(range(int(min_hop), int(max_hop) + 1))
model = torch.load(model_path, map_location="cpu", weights_only=False)
ids = torch.load(ids_path, map_location="cpu", weights_only=False)

for name, obj in (("model", model), ("ids", ids)):
    if obj.get("hops") != expected_hops:
        raise ValueError(
            f"{name} HOP mismatch: {obj.get('hops')} != {expected_hops}"
        )
    if obj.get("shared_gnn_across_hops") is not True:
        raise ValueError(f"{name} does not declare a shared GNN")
    if obj.get("shared_codebook_across_hops") is not False:
        raise ValueError(f"{name} does not declare per-HOP codebooks")
    if obj.get("hop_embedding") is not False:
        raise ValueError(f"{name} unexpectedly uses a HOP embedding")

matrix = ids["local_vq_ids_by_hop"]
if matrix.ndim != 2 or matrix.size(0) != len(expected_hops):
    raise ValueError(
        f"invalid local_vq_ids_by_hop shape: {tuple(matrix.shape)}"
    )
if matrix.size(1) != ids["bpe_ids"].numel():
    raise ValueError("BPE/VQ position count mismatch")
if ids.get("k_by_bpe_by_hop", torch.empty(0)).shape[0] != len(expected_hops):
    raise ValueError("missing per-HOP k_by_bpe metadata")
if not torch.equal(ids["local_vq_ids"], matrix[-1]):
    raise ValueError("maximum-HOP compatibility alias mismatch")

print(
    f"[verification OK] hops={expected_hops} "
    f"shape={tuple(matrix.shape)} positions={matrix.size(1):,}"
)
PY

printf "min_hop\tmax_hop\tlocal_clusters\ttied_gnn\tshared_codebook\thop_embedding\tmodel\tdictionary\tids\n" > "${MANIFEST_PATH}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${MIN_HOP}" "${MAX_HOP}" "${LOCAL_CLUSTERS}" \
  "true" "false" "false" \
  "${OUT}" "${DICTIONARY}" "${IDS}" >> "${MANIFEST_PATH}"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF_LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes
put "${OUT_PATH}" -o "${OUT}"
put "${DICTIONARY_PATH}" -o "${DICTIONARY}"
put "${MANIFEST_PATH}" -o "${MANIFEST_NAME}"
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

ls -lh "${OUT_PATH}" "${DICTIONARY_PATH}" "${IDS_PATH}" "${MANIFEST_PATH}"
echo "[completed] ${TAG}"
