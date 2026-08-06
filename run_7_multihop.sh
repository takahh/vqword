#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y lftp
pip install torch tqdm numpy

cd /
if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi
cd /vqword
git pull

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 {100k} {ar_seed} {hop}"
  echo
  echo "Example:"
  echo "  CENTER_SCALE=0 USE_VQW=1 VQW_INIT_SCALE=0.1 $0 100k 0 50"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
AR_SEED="$2"
HOP="$3"

case "${VQ_CODEBOOK_LABEL}" in
  100k) VQ_CODEBOOK_SIZE=100000 ;;
  *) echo "[error] expected codebook label: 100k"; exit 1 ;;
esac

if ! [[ "${AR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] ar_seed must be a non-negative integer: ${AR_SEED}"
  exit 1
fi
if ! [[ "${HOP}" =~ ^[0-9]+$ ]]; then
  echo "[error] hop must be a non-negative integer: ${HOP}"
  exit 1
fi

CENTER_SCALE="${CENTER_SCALE:-0}"
USE_VQW="${USE_VQW:-1}"
VQW_INIT_SCALE="${VQW_INIT_SCALE:-0.1}"
case "${USE_VQW}" in
  0|1) ;;
  *) echo "[error] USE_VQW must be 0 or 1"; exit 1 ;;
esac

BPE_VOCAB_LABEL=50257
IVF_NLIST=256
DISCRETIZATION_SEED=0
MODEL_VARIANT="deconly"
DECODER_EPOCHS=3

D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_LEN="${MAX_LEN:-255}"

AR_SCRIPT="/vqword/ar_singlehop.py"
HOP2=$(printf "%02d" "${HOP}")
TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center${CENTER_SCALE}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"
DATA_FILE="tinystories_vqword_${TAG}_ids.pt"
CODEBOOK_FILE="wikitext103_vqword_${TAG}.pt"
DATA_PATH="/vqword/${DATA_FILE}"
CODEBOOK_PATH="/vqword/${CODEBOOK_FILE}"

RUN="ar_inputcat_bpeonly_singlehop${HOP2}_usevqw${USE_VQW}_bpe${BPE_VOCAB_LABEL}_center${CENTER_SCALE}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_$(date +%Y%m%d_%H%M%S)"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

download_file() {
  local remote_file="$1"
  local local_path="$2"
  if [ -s "${local_path}" ]; then
    echo "[reuse] ${local_path}"
    return
  fi
  echo "[download] ${remote_file}"
  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${remote_file}" -o "${local_path}"
bye
LFTP
  [ -s "${local_path}" ] || { echo "[error] download failed: ${local_path}"; exit 1; }
}

download_file "${DATA_FILE}" "${DATA_PATH}"
download_file "${CODEBOOK_FILE}" "${CODEBOOK_PATH}"

[ -s "${AR_SCRIPT}" ] || { echo "[error] missing AR script: ${AR_SCRIPT}"; exit 1; }

python - "${DATA_PATH}" "${CODEBOOK_PATH}" "${HOP}" "${VQ_CODEBOOK_SIZE}" "${CENTER_SCALE}" <<'PY'
import sys
import torch

data_path, codebook_path = sys.argv[1], sys.argv[2]
expected_hop = int(sys.argv[3])
expected_vq_size = int(sys.argv[4])
expected_center_scale = float(sys.argv[5])

data = torch.load(data_path, map_location="cpu", weights_only=False)
codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)

for key in ("samples", "token_ids_flat", "vq_ids_flat"):
    if key not in data:
        raise KeyError(f"data missing key: {key}")
if "global_centers" not in codebook:
    raise KeyError("codebook missing global_centers")

data_hop = int(data.get("hop", -1))
codebook_hop = int(codebook.get("args", {}).get("hop", -1))
if data_hop != expected_hop or codebook_hop != expected_hop:
    raise ValueError(
        f"HOP mismatch: expected={expected_hop}, data={data_hop}, codebook={codebook_hop}"
    )

centers = codebook["global_centers"]
vq_ids = data["vq_ids_flat"].long().reshape(-1)
tokens = data["token_ids_flat"].long().reshape(-1)
if tokens.numel() != vq_ids.numel():
    raise ValueError(f"token/VQ length mismatch: {tokens.numel()} vs {vq_ids.numel()}")
if int(centers.shape[0]) != expected_vq_size:
    raise ValueError(f"VQ size mismatch: expected={expected_vq_size}, actual={centers.shape[0]}")
if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= int(centers.shape[0]):
    raise ValueError(f"VQ ID out of range: {int(vq_ids.min())}..{int(vq_ids.max())}")

actual_scale = float(codebook.get("args", {}).get("center_scale", -1.0))
if abs(actual_scale - expected_center_scale) > 1e-8:
    raise ValueError(f"center scale mismatch: expected={expected_center_scale}, actual={actual_scale}")

print("[verification OK]")
print("hop:", expected_hop)
print("tokens:", f"{tokens.numel():,}")
print("samples:", f"{len(data['samples']):,}")
print("used_vq:", f"{torch.unique(vq_ids).numel():,}")
print("centers:", tuple(centers.shape))
PY

echo "============================================================"
echo "[start single-HOP BPE-only AR training]"
echo "hop                   = ${HOP}"
echo "VQ distance threshold = ${HOP}"
echo "use VQW               = ${USE_VQW}"
echo "VQW initial scale     = ${VQW_INIT_SCALE}"
echo "center scale          = ${CENTER_SCALE}"
echo "AR seed               = ${AR_SEED}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
  --hop_data "${DATA_PATH}" \
  --hop_codebook "${CODEBOOK_PATH}" \
  --hop "${HOP}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --n_heads "${N_HEADS}" \
  --dropout "${DROPOUT}" \
  --max_len "${MAX_LEN}" \
  --seed "${AR_SEED}" \
  --use_vqw "${USE_VQW}" \
  --vqw_init_scale "${VQW_INIT_SCALE}" \
  --out "${FINAL_PATH}" \
  2>&1 | tee "${LOG_PATH}"

for PATH_TO_CHECK in "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"; do
  [ -s "${PATH_TO_CHECK}" ] || { echo "[error] missing output: ${PATH_TO_CHECK}"; exit 1; }
done

grep -E "\[epoch [0-9]+\]|\[save best\]|\[save final\]" "${LOG_PATH}" || true

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<LFTP
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
cd vqword_logs
put "${BEST_PATH}" -o "${RUN}_best.pt"
put "${FINAL_PATH}" -o "${RUN}.pt"
put "${LOG_PATH}" -o "${RUN}.log"
bye
LFTP

echo "============================================================"
echo "[completed]"
echo "run   = ${RUN}"
echo "best  = ${BEST_PATH}"
echo "final = ${FINAL_PATH}"
echo "log   = ${LOG_PATH}"
echo "============================================================"
