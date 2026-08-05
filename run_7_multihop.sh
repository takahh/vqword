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

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {100k} {ar_seed}"
  echo "Example: VQ_INPUT_WEIGHT=0.01 CENTER_SCALE=0 $0 100k 0"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
AR_SEED="$2"
case "${VQ_CODEBOOK_LABEL}" in
  100k) VQ_CODEBOOK_SIZE=100000 ;;
  *) echo "[error] expected 100k"; exit 1 ;;
esac

INPUT_MODE="${INPUT_MODE:-vqw}"
CONTROL_SEED="${CONTROL_SEED:-12345}"

case "${INPUT_MODE}" in
  vqw|bpe2|vq_shuffle|zero)
    ;;
  *)
    echo "[error] INPUT_MODE must be one of:"
    echo "        vqw, bpe2, vq_shuffle, zero"
    exit 1
    ;;
esac

CENTER_SCALE="${CENTER_SCALE:-0}"
TARGET_HOP="${TARGET_HOP:-10}"
HOP2=$(printf "%02d" "${TARGET_HOP}")
BPE_VOCAB_LABEL=50257
IVF_NLIST=256
DISCRETIZATION_SEED=0
MODEL_VARIANT="deconly"
DECODER_EPOCHS=3

D_MODEL=256
N_LAYERS=6
N_HEADS=8
DROPOUT=0.1
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_LEN="${MAX_LEN:-255}"
VQ_LOSS_WEIGHT="${VQ_LOSS_WEIGHT:-0.1}"

TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center${CENTER_SCALE}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"
DATA_FILE="tinystories_vqword_${TAG}_ids.pt"
CODEBOOK_FILE="wikitext103_vqword_${TAG}.pt"
DATA_PATH="/vqword/${DATA_FILE}"
CODEBOOK_PATH="/vqword/${CODEBOOK_FILE}"
AR_SCRIPT="/vqword/ar_multihop.py"


CONTROL_TAG="${INPUT_MODE}"

if [ "${INPUT_MODE}" = "vq_shuffle" ]; then
  CONTROL_TAG="${INPUT_MODE}${CONTROL_SEED}"
fi

RUN="ar_twostream_bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center${CENTER_SCALE}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_vqloss${VQ_LOSS_WEIGHT}_$(date +%Y%m%d_%H%M%S)"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

for pair in "${DATA_FILE}:${DATA_PATH}" "${CODEBOOK_FILE}:${CODEBOOK_PATH}"; do
  FILE="${pair%%:*}"
  PATH_OUT="${pair#*:}"
  if [ -s "${PATH_OUT}" ]; then
    echo "[reuse] ${PATH_OUT}"
    continue
  fi
  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${FILE}" -o "${PATH_OUT}"
bye
EOF
done

for p in "${DATA_PATH}" "${CODEBOOK_PATH}" "${AR_SCRIPT}"; do
  [ -s "${p}" ] || { echo "[error] missing ${p}"; exit 1; }
done

python - "${DATA_PATH}" "${CODEBOOK_PATH}" "${TARGET_HOP}" "${CENTER_SCALE}" <<'PY'
import sys, torch

data_path, codebook_path, expected_hop, expected_scale = sys.argv[1:]
expected_hop = int(expected_hop)
expected_scale = float(expected_scale)
data = torch.load(data_path, map_location="cpu", weights_only=False)
cb = torch.load(codebook_path, map_location="cpu", weights_only=False)

for key in ("samples", "token_ids_flat", "vq_ids_flat"):
    if key not in data:
        raise KeyError(f"data missing {key}")
if "global_centers" not in cb:
    raise KeyError("codebook missing global_centers")

hop_data = int(data.get("hop", -1))
hop_cb = int(cb.get("args", {}).get("hop", -1))
if hop_data != expected_hop or hop_cb != expected_hop:
    raise ValueError(f"hop mismatch: data={hop_data}, cb={hop_cb}, expected={expected_hop}")

centers = cb["global_centers"]
vq = data["vq_ids_flat"].long().reshape(-1)
tok = data["token_ids_flat"].long().reshape(-1)
if tok.numel() != vq.numel():
    raise ValueError("token/VQ length mismatch")
if centers.shape[0] != 100000:
    raise ValueError(f"expected 100000 centers, got {centers.shape[0]}")
if int(vq.min()) < 0 or int(vq.max()) >= centers.shape[0]:
    raise ValueError("VQ IDs out of range")
print("[verification OK]")
print("tokens:", f"{tok.numel():,}")
print("samples:", f"{len(data['samples']):,}")
print("centers:", tuple(centers.shape))
print("hop:", expected_hop)
print("center_scale requested:", expected_scale)
PY

echo "============================================================"
echo "[start BPE/SC0 two-stream autoregressive training]"
echo "BPE stream = BPE[t] -> h_bpe"
echo "VQ stream  = VQW[t] -> h_vq -> VQW[t+1]"
echo "fusion     = CAT(h_bpe, h_vq) -> BPE[t+1]"
echo "VQ loss weight = ${VQ_LOSS_WEIGHT}"
echo "hop        = bilateral ${TARGET_HOP}"
echo "center     = ${CENTER_SCALE}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${AR_SCRIPT}" \
  --data "${DATA_PATH}" \
  --codebook "${CODEBOOK_PATH}" \
  --input_mode "${INPUT_MODE}" \
  --control_seed "${CONTROL_SEED}" \
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
  --vq_loss_weight "${VQ_LOSS_WEIGHT}" \
  --out "${FINAL_PATH}" \
  2>&1 | tee "${LOG_PATH}"

for p in "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"; do
  [ -s "${p}" ] || { echo "[error] missing output ${p}"; exit 1; }
done

grep -E "\[epoch [0-9]+\]|\[save best\]|\[save final\]" "${LOG_PATH}" || true

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
cd vqword_logs
put "${BEST_PATH}" -o "${RUN}_best.pt"
put "${FINAL_PATH}" -o "${RUN}.pt"
put "${LOG_PATH}" -o "${RUN}.log"
bye
EOF

echo "[completed] ${RUN}"
