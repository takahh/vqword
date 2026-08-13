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
  echo "Usage: $0 {10k|25k|50k|100k} {center_scale} 10"
  echo
  echo "Example:"
  echo "  USE_VQW=1 VQW_INIT_SCALE=1 $0 10k 1 10"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
CENTER_SCALE="$2"
DISTANT_HOP="$3"

# 距離1..10にはHOP0..9、距離11以上にはHOP10を使う。
# 旧runnerとの3引数互換を保つため、第3引数は10だけを受け付ける。
if [ "${DISTANT_HOP}" != "10" ]; then
  echo "[error] this multi-HOP runner requires the third argument to be 10"
  exit 1
fi

AR_SEED="${AR_SEED:-0}"
USE_VQW="${USE_VQW:-1}"
VQW_INIT_SCALE="${VQW_INIT_SCALE:-0.1}"

case "${VQ_CODEBOOK_LABEL}" in
  10k)  VQ_CODEBOOK_SIZE=10000 ;;
  25k)  VQ_CODEBOOK_SIZE=25000 ;;
  50k)  VQ_CODEBOOK_SIZE=50000 ;;
  100k) VQ_CODEBOOK_SIZE=100000 ;;
  *)
    echo "[error] expected codebook label: 10k, 25k, 50k, or 100k"
    exit 1
    ;;
esac

if ! [[ "${CENTER_SCALE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "[error] center_scale must be a non-negative number: ${CENTER_SCALE}"
  exit 1
fi
if ! [[ "${AR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] AR_SEED must be a non-negative integer: ${AR_SEED}"
  exit 1
fi
case "${USE_VQW}" in
  0|1) ;;
  *)
    echo "[error] USE_VQW must be 0 or 1"
    exit 1
    ;;
esac

case "${CENTER_SCALE}" in
  1.0) CENTER_LABEL="1" ;;
  0.30) CENTER_LABEL="0.3" ;;
  *) CENTER_LABEL="${CENTER_SCALE}" ;;
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

AR_SCRIPT="/vqword/ar_multihop.py"

COMMON_SUFFIX="center${CENTER_LABEL}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"
DATA_PATTERN="/vqword/wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral{hop:02d}_${COMMON_SUFFIX}_ids.pt"
CODEBOOK_PATTERN="/vqword/wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral{hop:02d}_${COMMON_SUFFIX}.pt"

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

  if [ ! -s "${local_path}" ]; then
    echo "[error] download failed: ${local_path}"
    exit 1
  fi
}

for HOP_INDEX in $(seq 0 10); do
  HOP2=$(printf "%02d" "${HOP_INDEX}")
  PREFIX="wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_${COMMON_SUFFIX}"
  DATA_FILE="${PREFIX}_ids.pt"
  CODEBOOK_FILE="${PREFIX}.pt"

  download_file "${CODEBOOK_FILE}" "/vqword/${CODEBOOK_FILE}"

  if [ ! -s "/vqword/${DATA_FILE}" ]; then
    download_file "${DATA_FILE}.part000" "/vqword/${DATA_FILE}.part000"
    download_file "${DATA_FILE}.part001" "/vqword/${DATA_FILE}.part001"

    echo "[combine] ${DATA_FILE}"
    cat "/vqword/${DATA_FILE}.part000" \
        "/vqword/${DATA_FILE}.part001" \
        > "/vqword/${DATA_FILE}.tmp"
    mv "/vqword/${DATA_FILE}.tmp" "/vqword/${DATA_FILE}"
  else
    echo "[reuse] /vqword/${DATA_FILE}"
  fi
done

if [ ! -s "${AR_SCRIPT}" ]; then
  echo "[error] missing AR script: ${AR_SCRIPT}"
  exit 1
fi

# 全HOPのdata/codebook、token配列、sample境界、VQ範囲を事前検証する。
python - \
    "${DATA_PATTERN}" \
    "${CODEBOOK_PATTERN}" \
    "${VQ_CODEBOOK_SIZE}" \
    "${CENTER_SCALE}" <<'PY'
import sys
import torch

data_pattern = sys.argv[1]
codebook_pattern = sys.argv[2]
expected_vq_size = int(sys.argv[3])
expected_center_scale = float(sys.argv[4])

reference_tokens = None
reference_bounds = None

for hop in range(11):
    data_path = data_pattern.format(hop=hop)
    codebook_path = codebook_pattern.format(hop=hop)
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)

    for key in ("samples", "token_ids_flat", "vq_ids_flat"):
        if key not in data:
            raise KeyError(f"HOP{hop} data missing key: {key}")
    if "global_centers" not in codebook:
        raise KeyError(f"HOP{hop} codebook missing global_centers")

    data_hop = int(data.get("hop", -1))
    codebook_hop = int(codebook.get("args", {}).get("hop", -1))
    if data_hop != hop or codebook_hop != hop:
        raise ValueError(
            f"HOP mismatch at HOP{hop}: data={data_hop}, codebook={codebook_hop}"
        )

    tokens = data["token_ids_flat"].long().reshape(-1)
    vq_ids = data["vq_ids_flat"].long().reshape(-1)
    centers = codebook["global_centers"]
    bounds = [(int(s["start"]), int(s["end"])) for s in data["samples"]]

    if tokens.numel() != vq_ids.numel():
        raise ValueError(f"HOP{hop} token/VQ length mismatch")
    if int(centers.shape[0]) != expected_vq_size:
        raise ValueError(
            f"HOP{hop} VQ size mismatch: expected={expected_vq_size}, "
            f"actual={centers.shape[0]}"
        )
    vq_min = int(vq_ids.min())
    vq_max = int(vq_ids.max())
    if vq_min < 0 or vq_max >= int(centers.shape[0]):
        raise ValueError(f"HOP{hop} VQ ID out of range: {vq_min}..{vq_max}")

    actual_scale = float(codebook.get("args", {}).get("center_scale", -1.0))
    if abs(actual_scale - expected_center_scale) > 1e-8:
        raise ValueError(
            f"HOP{hop} center scale mismatch: "
            f"expected={expected_center_scale}, actual={actual_scale}"
        )

    if reference_tokens is None:
        reference_tokens = tokens
        reference_bounds = bounds
    else:
        if not torch.equal(tokens, reference_tokens):
            raise ValueError(f"HOP{hop} token_ids_flat differs from HOP0")
        if bounds != reference_bounds:
            raise ValueError(f"HOP{hop} sample boundaries differ from HOP0")

    print(
        f"[verification OK] HOP{hop:02d} "
        f"tokens={tokens.numel():,} samples={len(bounds):,} "
        f"used_vq={torch.unique(vq_ids).numel():,} centers={tuple(centers.shape)}"
    )
PY

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN="ar_inputcat_bpeonly_multihop00to10_usevqw${USE_VQW}_bpe${BPE_VOCAB_LABEL}_center${CENTER_LABEL}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_${TIMESTAMP}"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

echo "============================================================"
echo "[start multi-HOP BPE-only AR training]"
echo "VQW distance mapping  = 1:HOP0 ... 10:HOP9, 11+:HOP10"
echo "data pattern          = ${DATA_PATTERN}"
echo "codebook pattern      = ${CODEBOOK_PATTERN}"
echo "use VQW               = ${USE_VQW}"
echo "VQW initial scale     = ${VQW_INIT_SCALE}"
echo "center scale          = ${CENTER_SCALE}"
echo "codebook size         = ${VQ_CODEBOOK_SIZE}"
echo "AR seed               = ${AR_SEED}"
echo "run                   = ${RUN}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
    --hop_data_pattern "${DATA_PATTERN}" \
    --hop_codebook_pattern "${CODEBOOK_PATTERN}" \
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
  if [ ! -s "${PATH_TO_CHECK}" ]; then
    echo "[error] missing output: ${PATH_TO_CHECK}"
    exit 1
  fi
done

grep -E '^\[epoch [0-9]+\]|^\[save best\]|^\[save final\]' \
  "${LOG_PATH}" || true

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
