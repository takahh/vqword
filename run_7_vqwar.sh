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

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {local_bpe_direct|global_vqwar}"
  echo "  $0 local_bpe_direct"
  echo "  $0 global_vqwar"
  exit 1
fi

AR_MODE="$1"
case "${AR_MODE}" in
  local_bpe_direct)
    DATA_FILE="tinystories_vqword_bpe50257_tiedgnn_separatehop10_center0_localbpe5_seed0_ids.pt"
    CODEBOOK_FILE="wikitext103_vqword_bpe50257_tiedgnn_separatehop01to10_center0_localbpe5_seed0.pt"
    VQ_LABEL="localbpe5_center0"
    ;;
  global_vqwar)
    DATA_FILE="tinystories_vqword_bpe50257_tiedgnn_separatehop10_center0_localbpe5_seed0_ids.pt"
    CODEBOOK_FILE="wikitext103_vqword_bpe50257_tiedgnn_separatehop01to10_center0_localbpe5_seed0.pt"
    VQ_LABEL="pairglobal_localbpe5_center0"
    ;;
  *)
    echo "[error] mode must be local_bpe_direct or global_vqwar"
    exit 1
    ;;
esac

AR_SEED="${AR_SEED:-0}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_LEN="${MAX_LEN:-255}"
VQ_GAP="${VQ_GAP:-11}"
LOCAL_BPE_TOKENS="${LOCAL_BPE_TOKENS:-10}"
MIXTURE_TOPK="${MIXTURE_TOPK:-32}"
VQW_ALPHA_INIT="${VQW_ALPHA_INIT:-0.5}"
AR_SCRIPT="/vqword/ar_vqwar.py"
DISABLE_VQW="${DISABLE_VQW:-0}"

EXTRA_ARGS=()
if [ "${DISABLE_VQW}" = "1" ]; then
  EXTRA_ARGS+=(--disable_vqw)
  VQ_LABEL="${VQ_LABEL}_matched_residual_bpebaseline"
fi

if [ "${VQ_GAP}" -ne 11 ] || [ "${LOCAL_BPE_TOKENS}" -ne 10 ]; then
  echo "[error] bilateral HOP10 requires VQ_GAP=11 and LOCAL_BPE_TOKENS=10"
  exit 1
fi

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
  test -s "${local_path}" || { echo "[error] download failed"; exit 1; }
}

download_file "${CODEBOOK_FILE}" "/vqword/${CODEBOOK_FILE}"
download_file "${DATA_FILE}" "/vqword/${DATA_FILE}"

python - "${AR_MODE}" "/vqword/${DATA_FILE}" "/vqword/${CODEBOOK_FILE}" <<'PY'
import sys
import torch

mode, data_path, codebook_path = sys.argv[1:]

data = torch.load(data_path, map_location="cpu", weights_only=False)
cb = torch.load(codebook_path, map_location="cpu", weights_only=False)

hop_data = int(data.get("hop", data.get("args", {}).get("hop", -1)))
hop_cb = int(cb.get("hop", cb.get("args", {}).get("hop", -1)))

if hop_data != 10 or hop_cb != 10:
    raise ValueError(f"HOP mismatch: data={hop_data}, codebook={hop_cb}")

if cb.get("partition_type") != "bpe_local_kmeans":
    raise ValueError(
        f"{mode} requires bpe_local_kmeans, "
        f"got {cb.get('partition_type')!r}"
    )

tokens = data["token_ids_flat"].long().reshape(-1)
local_vq = data["vq_ids_flat"].long().reshape(-1)

if tokens.numel() != local_vq.numel():
    raise ValueError("token/VQ length mismatch")

local_vq_size = int(
    data.get("vq_vocab_size", cb.get("max_local_clusters", -1))
)
if local_vq_size < 1:
    raise ValueError(f"invalid local VQ vocabulary size: {local_vq_size}")

if int(local_vq.min()) < 0 or int(local_vq.max()) >= local_vq_size:
    raise ValueError(
        f"local VQ IDs out of range 0..{local_vq_size - 1}"
    )

message = (
    f"[verification OK] mode={mode} "
    f"tokens={tokens.numel():,} "
    f"samples={len(data['samples']):,} "
    f"local_vq_vocab={local_vq_size} "
    f"local_vq_range={int(local_vq.min())}..{int(local_vq.max())}"
)

if mode == "global_vqwar":
    pair_keys = tokens * local_vq_size + local_vq
    pair_global_vocab = int(torch.unique(pair_keys).numel())
    message += f" pair_global_vocab={pair_global_vocab:,}"

print(message)
PY

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN="${AR_MODE}_hop10_gap11_localbpe10_${VQ_LABEL}_arseed${AR_SEED}_${TIMESTAMP}"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

echo "============================================================"
echo "mode              = ${AR_MODE}"
echo "data              = /vqword/${DATA_FILE}"
echo "codebook          = /vqword/${CODEBOOK_FILE}"
echo "alignment         = distant t-11 + recent BPE t-10..t-1"
echo "epochs/batch/lr   = ${EPOCHS}/${BATCH_SIZE}/${LR}"
echo "VQW alpha init    = ${VQW_ALPHA_INIT}"
echo "disable VQW       = ${DISABLE_VQW}"
echo "run               = ${RUN}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
  --mode "${AR_MODE}" \
  --data "/vqword/${DATA_FILE}" \
  --codebook "/vqword/${CODEBOOK_FILE}" \
  --gap "${VQ_GAP}" \
  --local_bpe_tokens "${LOCAL_BPE_TOKENS}" \
  --mixture_topk "${MIXTURE_TOPK}" \
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
  --vqw_alpha_init "${VQW_ALPHA_INIT}" \
  --out "${FINAL_PATH}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

for path_to_check in "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"; do
  test -s "${path_to_check}" || { echo "[error] missing ${path_to_check}"; exit 1; }
done

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

echo "[completed] ${RUN}"
