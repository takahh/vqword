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
  exit 1
fi

AR_MODE="$1"
case "${AR_MODE}" in
  local_bpe_direct)
    VQ_LABEL="samidare_localbpe5_center0"
    ;;
  global_vqwar)
    VQ_LABEL="samidare_pairglobal_localbpe5_center0"
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
  echo "[error] HOP1..10 samidare requires VQ_GAP=11 and LOCAL_BPE_TOKENS=10"
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

  test -s "${local_path}" || {
    echo "[error] download failed: ${remote_file}"
    exit 1
  }
}

# ============================================================
# TinyStories HOP1..10 inputs
# ============================================================

DATA_PREFIX="tinystories_vqword_bpe50257_tiedgnn_separatehop"
DATA_SUFFIX="_center0_localbpe5_seed0_ids.pt"

DATA_HOPS=()

for HOP in $(seq 1 10); do
  HOP2=$(printf "%02d" "${HOP}")
  DATA_FILE="${DATA_PREFIX}${HOP2}${DATA_SUFFIX}"
  DATA_PATH="/vqword/${DATA_FILE}"

  download_file "${DATA_FILE}" "${DATA_PATH}"
  DATA_HOPS+=("${DATA_PATH}")
done

# ============================================================
# Shared model: rebuild once from WikiText HOP1 part if needed.
# ============================================================

CKPT_PREFIX="wikitext103_vqword_bpe50257_tiedgnn_separatehop01to10_center0_localbpe5_seed0"
HOP1_FILE="${CKPT_PREFIX}_hop_001_ids.pt"
HOP1_PATH="/vqword/${HOP1_FILE}"

CODEBOOK_FILE="${CKPT_PREFIX}_shared_model.pt"
CODEBOOK_PATH="/vqword/${CODEBOOK_FILE}"

if [ ! -s "${CODEBOOK_PATH}" ]; then
  download_file "${HOP1_FILE}" "${HOP1_PATH}"

  if [ -f /vqword/rebuild_shared_model_from_hop.py ]; then
    REBUILD_SCRIPT="/vqword/rebuild_shared_model_from_hop.py"
  elif [ -f /vqword/rebuild_shared_gnn.py ]; then
    REBUILD_SCRIPT="/vqword/rebuild_shared_gnn.py"
  else
    echo "[error] rebuild helper not found"
    exit 1
  fi

  echo "============================================================"
  echo "[rebuild] shared model from WikiText HOP1"
  echo "source            = ${HOP1_PATH}"
  echo "output            = ${CODEBOOK_PATH}"
  echo "============================================================"

  python "${REBUILD_SCRIPT}" \
    --hop_file "${HOP1_PATH}" \
    --out "${CODEBOOK_PATH}"

  test -s "${CODEBOOK_PATH}" || {
    echo "[error] shared-model rebuild failed"
    exit 1
  }

  rm -f "${HOP1_PATH}"
else
  echo "[reuse] ${CODEBOOK_PATH}"
fi

# ============================================================
# Verification: ten TinyStories HOPs + ONE shared model
# ============================================================

python - "${AR_MODE}" "${CODEBOOK_PATH}" "${DATA_HOPS[@]}" <<'PY'
import sys
import torch

mode = sys.argv[1]
codebook_path = sys.argv[2]
data_paths = sys.argv[3:]

if len(data_paths) != 10:
    raise ValueError(f"expected 10 HOP data files, got {len(data_paths)}")

cb = torch.load(codebook_path, map_location="cpu", weights_only=False)

if cb.get("partition_type") != "bpe_local_kmeans":
    raise ValueError(
        f"{mode} requires shared model partition_type='bpe_local_kmeans', "
        f"got {cb.get('partition_type')!r}"
    )

cb_hops = cb.get("hops")
if cb_hops is not None:
    cb_hops = [int(x) for x in cb_hops]
    if cb_hops != list(range(1, 11)):
        raise ValueError(f"shared model hops mismatch: {cb_hops}")

reference_tokens = None
reference_samples = None
local_sizes = []

for expected_hop, data_path in enumerate(data_paths, 1):
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    hop = int(data.get("hop", data.get("args", {}).get("hop", -1)))
    if hop != expected_hop:
        raise ValueError(
            f"HOP order mismatch: slot={expected_hop} data_hop={hop}"
        )

    for key in ("samples", "token_ids_flat", "vq_ids_flat"):
        if key not in data:
            raise KeyError(f"HOP{expected_hop} data missing {key}")

    tokens = data["token_ids_flat"].long().reshape(-1)
    local_vq = data["vq_ids_flat"].long().reshape(-1)

    if tokens.numel() != local_vq.numel():
        raise ValueError(f"HOP{expected_hop}: token/VQ length mismatch")

    if reference_tokens is None:
        reference_tokens = tokens
        reference_samples = len(data["samples"])
    else:
        if not torch.equal(tokens, reference_tokens):
            raise ValueError(f"HOP{expected_hop}: token_ids differ from HOP1")
        if len(data["samples"]) != reference_samples:
            raise ValueError(f"HOP{expected_hop}: sample count differs from HOP1")

    local_size = int(
        data.get("vq_vocab_size", cb.get("max_local_clusters", -1))
    )
    if local_size < 1:
        raise ValueError(f"HOP{expected_hop}: invalid local VQ size")

    vmin = int(local_vq.min())
    vmax = int(local_vq.max())
    if vmin < 0 or vmax >= local_size:
        raise ValueError(
            f"HOP{expected_hop}: local VQ range {vmin}..{vmax}, vocab={local_size}"
        )

    local_sizes.append(local_size)

    print(
        f"[HOP{expected_hop:02d}] "
        f"tokens={tokens.numel():,} "
        f"samples={len(data['samples']):,} "
        f"local_vq_vocab={local_size} "
        f"range={vmin}..{vmax}"
    )

print(f"[verification OK] mode={mode}; HOP1..10 + one shared model")

if mode == "global_vqwar":
    d10 = torch.load(data_paths[9], map_location="cpu", weights_only=False)
    tokens = d10["token_ids_flat"].long().reshape(-1)
    local_vq = d10["vq_ids_flat"].long().reshape(-1)
    size = local_sizes[9]
    pair_vocab = int(torch.unique(tokens * size + local_vq).numel())
    print(f"[global target] HOP10 pair_global_vocab={pair_vocab:,}")
PY

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN="${AR_MODE}_samidare_hop01to10_gap11_localbpe10_${VQ_LABEL}_arseed${AR_SEED}_${TIMESTAMP}"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

echo "============================================================"
echo "mode              = ${AR_MODE}"
echo "HOP input         = HOP1..10 samidare"
echo "alignment         = <=t-11:HOP10, t-10:HOP9, ..., t-2:HOP1, t-1:none"
echo "shared model      = ${CODEBOOK_PATH}"
echo "data HOP1         = ${DATA_HOPS[0]}"
echo "data HOP10        = ${DATA_HOPS[9]}"
echo "epochs/batch/lr   = ${EPOCHS}/${BATCH_SIZE}/${LR}"
echo "disable VQW       = ${DISABLE_VQW}"
echo "run               = ${RUN}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
  --mode "${AR_MODE}" \
  --data_hops "${DATA_HOPS[@]}" \
  --codebook "${CODEBOOK_PATH}" \
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

for p in "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"; do
  test -s "${p}" || {
    echo "[error] missing ${p}"
    exit 1
  }
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
