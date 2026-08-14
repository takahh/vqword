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
  echo "Usage: $0 {10k|25k|50k|100k} {center_scale} {1..10}"
  echo
  echo "Examples:"
  echo "  USE_VQW=1 PURE_BPE_MODE=0 SAMIDARE_HOP=1 VQW_INIT_SCALE=1 $0 10k 1 10"
  echo "  USE_VQW=0 PURE_BPE_MODE=0 SAMIDARE_HOP=0 VQW_INIT_SCALE=1 $0 10k 1 10"
  echo "  USE_VQW=0 PURE_BPE_MODE=1 SAMIDARE_HOP=0 $0 10k 1 10"
  exit 1
fi

# ============================================================
# 実験設定
# ============================================================

VQ_CODEBOOK_LABEL="$1"
CENTER_SCALE="$2"
DISTANT_HOP="$3"
USE_GLOBAL_VQW_ID="${USE_GLOBAL_VQW_ID:-0}"
case "${USE_GLOBAL_VQW_ID}" in
  0|1) ;;
  *)
    echo "[error] USE_GLOBAL_VQW_ID must be 0 or 1"
    exit 1
    ;;
esac

AR_SEED="${AR_SEED:-0}"
USE_VQW="${USE_VQW:-1}"
PURE_BPE_MODE="${PURE_BPE_MODE:-0}"
SAMIDARE_HOP="${SAMIDARE_HOP:-${USE_VQW}}"
VQW_INIT_SCALE="${VQW_INIT_SCALE:-0.1}"

# SAMIDARE_HOP=0では、第3引数のHOP未満の距離をマスクし、
# 指定HOP距離以上だけで固定HOPを使う。
if ! [[ "${DISTANT_HOP}" =~ ^([1-9]|10)$ ]]; then
  echo "[error] distant_hop must be an integer from 1 to 10"
  exit 1
fi

# ============================================================
# 引数検証
# ============================================================

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

case "${PURE_BPE_MODE}" in
  0|1) ;;
  *)
    echo "[error] PURE_BPE_MODE must be 0 or 1"
    exit 1
    ;;
esac

case "${SAMIDARE_HOP}" in
  0|1) ;;
  *)
    echo "[error] SAMIDARE_HOP must be 0 or 1"
    exit 1
    ;;
esac

if [ "${PURE_BPE_MODE}" = "1" ] && [ "${USE_VQW}" != "0" ]; then
  echo "[error] PURE_BPE_MODE=1 requires USE_VQW=0"
  exit 1
fi

case "${CENTER_SCALE}" in
  1.0)  CENTER_LABEL="1" ;;
  0.30) CENTER_LABEL="0.3" ;;
  *)    CENTER_LABEL="${CENTER_SCALE}" ;;
esac

# ============================================================
# VQW設定
# ============================================================

BPE_VOCAB_LABEL=50257
IVF_NLIST=256
DISCRETIZATION_SEED=0
MODEL_VARIANT="deconly"
DECODER_EPOCHS=3

# ============================================================
# AR設定
# ============================================================

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

# ============================================================
# ファイル名
# ============================================================

COMMON_SUFFIX="center${CENTER_LABEL}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"

DATA_PATTERN="/vqword/tinystories_vqword_bpe${BPE_VOCAB_LABEL}_bilateral{hop:02d}_${COMMON_SUFFIX}_ids.pt"

CODEBOOK_PATTERN="/vqword/wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral{hop:02d}_${COMMON_SUFFIX}.pt"

# ============================================================
# FTPダウンロード
# ============================================================

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

# ============================================================
# HOP1..10をダウンロード
#
# codebook：WikiText-103で学習したトークナイザー
# data：そのトークナイザーをTinyStoriesへ適用したAR用ID
# ============================================================

for HOP_INDEX in $(seq 1 10); do
  HOP2=$(printf "%02d" "${HOP_INDEX}")

  CODEBOOK_PREFIX="wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_${COMMON_SUFFIX}"

  DATA_PREFIX="tinystories_vqword_bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_${COMMON_SUFFIX}"

  CODEBOOK_FILE="${CODEBOOK_PREFIX}.pt"
  DATA_FILE="${DATA_PREFIX}_ids.pt"

  download_file \
    "${CODEBOOK_FILE}" \
    "/vqword/${CODEBOOK_FILE}"

  download_file \
    "${DATA_FILE}" \
    "/vqword/${DATA_FILE}"
done
# ============================================================
# HOP1..10の事前検証
# ============================================================

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

for hop in range(1, 11):
    data_path = data_pattern.format(hop=hop)
    codebook_path = codebook_pattern.format(hop=hop)

    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=False,
    )

    codebook = torch.load(
        codebook_path,
        map_location="cpu",
        weights_only=False,
    )

    for key in (
        "samples",
        "token_ids_flat",
        "vq_ids_flat",
    ):
        if key not in data:
            raise KeyError(
                f"HOP{hop} data missing key: {key}"
            )

    if "global_centers" not in codebook:
        raise KeyError(
            f"HOP{hop} codebook missing global_centers"
        )

    data_hop = int(
        data.get("hop", -1)
    )

    codebook_hop = int(
        codebook.get("args", {}).get("hop", -1)
    )

    if data_hop != hop or codebook_hop != hop:
        raise ValueError(
            f"HOP mismatch at HOP{hop}: "
            f"data={data_hop}, codebook={codebook_hop}"
        )

    tokens = (
        data["token_ids_flat"]
        .long()
        .reshape(-1)
    )

    vq_ids = (
        data["vq_ids_flat"]
        .long()
        .reshape(-1)
    )

    centers = codebook["global_centers"]

    bounds = [
        (
            int(sample["start"]),
            int(sample["end"]),
        )
        for sample in data["samples"]
    ]

    if tokens.numel() != vq_ids.numel():
        raise ValueError(
            f"HOP{hop} token/VQ length mismatch: "
            f"tokens={tokens.numel()}, "
            f"vq_ids={vq_ids.numel()}"
        )

    if int(centers.shape[0]) != expected_vq_size:
        raise ValueError(
            f"HOP{hop} VQ size mismatch: "
            f"expected={expected_vq_size}, "
            f"actual={centers.shape[0]}"
        )

    if vq_ids.numel() == 0:
        raise ValueError(
            f"HOP{hop} contains no VQ IDs"
        )

    vq_min = int(vq_ids.min().item())
    vq_max = int(vq_ids.max().item())

    if vq_min < 0 or vq_max >= int(centers.shape[0]):
        raise ValueError(
            f"HOP{hop} VQ ID out of range: "
            f"{vq_min}..{vq_max}"
        )

    actual_scale = float(
        codebook
        .get("args", {})
        .get("center_scale", -1.0)
    )

    if abs(actual_scale - expected_center_scale) > 1e-8:
        raise ValueError(
            f"HOP{hop} center scale mismatch: "
            f"expected={expected_center_scale}, "
            f"actual={actual_scale}"
        )

    if reference_tokens is None:
        reference_tokens = tokens
        reference_bounds = bounds
    else:
        if not torch.equal(tokens, reference_tokens):
            raise ValueError(
                f"HOP{hop} token_ids_flat differs from HOP1"
            )

        if bounds != reference_bounds:
            raise ValueError(
                f"HOP{hop} sample boundaries differ from HOP1"
            )

    used_vq = torch.unique(vq_ids).numel()

    print(
        f"[verification OK] HOP{hop:02d} "
        f"tokens={tokens.numel():,} "
        f"samples={len(bounds):,} "
        f"used_vq={used_vq:,} "
        f"centers={tuple(centers.shape)}"
    )
PY

# ============================================================
# 出力名
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN="ar_inputcat_bpeonly_multihop01to10_usevqw${USE_VQW}_purebpe${PURE_BPE_MODE}_samidare${SAMIDARE_HOP}_globalid${USE_GLOBAL_VQW_ID}_distanthop${DISTANT_HOP}_bpe${BPE_VOCAB_LABEL}_center${CENTER_LABEL}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_${TIMESTAMP}"

FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

# ============================================================
# 設定表示
# ============================================================

echo "============================================================"
echo "[start multi-HOP BPE-only AR training]"
if [ "${SAMIDARE_HOP}" = "1" ]; then
  echo "VQW distance mapping  = 1:HOP1 ... 10:HOP10, 11+:HOP10"
else
  echo "VQW distance mapping  = distance ${DISTANT_HOP}+: fixed HOP${DISTANT_HOP}"
fi
echo "data HOP range        = HOP1..HOP10"
echo "data pattern          = ${DATA_PATTERN}"
echo "codebook pattern      = ${CODEBOOK_PATTERN}"
echo "use VQW               = ${USE_VQW}"
echo "pure BPE mode         = ${PURE_BPE_MODE}"
echo "samidare HOP          = ${SAMIDARE_HOP}"
echo "fixed distant HOP     = ${DISTANT_HOP}"
echo "minimum key distance  = ${DISTANT_HOP} (when samidare=0)"
echo "prediction-relative   = $((DISTANT_HOP + 1)) tokens back or earlier"
echo "VQW initial scale     = ${VQW_INIT_SCALE}"
echo "center scale          = ${CENTER_SCALE}"
echo "codebook size         = ${VQ_CODEBOOK_SIZE}"
echo "AR seed               = ${AR_SEED}"
echo "global VQW ID         = ${USE_GLOBAL_VQW_ID}"
echo "run                   = ${RUN}"
echo "============================================================"

# ============================================================
# AR実行
# ============================================================

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
  --pure_bpe_mode "${PURE_BPE_MODE}" \
  --samidare_hop "${SAMIDARE_HOP}" \
  --distant_hop "${DISTANT_HOP}" \
  --vqw_init_scale "${VQW_INIT_SCALE}" \
  --use_global_vqw_id "${USE_GLOBAL_VQW_ID}" \
  --out "${FINAL_PATH}" \
  2>&1 | tee "${LOG_PATH}"

# ============================================================
# 出力確認
# ============================================================

for PATH_TO_CHECK in \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"
do
  if [ ! -s "${PATH_TO_CHECK}" ]; then
    echo "[error] missing output: ${PATH_TO_CHECK}"
    exit 1
  fi
done

grep -E \
  '^\[epoch [0-9]+\]|^\[save best\]|^\[save final\]' \
  "${LOG_PATH}" || true

# ============================================================
# FTPアップロード
# ============================================================

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
