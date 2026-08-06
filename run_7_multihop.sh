
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

# ============================================================
# FTP
# ============================================================

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

# ============================================================
# Usage
# ============================================================

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {100k} {ar_seed}"
  echo
  echo "Example:"
  echo "  CENTER_SCALE=0 \\"
  echo "  $0 100k 0"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
AR_SEED="$2"

case "${VQ_CODEBOOK_LABEL}" in
  100k)
    VQ_CODEBOOK_SIZE=100000
    ;;
  *)
    echo "[error] expected codebook label: 100k"
    exit 1
    ;;
esac

# ============================================================
# Common settings
# ============================================================

CENTER_SCALE="${CENTER_SCALE:-0}"
USE_VQW="${USE_VQW:-1}"
VQW_INIT_SCALE="${VQW_INIT_SCALE:-0.1}"
case "${USE_VQW}" in
  0|1)
    ;;
  *)
    echo "[error] USE_VQW must be 0 or 1"
    exit 1
    ;;
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

# Input-CAT -> shared Transformer -> BPE-only model
AR_SCRIPT="/vqword/ar_multihop.py"

# ============================================================
# Construct HOP0 ... HOP10 filenames
# ============================================================

declare -a HOP_DATA_FILES
declare -a HOP_DATA_PATHS
declare -a HOP_CODEBOOK_FILES
declare -a HOP_CODEBOOK_PATHS

for HOP in $(seq 0 10); do
  HOP2=$(printf "%02d" "${HOP}")

  TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center${CENTER_SCALE}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"

  DATA_FILE="tinystories_vqword_${TAG}_ids.pt"
  CODEBOOK_FILE="wikitext103_vqword_${TAG}.pt"

  HOP_DATA_FILES+=("${DATA_FILE}")
  HOP_DATA_PATHS+=("/vqword/${DATA_FILE}")

  HOP_CODEBOOK_FILES+=("${CODEBOOK_FILE}")
  HOP_CODEBOOK_PATHS+=("/vqword/${CODEBOOK_FILE}")
done

# ============================================================
# Output names
# ============================================================

RUN="ar_inputcat_bpeonly_multihop_usevqw${USE_VQW}_bpe${BPE_VOCAB_LABEL}_bilateral00to10_center${CENTER_SCALE}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_$(date +%Y%m%d_%H%M%S)"

FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

# ============================================================
# Download HOP0 ... HOP10 data and codebooks
# ============================================================

download_file() {
  local remote_file="$1"
  local local_path="$2"

  if [ -s "${local_path}" ]; then
    echo "[reuse] ${local_path}"
    return
  fi

  echo "[download] ${remote_file}"

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${remote_file}" -o "${local_path}"
bye
EOF

  if [ ! -s "${local_path}" ]; then
    echo "[error] download failed: ${local_path}"
    exit 1
  fi
}

for HOP in $(seq 0 10); do
  download_file \
    "${HOP_DATA_FILES[$HOP]}" \
    "${HOP_DATA_PATHS[$HOP]}"

  download_file \
    "${HOP_CODEBOOK_FILES[$HOP]}" \
    "${HOP_CODEBOOK_PATHS[$HOP]}"
done

# ============================================================
# Required-file check
# ============================================================

for HOP in $(seq 0 10); do
  for PATH_TO_CHECK in \
    "${HOP_DATA_PATHS[$HOP]}" \
    "${HOP_CODEBOOK_PATHS[$HOP]}"
  do
    if [ ! -s "${PATH_TO_CHECK}" ]; then
      echo "[error] missing: ${PATH_TO_CHECK}"
      exit 1
    fi
  done
done

if [ ! -s "${AR_SCRIPT}" ]; then
  echo "[error] missing AR script: ${AR_SCRIPT}"
  exit 1
fi

# ============================================================
# Verify all 11 data/codebook pairs
# ============================================================

python - \
  "${VQ_CODEBOOK_SIZE}" \
  "${CENTER_SCALE}" \
  "${HOP_DATA_PATHS[@]}" \
  "${HOP_CODEBOOK_PATHS[@]}" <<'PY'
import sys
import torch

if len(sys.argv) != 25:
    raise ValueError(
        f"expected 24 arguments after script name, got {len(sys.argv) - 1}"
    )

expected_vq_size = int(sys.argv[1])
expected_center_scale = float(sys.argv[2])

data_paths = sys.argv[3:14]
codebook_paths = sys.argv[14:25]

if len(data_paths) != 11:
    raise ValueError(f"expected 11 data files, got {len(data_paths)}")

if len(codebook_paths) != 11:
    raise ValueError(
        f"expected 11 codebook files, got {len(codebook_paths)}"
    )

all_data = [
    torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    for path in data_paths
]

all_codebooks = [
    torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    for path in codebook_paths
]

reference_data = all_data[0]

for key in ("samples", "token_ids_flat", "vq_ids_flat"):
    if key not in reference_data:
        raise KeyError(f"HOP0 data missing key: {key}")

reference_tokens = (
    reference_data["token_ids_flat"]
    .long()
    .reshape(-1)
)

reference_samples = list(reference_data["samples"])

reference_center_shape = None

for hop, (data, codebook) in enumerate(
    zip(all_data, all_codebooks)
):
    for key in ("samples", "token_ids_flat", "vq_ids_flat"):
        if key not in data:
            raise KeyError(
                f"HOP{hop} data missing key: {key}"
            )

    if "global_centers" not in codebook:
        raise KeyError(
            f"HOP{hop} codebook missing global_centers"
        )

    data_hop = int(data.get("hop", -1))
    codebook_hop = int(
        codebook.get("args", {}).get("hop", -1)
    )

    if data_hop != hop:
        raise ValueError(
            f"HOP{hop} data metadata mismatch: "
            f"recorded hop={data_hop}"
        )

    if codebook_hop != hop:
        raise ValueError(
            f"HOP{hop} codebook metadata mismatch: "
            f"recorded hop={codebook_hop}"
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

    if not torch.equal(tokens, reference_tokens):
        raise ValueError(
            f"HOP{hop}: token_ids_flat differs from HOP0"
        )

    if tokens.numel() != vq_ids.numel():
        raise ValueError(
            f"HOP{hop}: token/VQ length mismatch: "
            f"{tokens.numel()} vs {vq_ids.numel()}"
        )

    samples = list(data["samples"])

    if len(samples) != len(reference_samples):
        raise ValueError(
            f"HOP{hop}: sample count mismatch: "
            f"{len(samples)} vs {len(reference_samples)}"
        )

    for sample_index, (ref_sample, current_sample) in enumerate(
        zip(reference_samples, samples)
    ):
        for key in ("sample_idx", "start", "end", "length"):
            ref_value = int(ref_sample[key])
            current_value = int(current_sample[key])

            if ref_value != current_value:
                raise ValueError(
                    f"HOP{hop}: sample metadata mismatch "
                    f"at sample={sample_index}, key={key}: "
                    f"{current_value} vs {ref_value}"
                )

    centers = codebook["global_centers"]

    if int(centers.shape[0]) != expected_vq_size:
        raise ValueError(
            f"HOP{hop}: expected {expected_vq_size} centers, "
            f"got {centers.shape[0]}"
        )

    if reference_center_shape is None:
        reference_center_shape = tuple(centers.shape)
    elif tuple(centers.shape) != reference_center_shape:
        raise ValueError(
            f"HOP{hop}: center shape mismatch: "
            f"{tuple(centers.shape)} vs "
            f"{reference_center_shape}"
        )

    if int(vq_ids.min()) < 0:
        raise ValueError(
            f"HOP{hop}: negative VQ ID found"
        )

    if int(vq_ids.max()) >= centers.shape[0]:
        raise ValueError(
            f"HOP{hop}: VQ ID out of range: "
            f"max={int(vq_ids.max())}, "
            f"vocab={centers.shape[0]}"
        )

    data_center_scale = data.get(
        "center_scale",
        None,
    )

    codebook_center_scale = (
        codebook
        .get("args", {})
        .get("center_scale", None)
    )

    print(
        f"[HOP{hop:02d}] "
        f"tokens={tokens.numel():,} "
        f"samples={len(samples):,} "
        f"used_vq={torch.unique(vq_ids).numel():,} "
        f"range={int(vq_ids.min())}..{int(vq_ids.max())} "
        f"centers={tuple(centers.shape)} "
        f"data_center_scale={data_center_scale} "
        f"codebook_center_scale={codebook_center_scale}"
    )

print("============================================================")
print("[verification OK]")
print("HOP range: 0 ... 10")
print("tokens:", f"{reference_tokens.numel():,}")
print("samples:", f"{len(reference_samples):,}")
print("center shape:", reference_center_shape)
print("requested center scale:", expected_center_scale)
print("============================================================")
PY

# ============================================================
# Start training
# ============================================================

echo "============================================================"
echo "[start input-CAT BPE-only AR training]"
echo
echo "Input:"
echo "  BPE[t] embedding"

if [ "${USE_VQW}" -eq 1 ]; then
  echo "  + aggregated HOP0..HOP10 frozen-center context"
else
  echo "  + zero VQ context"
fi
echo "Fusion:"
echo "  CAT(BPE embedding, multi-hop VQ context)"
echo "  -> learned projection"
echo "  -> shared causal Transformer"
echo "  -> BPE[t+1]"
echo
echo "use VQW              = ${USE_VQW}"
echo "VQ prediction head   = disabled"
echo "VQ auxiliary loss    = disabled"
echo "center scale         = ${CENTER_SCALE}"
echo "AR seed              = ${AR_SEED}"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
  --hop_data \
    "${HOP_DATA_PATHS[@]}" \
  --hop_codebooks \
    "${HOP_CODEBOOK_PATHS[@]}" \
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
  --max_len 255 \
  2>&1 | tee "${LOG_PATH}"

# ============================================================
# Output check
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

echo
echo "============================================================"
echo "[training summary]"
echo "============================================================"

grep -E \
  "\[epoch [0-9]+\]|\[save best\]|\[save final\]" \
  "${LOG_PATH}" \
  || true

# ============================================================
# Upload outputs
# ============================================================

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

echo
echo "============================================================"
echo "[completed]"
echo "run   = ${RUN}"
echo "best  = ${BEST_PATH}"
echo "final = ${FINAL_PATH}"
echo "log   = ${LOG_PATH}"
echo "============================================================"

