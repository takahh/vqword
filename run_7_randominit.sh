#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-HOP VQW autoregressive training
#
# Input for target position t:
#   t-1  -> HOP0
#   t-2  -> HOP1
#   ...
#   t-11 -> HOP10
#
# Target:
#   HOP10 VQW[t]
#
# Evaluation:
#   predicted HOP10 VQW
#       -> frozen HOP10 decoder
#       -> BPE[t]
#
# Usage:
#   export FTP_PASS='...'
#   export BPE_INPUT_WEIGHT=0.01
#   bash run_7_multihop.sh 100k 1
#
# Arguments:
#   $1 = VQ codebook label: 100k
#   $2 = AR seed
# ============================================================

apt update
apt install -y lftp

pip install \
  torch \
  tqdm \
  numpy

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
# Arguments
# ============================================================

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {100k} {ar_seed}"
  echo
  echo "Example:"
  echo "  $0 100k 1"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
AR_SEED="$2"

case "${VQ_CODEBOOK_LABEL}" in
  100k)
    VQ_CODEBOOK_SIZE=100000
    ;;
  *)
    echo "[error] This multi-HOP experiment currently expects 100k."
    echo "        received: ${VQ_CODEBOOK_LABEL}"
    exit 1
    ;;
esac

if ! [[ "${AR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] ar_seed must be a non-negative integer:"
  echo "        ${AR_SEED}"
  exit 1
fi

# ============================================================
# Shared settings
# ============================================================

BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257

IVF_NLIST=256
DISCRETIZATION_SEED=0
TARGET_HOP=10
NUM_HOPS=11

MODEL_VARIANT="deconly"
DECODER_EPOCHS=3

# ============================================================
# AR settings
# ============================================================

D_MODEL=256
N_LAYERS=6
N_HEADS=8
DROPOUT=0.1

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# TinyStories ID作成時のseq_lenが256なので、255で十分
MAX_LEN="${MAX_LEN:-255}"

BPE_INPUT_WEIGHT="${BPE_INPUT_WEIGHT:-0.01}"

PIPELINE_BPE_LOSS_WEIGHT="${PIPELINE_BPE_LOSS_WEIGHT:-0}"
PIPELINE_TOPK="${PIPELINE_TOPK:-32}"
PIPELINE_BPE_MAX_TOKENS="${PIPELINE_BPE_MAX_TOKENS:-256}"

# 初回は評価を軽めにする
MARGINAL_TOPKS="${MARGINAL_TOPKS:-1 8 32}"
MARGINAL_MAX_TOKENS="${MARGINAL_MAX_TOKENS:-512}"

# ============================================================
# Python script
# ============================================================

AR_SCRIPT="/vqword/ar_multihop.py"

# 実際に新コードを ar.py として保存した場合は、こちらに変更:
# AR_SCRIPT="/vqword/ar.py"

if [ ! -f "${AR_SCRIPT}" ]; then
  echo "[error] Multi-HOP AR script was not found:"
  echo "        ${AR_SCRIPT}"
  exit 1
fi

# ============================================================
# Filenames
# ============================================================

declare -a HOP_DATA_FILES
declare -a HOP_CODEBOOK_FILES
declare -a HOP_DATA_PATHS
declare -a HOP_CODEBOOK_PATHS

for HOP in $(seq 0 10); do
  HOP2=$(printf "%02d" "${HOP}")

  TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center0_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"

  DATA_FILE="tinystories_vqword_${TAG}_ids.pt"
  CODEBOOK_FILE="wikitext103_vqword_${TAG}.pt"

  HOP_DATA_FILES[HOP]="${DATA_FILE}"
  HOP_CODEBOOK_FILES[HOP]="${CODEBOOK_FILE}"

  HOP_DATA_PATHS[HOP]="/vqword/${DATA_FILE}"
  HOP_CODEBOOK_PATHS[HOP]="/vqword/${CODEBOOK_FILE}"
done

TARGET_TAG="bpe${BPE_VOCAB_LABEL}_bilateral10_center0_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"

TARGET_CODEBOOK="${HOP_CODEBOOK_FILES[10]}"
TARGET_CODEBOOK_PATH="${HOP_CODEBOOK_PATHS[10]}"

DICTIONARY="wikitext103_vqword_${TARGET_TAG}_dictionary.pt"
DICTIONARY_PATH="/vqword/${DICTIONARY}"

RUN="ar_multihop00_10_to_hop10_bpe${BPE_VOCAB_LABEL}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_bpein${BPE_INPUT_WEIGHT}_pipebpe${PIPELINE_BPE_LOSS_WEIGHT}_$(date +%Y%m%d_%H%M%S)"

FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

# ============================================================
# Configuration
# ============================================================

echo "============================================================"
echo "[configuration]"
echo "task                    = distance-dependent HOP0-10 -> HOP10 VQW -> BPE"
echo "distance 1              = HOP0"
echo "distance 2              = HOP1"
echo "distance 11             = HOP10"
echo "number of HOP files     = ${NUM_HOPS}"
echo "target HOP              = ${TARGET_HOP}"
echo "BPE auxiliary weight    = ${BPE_INPUT_WEIGHT}"
echo "VQ vocabulary           = ${VQ_CODEBOOK_SIZE}"
echo "BPE vocabulary          = ${BPE_VOCAB_SIZE}"
echo "AR seed                 = ${AR_SEED}"
echo "d_model                 = ${D_MODEL}"
echo "layers / heads          = ${N_LAYERS} / ${N_HEADS}"
echo "epochs                   = ${EPOCHS}"
echo "batch size               = ${BATCH_SIZE}"
echo "max length               = ${MAX_LEN}"
echo "learning rate            = ${LR}"
echo "marginal top-k           = ${MARGINAL_TOPKS}"
echo "marginal max tokens      = ${MARGINAL_MAX_TOKENS}"
echo "target codebook          = ${TARGET_CODEBOOK}"
echo "target dictionary        = ${DICTIONARY}"
echo "AR script                = ${AR_SCRIPT}"
echo "run                      = ${RUN}"
echo "============================================================"

# ============================================================
# Remove previous local inputs and outputs
# ============================================================

rm -f \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"

for HOP in $(seq 0 10); do
  rm -f "${HOP_DATA_PATHS[HOP]}"
  rm -f "${HOP_CODEBOOK_PATHS[HOP]}"
done

rm -f "${DICTIONARY_PATH}"

# ============================================================
# Download HOP0-10 ID files
# ============================================================

echo "============================================================"
echo "[download HOP0-10 TinyStories ID files]"
echo "============================================================"

for HOP in $(seq 0 10); do
  echo "[download data HOP${HOP}] ${HOP_DATA_FILES[HOP]}"

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes

get "${HOP_DATA_FILES[HOP]}" \
  -o "${HOP_DATA_PATHS[HOP]}"

bye
EOF
done

# ============================================================
# Download HOP0-10 codebooks
# ============================================================

echo "============================================================"
echo "[download HOP0-10 codebooks]"
echo "============================================================"

for HOP in $(seq 0 10); do
  echo "[download codebook HOP${HOP}] ${HOP_CODEBOOK_FILES[HOP]}"

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes

get "${HOP_CODEBOOK_FILES[HOP]}" \
  -o "${HOP_CODEBOOK_PATHS[HOP]}"

bye
EOF
done

# ============================================================
# Download HOP10 decoder dictionary
# ============================================================

echo "============================================================"
echo "[download HOP10 decoder dictionary]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes

get "${DICTIONARY}" \
  -o "${DICTIONARY_PATH}"

bye
EOF

# ============================================================
# Verify downloaded files exist
# ============================================================

for HOP in $(seq 0 10); do
  for PATH_TO_CHECK in \
    "${HOP_DATA_PATHS[HOP]}" \
    "${HOP_CODEBOOK_PATHS[HOP]}"
  do
    if [ ! -s "${PATH_TO_CHECK}" ]; then
      echo "[error] Missing or empty file:"
      echo "        ${PATH_TO_CHECK}"
      exit 1
    fi
  done
done

if [ ! -s "${DICTIONARY_PATH}" ]; then
  echo "[error] Missing or empty dictionary:"
  echo "        ${DICTIONARY_PATH}"
  exit 1
fi

echo "============================================================"
echo "[downloaded files]"
echo "============================================================"

for HOP in $(seq 0 10); do
  ls -lh \
    "${HOP_DATA_PATHS[HOP]}" \
    "${HOP_CODEBOOK_PATHS[HOP]}"
done

ls -lh "${DICTIONARY_PATH}"

# ============================================================
# Cross-file verification
# ============================================================

echo "============================================================"
echo "[verify HOP0-10 inputs]"
echo "============================================================"

python - \
  "${DICTIONARY_PATH}" \
  "${HOP_DATA_PATHS[@]}" \
  "${HOP_CODEBOOK_PATHS[@]}" <<'PY'
import sys
import torch

NUM_HOPS = 11
EXPECTED_VQ_VOCAB = 100000
EXPECTED_TOKEN_VOCAB = 50257
EXPECTED_CENTER_DIM = 256

dictionary_path = sys.argv[1]
data_paths = sys.argv[2:2 + NUM_HOPS]
codebook_paths = sys.argv[2 + NUM_HOPS:2 + 2 * NUM_HOPS]

if len(data_paths) != NUM_HOPS:
    raise ValueError(f"Expected {NUM_HOPS} data files, got {len(data_paths)}")

if len(codebook_paths) != NUM_HOPS:
    raise ValueError(
        f"Expected {NUM_HOPS} codebook files, got {len(codebook_paths)}"
    )

dictionary = torch.load(
    dictionary_path,
    map_location="cpu",
    weights_only=False,
)

decoder_state = dictionary.get("decoder_state_dict")
if decoder_state is None:
    raise KeyError("HOP10 dictionary does not contain decoder_state_dict")

weight = decoder_state.get("weight")
bias = decoder_state.get("bias")

if weight is None or bias is None:
    raise KeyError(
        f"Decoder state is incomplete: keys={list(decoder_state.keys())}"
    )

if tuple(weight.shape) != (EXPECTED_TOKEN_VOCAB, EXPECTED_CENTER_DIM):
    raise ValueError(
        "Decoder weight shape mismatch: "
        f"expected={(EXPECTED_TOKEN_VOCAB, EXPECTED_CENTER_DIM)}, "
        f"actual={tuple(weight.shape)}"
    )

if tuple(bias.shape) != (EXPECTED_TOKEN_VOCAB,):
    raise ValueError(
        "Decoder bias shape mismatch: "
        f"expected={(EXPECTED_TOKEN_VOCAB,)}, "
        f"actual={tuple(bias.shape)}"
    )

reference_tokens = None
reference_samples = None

for expected_hop, path in enumerate(data_paths):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "samples",
        "token_ids_flat",
        "vq_ids_flat",
        "vq_vocab_size",
        "hop",
    }

    missing = sorted(required - set(data.keys()))
    if missing:
        raise KeyError(
            f"HOP{expected_hop} data missing keys: {missing}"
        )

    actual_hop = int(data["hop"])
    if actual_hop != expected_hop:
        raise ValueError(
            f"Data order mismatch: expected HOP{expected_hop}, "
            f"actual HOP{actual_hop}, path={path}"
        )

    actual_vocab = int(data["vq_vocab_size"])
    if actual_vocab != EXPECTED_VQ_VOCAB:
        raise ValueError(
            f"HOP{expected_hop}: VQ vocab mismatch: "
            f"expected={EXPECTED_VQ_VOCAB}, actual={actual_vocab}"
        )

    tokens = data["token_ids_flat"].long().reshape(-1)
    vq_ids = data["vq_ids_flat"].long().reshape(-1)
    samples = data["samples"]

    if tokens.numel() != vq_ids.numel():
        raise ValueError(
            f"HOP{expected_hop}: token/VQ length mismatch: "
            f"token={tokens.numel()}, vq={vq_ids.numel()}"
        )

    if int(tokens.min()) < 0 or int(tokens.max()) >= EXPECTED_TOKEN_VOCAB:
        raise ValueError(
            f"HOP{expected_hop}: BPE IDs out of range: "
            f"{int(tokens.min())}..{int(tokens.max())}"
        )

    if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= EXPECTED_VQ_VOCAB:
        raise ValueError(
            f"HOP{expected_hop}: VQ IDs out of range: "
            f"{int(vq_ids.min())}..{int(vq_ids.max())}"
        )

    if reference_tokens is None:
        reference_tokens = tokens
        reference_samples = samples
    else:
        if not torch.equal(reference_tokens, tokens):
            raise ValueError(
                f"HOP{expected_hop}: token_ids_flat differs from HOP0"
            )

        if len(reference_samples) != len(samples):
            raise ValueError(
                f"HOP{expected_hop}: sample count differs from HOP0"
            )

        for sample_index, (ref, current) in enumerate(
            zip(reference_samples, samples)
        ):
            for key in ("sample_idx", "start", "end", "length"):
                if int(ref[key]) != int(current[key]):
                    raise ValueError(
                        f"HOP{expected_hop}: sample metadata mismatch: "
                        f"sample={sample_index}, key={key}"
                    )

    print(
        f"[data HOP{expected_hop}] "
        f"tokens={tokens.numel():,} "
        f"used_vq={torch.unique(vq_ids).numel():,} "
        f"range={int(vq_ids.min())}..{int(vq_ids.max())}"
    )

for expected_hop, path in enumerate(codebook_paths):
    codebook = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "global_centers" not in codebook:
        raise KeyError(
            f"HOP{expected_hop} codebook does not contain global_centers"
        )

    actual_hop = int(codebook.get("args", {}).get("hop", -1))
    if actual_hop != expected_hop:
        raise ValueError(
            f"Codebook order mismatch: expected HOP{expected_hop}, "
            f"actual HOP{actual_hop}, path={path}"
        )

    centers = codebook["global_centers"]

    if tuple(centers.shape) != (
        EXPECTED_VQ_VOCAB,
        EXPECTED_CENTER_DIM,
    ):
        raise ValueError(
            f"HOP{expected_hop}: center shape mismatch: "
            f"actual={tuple(centers.shape)}"
        )

    print(
        f"[codebook HOP{expected_hop}] centers={tuple(centers.shape)}"
    )

print("============================================================")
print("[all input verification OK]")
print("dictionary:", dictionary_path)
print("samples:", f"{len(reference_samples):,}")
print("tokens:", f"{reference_tokens.numel():,}")
print("============================================================")
PY

# ============================================================
# Build command arrays
# ============================================================

HOP_DATA_ARGS=()
HOP_CODEBOOK_ARGS=()

for HOP in $(seq 0 10); do
  HOP_DATA_ARGS+=("${HOP_DATA_PATHS[HOP]}")
  HOP_CODEBOOK_ARGS+=("${HOP_CODEBOOK_PATHS[HOP]}")
done

# shellcheck disable=SC2206
MARGINAL_TOPK_ARRAY=(${MARGINAL_TOPKS})

# ============================================================
# Train
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "[start multi-HOP autoregressive training]"
echo "input:"
echo "  distance 1  -> HOP0"
echo "  distance 2  -> HOP1"
echo "  ..."
echo "  distance 11 -> HOP10"
echo "target:"
echo "  HOP10 VQW"
echo "evaluation:"
echo "  predicted HOP10 VQW -> frozen decoder -> BPE"
echo "============================================================"

python "${AR_SCRIPT}" \
  --hop_data "${HOP_DATA_ARGS[@]}" \
  --hop_codebooks "${HOP_CODEBOOK_ARGS[@]}" \
  --codebook "${TARGET_CODEBOOK_PATH}" \
  --dictionary "${DICTIONARY_PATH}" \
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
  --bpe_input_weight "${BPE_INPUT_WEIGHT}" \
  --pipeline_bpe_loss_weight "${PIPELINE_BPE_LOSS_WEIGHT}" \
  --pipeline_topk "${PIPELINE_TOPK}" \
  --pipeline_bpe_max_tokens "${PIPELINE_BPE_MAX_TOKENS}" \
  --marginal_topks "${MARGINAL_TOPK_ARRAY[@]}" \
  --marginal_max_tokens "${MARGINAL_MAX_TOKENS}" \
  --out "${FINAL_PATH}" \
  2>&1 | tee "${LOG_PATH}"

# ============================================================
# Verify outputs
# ============================================================

for PATH_TO_CHECK in \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"
do
  if [ ! -s "${PATH_TO_CHECK}" ]; then
    echo "[error] Expected output was not created:"
    echo "        ${PATH_TO_CHECK}"
    exit 1
  fi
done

ls -lh \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"

# ============================================================
# Verify multi-HOP checkpoints
# ============================================================

python - \
  "${BEST_PATH}" \
  "${FINAL_PATH}" <<'PY'
import sys
import torch

for path in sys.argv[1:]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model",
        "args",
        "history",
        "vq_vocab_size",
        "vq_pad_id",
        "token_vocab_size",
        "decoder_frozen",
        "decoder_source",
        "codebook_source",
    }

    missing = sorted(required - set(checkpoint.keys()))
    if missing:
        raise KeyError(f"{path}: missing checkpoint keys: {missing}")

    state = checkpoint["model"]

    required_model = {
        "hop_gates",
        "tok_emb.weight",
        "bpe_proj.weight",
        "pos_emb.weight",
        "vq_head.weight",
        "vq_head.bias",
    }

    for hop in range(11):
        required_model.add(f"hop_projections.{hop}.weight")

    missing_model = sorted(required_model - set(state.keys()))
    if missing_model:
        raise KeyError(
            f"{path}: missing multi-HOP model keys: {missing_model}"
        )

    # frozen centersはpersistent=Falseなので保存されないのが正常
    center_keys = [
        key for key in state
        if key.startswith("center_embeddings.")
    ]

    if center_keys:
        raise ValueError(
            f"{path}: frozen center buffers were unexpectedly saved: "
            f"{center_keys[:5]}"
        )

    if checkpoint["decoder_frozen"] is not True:
        raise ValueError(f"{path}: decoder_frozen is not True")

    print("============================================================")
    print("[checkpoint OK]")
    print("path:", path)
    print("vq_vocab_size:", checkpoint["vq_vocab_size"])
    print("token_vocab_size:", checkpoint["token_vocab_size"])
    print("decoder source:", checkpoint["decoder_source"])
    print("codebook source:", checkpoint["codebook_source"])
    print("last valid:", checkpoint.get("last_valid"))
    print("last test:", checkpoint.get("last_test"))

print("============================================================")
PY

# ============================================================
# Show evaluation lines
# ============================================================

echo "============================================================"
echo "[evaluation lines]"
echo "============================================================"

grep -E \
  "\[epoch [0-9]+\]|\[save best\]|\[save final\]" \
  "${LOG_PATH}" \
  || true

# ============================================================
# Upload outputs
# ============================================================

echo "============================================================"
echo "[upload outputs]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes

cd vqword_logs

put "${BEST_PATH}" \
  -o "${RUN}_best.pt"

put "${FINAL_PATH}" \
  -o "${RUN}.pt"

put "${LOG_PATH}" \
  -o "${RUN}.log"

bye
EOF

echo "============================================================"
echo "[completed]"
echo "TASK       = distance-dependent HOP0-10 -> HOP10 VQW -> BPE"
echo "TARGET     = HOP10"
echo "VQ VOCAB   = ${VQ_CODEBOOK_SIZE}"
echo "BPE WEIGHT = ${BPE_INPUT_WEIGHT}"
echo "BEST       = vqword_logs/${RUN}_best.pt"
echo "FINAL      = vqword_logs/${RUN}.pt"
echo "LOG        = vqword_logs/${RUN}.log"
echo "============================================================"