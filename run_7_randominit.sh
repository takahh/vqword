#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 7: VQW[t] + alpha * BPE[t] -> VQW[t+1]
#         -> fixed decoder -> BPE[t+1]
#
# VQW is the primary input and target.
# BPE[t] is used only as auxiliary input.
# The learned VQ-center-to-BPE decoder is loaded from the
# VQWord dictionary and remains frozen during AR training.
#
# Usage:
#   export FTP_PASS='...'
#   bash run_7_vqw_ar.sh 200k 1.0 0
#
# Optional artifact filename suffix:
#   export ARTIFACT_SUFFIX='_deconly_dec3'
#   bash run_7_vqw_ar.sh 200k 1.0 0
#
# With ARTIFACT_SUFFIX='_deconly_dec3', expected input names are:
#   wikitext103_vqword_<TAG>_deconly_dec3.pt
#   wikitext103_vqword_<TAG>_deconly_dec3_dictionary.pt
#   tinystories_vqword_<TAG>_deconly_dec3_ids.pt
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

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 {25k|50k|100k|200k|300k} {center_scale} {ar_seed}"
  echo
  echo "Example:"
  echo "  $0 200k 1.0 0"
  echo
  echo "Optional:"
  echo "  ARTIFACT_SUFFIX='_deconly_dec3' $0 200k 1.0 0"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
CENTER_SCALE_RAW="$2"
AR_SEED="$3"

CENTER_SCALE="$(
  python -c '
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit(f"[error] invalid center_scale: {value}")
print(f"{value:g}")
' "${CENTER_SCALE_RAW}"
)"

case "${VQ_CODEBOOK_LABEL}" in
  25k)  VQ_CODEBOOK_SIZE=25000 ;;
  50k)  VQ_CODEBOOK_SIZE=50000 ;;
  100k) VQ_CODEBOOK_SIZE=100000 ;;
  200k) VQ_CODEBOOK_SIZE=200000 ;;
  300k) VQ_CODEBOOK_SIZE=300000 ;;
  *)
    echo "[error] Unsupported VQ codebook: ${VQ_CODEBOOK_LABEL}"
    exit 1
    ;;
esac

if ! [[ "${AR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] ar_seed must be a non-negative integer: ${AR_SEED}"
  exit 1
fi

# ============================================================
# Tokenizer / VQ settings
# ============================================================

BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257

IVF_NLIST=256
DISCRETIZATION_SEED=0

# This suffix must exactly match the filenames produced by:
#   1. train_vqword_decoder_only.py
#   2. TinyStories VQ-ID assignment
#
# Keep empty only when those artifacts use the legacy base TAG.
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"

# ============================================================
# AR settings
# ============================================================

D_MODEL=256
N_LAYERS=6
N_HEADS=8
DROPOUT=0.1

EPOCHS=30
BATCH_SIZE=16
LR=3e-4
WEIGHT_DECAY=1e-4
MAX_LEN=512

# Auxiliary BPE input strength.
# 0 = VQW-only baseline; recommended first trial = 0.01
BPE_INPUT_WEIGHT="${BPE_INPUT_WEIGHT:-0.01}"

# Keep the training target as next VQW.
PIPELINE_BPE_LOSS_WEIGHT="${PIPELINE_BPE_LOSS_WEIGHT:-0}"
#PIPELINE_TOPK="${PIPELINE_TOPK:-8}"
PIPELINE_TOPK=128
PIPELINE_BPE_MAX_TOKENS="${PIPELINE_BPE_MAX_TOKENS:-512}"

# ============================================================
# Filenames
# ============================================================
HOP=75

BASE_TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center${CENTER_SCALE}_deconly_dec3_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"
ARTIFACT_TAG="${BASE_TAG}"

DATA="${DATA_FILE:-tinystories_vqword_${ARTIFACT_TAG}_ids.pt}"
CODEBOOK="${CODEBOOK_FILE:-wikitext103_vqword_${ARTIFACT_TAG}.pt}"
DICTIONARY="${DICTIONARY_FILE:-wikitext103_vqword_${ARTIFACT_TAG}_dictionary.pt}"

DATA_PATH="/vqword/${DATA}"
CODEBOOK_PATH="/vqword/${CODEBOOK}"
DICTIONARY_PATH="/vqword/${DICTIONARY}"

AR_SCRIPT="/vqword/ar.py"

RUN="ar_vqw_bpeaux2vqw2bpe_${ARTIFACT_TAG}_arseed${AR_SEED}_bpein${BPE_INPUT_WEIGHT}_pipebpe${PIPELINE_BPE_LOSS_WEIGHT}_$(date +%Y%m%d_%H%M%S)"

FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

# ============================================================
# Configuration
# ============================================================

echo "============================================================"
echo "[configuration]"
echo "task                   =  concat(VQW[t], α·BPE[t]) -> VQW[t+1] -> fixed decoder -> BPE[t+1]"
echo "AR primary input        = VQW[t]"
echo "AR auxiliary input      = BPE[t]"
echo "BPE input weight        = ${BPE_INPUT_WEIGHT}"
echo "AR target               = next VQW"
echo "decoder                 = pretrained and frozen"
echo "BPE vocabulary          = ${BPE_VOCAB_SIZE}"
echo "VQW codebook            = ${VQ_CODEBOOK_SIZE}"
echo "center scale            = ${CENTER_SCALE}"
echo "hop                     = ${HOP}"
echo "IVF nlist               = ${IVF_NLIST}"
echo "discretization seed     = ${DISCRETIZATION_SEED}"
echo "AR seed                 = ${AR_SEED}"
echo "artifact suffix         = '${ARTIFACT_SUFFIX}'"
echo "data                    = ${DATA}"
echo "codebook                = ${CODEBOOK}"
echo "dictionary/decoder      = ${DICTIONARY}"
echo "d_model                 = ${D_MODEL}"
echo "layers / heads          = ${N_LAYERS} / ${N_HEADS}"
echo "epochs                  = ${EPOCHS}"
echo "batch size              = ${BATCH_SIZE}"
echo "learning rate           = ${LR}"
echo "pipeline BPE loss weight= ${PIPELINE_BPE_LOSS_WEIGHT}"
echo "run                     = ${RUN}"
echo "============================================================"

# ============================================================
# Required script
# ============================================================

if [ ! -f "${AR_SCRIPT}" ]; then
  echo "[error] Missing AR script:"
  echo "        ${AR_SCRIPT}"
  exit 1
fi

# ============================================================
# Clean local inputs/outputs
# ============================================================

rm -f \
  "${DATA_PATH}" \
  "${CODEBOOK_PATH}" \
  "${DICTIONARY_PATH}" \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"

# ============================================================
# Download TinyStories IDs, VQ codebook, learned decoder
# ============================================================

echo "============================================================"
echo "[download input files]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${DATA}" -o "${DATA_PATH}"
get "${CODEBOOK}" -o "${CODEBOOK_PATH}"
get "${DICTIONARY}" -o "${DICTIONARY_PATH}"

bye
EOF

for path in \
  "${DATA_PATH}" \
  "${CODEBOOK_PATH}" \
  "${DICTIONARY_PATH}"
do
  if [ ! -s "${path}" ]; then
    echo "[error] Missing or empty input file:"
    echo "        ${path}"
    exit 1
  fi
done

ls -lh \
  "${DATA_PATH}" \
  "${CODEBOOK_PATH}" \
  "${DICTIONARY_PATH}"

# ============================================================
# Cross-file verification
#
# Especially important:
#   dictionary must contain the TRAINED decoder_state_dict.
# ============================================================

python - <<PY
import torch

data_path = "${DATA_PATH}"
codebook_path = "${CODEBOOK_PATH}"
dictionary_path = "${DICTIONARY_PATH}"

expected_token_vocab = ${BPE_VOCAB_SIZE}
expected_vq_vocab = ${VQ_CODEBOOK_SIZE}
expected_center_dim = ${D_MODEL}

data = torch.load(data_path, map_location="cpu", weights_only=False)
codebook = torch.load(codebook_path, map_location="cpu", weights_only=False)
dictionary = torch.load(dictionary_path, map_location="cpu", weights_only=False)

required_data = {
    "samples",
    "token_ids_flat",
    "vq_ids_flat",
    "vq_vocab_size",
}
missing = sorted(required_data - set(data))
if missing:
    raise KeyError(f"Missing data keys: {missing}")

if "global_centers" not in codebook:
    raise KeyError("Codebook does not contain global_centers")

if "decoder_state_dict" not in dictionary:
    raise KeyError(
        "Dictionary does not contain decoder_state_dict. "
        "This is probably the old dictionary, not the decoder-trained file."
    )

if dictionary.get("decoder_type") != "linear_center_to_bpe":
    raise ValueError(
        "Unexpected decoder_type: "
        f"{dictionary.get('decoder_type')!r}"
    )

token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)
centers = codebook["global_centers"].float()
decoder = dictionary["decoder_state_dict"]

data_vq_vocab = int(data["vq_vocab_size"])
dict_vq_vocab = int(dictionary["vq_vocab_size"])
center_vq_vocab, center_dim = map(int, centers.shape)

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        f"Token/VQ length mismatch: "
        f"{token_ids.numel():,} vs {vq_ids.numel():,}"
    )

if data_vq_vocab != expected_vq_vocab:
    raise ValueError(
        f"Data VQ vocab mismatch: "
        f"expected={expected_vq_vocab:,}, actual={data_vq_vocab:,}"
    )

if dict_vq_vocab != expected_vq_vocab:
    raise ValueError(
        f"Dictionary VQ vocab mismatch: "
        f"expected={expected_vq_vocab:,}, actual={dict_vq_vocab:,}"
    )

if center_vq_vocab != expected_vq_vocab:
    raise ValueError(
        f"Codebook VQ vocab mismatch: "
        f"expected={expected_vq_vocab:,}, actual={center_vq_vocab:,}"
    )

if center_dim != expected_center_dim:
    raise ValueError(
        f"Center dimension mismatch: "
        f"expected={expected_center_dim}, actual={center_dim}"
    )

weight = decoder.get("weight")
bias = decoder.get("bias")

if weight is None or bias is None:
    raise KeyError(
        f"Decoder state must contain weight and bias; "
        f"keys={list(decoder)}"
    )

if tuple(weight.shape) != (expected_token_vocab, center_dim):
    raise ValueError(
        f"Decoder weight shape mismatch: "
        f"expected={(expected_token_vocab, center_dim)}, "
        f"actual={tuple(weight.shape)}"
    )

if tuple(bias.shape) != (expected_token_vocab,):
    raise ValueError(
        f"Decoder bias shape mismatch: "
        f"expected={(expected_token_vocab,)}, "
        f"actual={tuple(bias.shape)}"
    )

if int(token_ids.min()) < 0 or int(token_ids.max()) >= expected_token_vocab:
    raise ValueError(
        f"BPE IDs out of range: "
        f"min={int(token_ids.min())}, max={int(token_ids.max())}"
    )

if int(vq_ids.min()) < 0 or int(vq_ids.max()) >= expected_vq_vocab:
    raise ValueError(
        f"VQ IDs out of range: "
        f"min={int(vq_ids.min())}, max={int(vq_ids.max())}"
    )

print("============================================================")
print("[input verification OK]")
print("samples:", f"{len(data['samples']):,}")
print("tokens:", f"{token_ids.numel():,}")
print("used BPE IDs:", f"{torch.unique(token_ids).numel():,}")
print("used VQ IDs:", f"{torch.unique(vq_ids).numel():,}")
print("centers:", tuple(centers.shape))
print("decoder weight:", tuple(weight.shape))
print("decoder metrics:", dictionary.get("decoder_metrics"))
print("============================================================")
PY

# ============================================================
# Train VQW -> VQW; decode predicted VQW with frozen decoder
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "[start VQW autoregressive training]"
echo "input       = VQW[t] + ${BPE_INPUT_WEIGHT} * BPE[t]"
echo "target      = VQW[t+1]"
echo "evaluation  = predicted VQW -> frozen decoder -> BPE[t+1]"
echo "============================================================"

python "${AR_SCRIPT}" \
  --data "${DATA_PATH}" \
  --dictionary "${DICTIONARY_PATH}" \
  --codebook "${CODEBOOK_PATH}" \
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
  --out "${FINAL_PATH}" \
  --marginal_topks 1 8 32 128 \
  --marginal_max_tokens 0 \
  2>&1 | tee "${LOG_PATH}"

# ============================================================
# Generated outputs
# ============================================================

for path in \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"
do
  if [ ! -s "${path}" ]; then
    echo "[error] Expected output was not generated:"
    echo "        ${path}"
    exit 1
  fi
done

ls -lh \
  "${FINAL_PATH}" \
  "${BEST_PATH}" \
  "${LOG_PATH}"

# ============================================================
# Summary
# ============================================================

echo "============================================================"
echo "[evaluation lines]"
echo "============================================================"

grep -E \
  "\[epoch [0-9]+\]|\[save best\]|\[save final\]" \
  "${LOG_PATH}" \
  || true

python - <<PY
import re

log_path = "${LOG_PATH}"

pattern = re.compile(
    r"\[epoch\s+(\d+)\]\s+"
    r"valid_vq_ppl=([0-9.]+)\s+"
    r"valid_vq_acc=([0-9.]+)\s+"
    r"valid_bpe_top1=([0-9.]+)\s+"
    r"test_vq_ppl=([0-9.]+)\s+"
    r"test_bpe_top1=([0-9.]+)\s+"
    r"oracle_bpe_top1=([0-9.]+)"
)

rows = []
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "valid_vq_ppl": float(m.group(2)),
                "valid_vq_acc": float(m.group(3)),
                "valid_bpe_top1": float(m.group(4)),
                "test_vq_ppl": float(m.group(5)),
                "test_bpe_top1": float(m.group(6)),
                "oracle_bpe_top1": float(m.group(7)),
            })

print("============================================================")
print("[AR summary]")

if not rows:
    print("No epoch result lines found")
else:
    best = min(rows, key=lambda x: x["valid_vq_ppl"])
    print("best epoch by valid VQ PPL:", best["epoch"])
    print("valid VQ PPL:", best["valid_vq_ppl"])
    print("valid VQ accuracy:", best["valid_vq_acc"])
    print("test VQ PPL:", best["test_vq_ppl"])
    print("test pipeline BPE top1:", best["test_bpe_top1"])
    print("oracle decoder BPE top1:", best["oracle_bpe_top1"])

print("============================================================")
PY

# ============================================================
# Checkpoints
# ============================================================

python - <<PY
import torch

for path in ["${BEST_PATH}", "${FINAL_PATH}"]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

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
    missing = sorted(required - set(ckpt))
    if missing:
        raise KeyError(f"{path}: missing checkpoint keys: {missing}")

    model = ckpt["model"]
    required_model = {
        "vq_emb.weight",
        "tok_emb.weight",
        "bpe_proj.weight",
        "pos_emb.weight",
        "vq_head.weight",
        "vq_head.bias",
    }
    missing_model = sorted(required_model - set(model))
    if missing_model:
        raise KeyError(f"{path}: missing model keys: {missing_model}")

    forbidden_model = {
        "tok_head.weight",
        "input_fusion.weight",
        "fusion.weight",
        "fusion.bias",
    }
    present_forbidden = sorted(forbidden_model & set(model))
    if present_forbidden:
        raise ValueError(
            f"{path}: old BPE/fusion parameters remain: "
            f"{present_forbidden}"
        )

    if ckpt["decoder_frozen"] is not True:
        raise ValueError(f"{path}: decoder_frozen is not True")

    print("============================================================")
    print("[checkpoint OK]")
    print("path:", path)
    print("vq_vocab_size:", ckpt["vq_vocab_size"])
    print("token_vocab_size:", ckpt["token_vocab_size"])
    print("decoder source:", ckpt["decoder_source"])
    print("codebook source:", ckpt["codebook_source"])
    print("last valid:", ckpt.get("last_valid"))
    print("last test:", ckpt.get("last_test"))

print("============================================================")
PY

# ============================================================
# Upload outputs
# ============================================================

echo "============================================================"
echo "[upload outputs]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

cd vqword_logs

put "${BEST_PATH}" -o "${RUN}_best.pt"
put "${FINAL_PATH}" -o "${RUN}.pt"
put "${LOG_PATH}" -o "${RUN}.log"

bye
EOF

echo "============================================================"
echo "[completed]"
echo "TASK       = VQW + BPE auxiliary -> VQW -> frozen decoder -> BPE"
echo "DATA       = ${DATA}"
echo "CODEBOOK   = ${CODEBOOK}"
echo "DICTIONARY = ${DICTIONARY}"
echo "BEST       = vqword_logs/${RUN}_best.pt"
echo "FINAL      = vqword_logs/${RUN}.pt"
echo "LOG        = vqword_logs/${RUN}.log"
echo "============================================================"
