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
  echo "Example: $0 100k 1"
  exit 1
fi

VQ_CODEBOOK_LABEL="$1"
AR_SEED="$2"
if [ "${VQ_CODEBOOK_LABEL}" != "100k" ]; then
  echo "[error] Expected VQ label 100k, got ${VQ_CODEBOOK_LABEL}"
  exit 1
fi
if ! [[ "${AR_SEED}" =~ ^[0-9]+$ ]]; then
  echo "[error] ar_seed must be a non-negative integer"
  exit 1
fi

CENTER_SCALE="${CENTER_SCALE:-1}"
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
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_LEN="${MAX_LEN:-255}"
CONVERTER_LOSS_WEIGHT="${CONVERTER_LOSS_WEIGHT:-1.0}"

AR_SCRIPT="/vqword/ar_sc0_to_sc0_with_sc1_converter.py"
if [ ! -f "${AR_SCRIPT}" ]; then
  echo "[error] AR script was not found: ${AR_SCRIPT}"
  exit 1
fi

make_tag() {
  local scale2="$1"
  echo "bpe${BPE_VOCAB_LABEL}_bilateral${scale2}_center${CENTER_SCALE}_${MODEL_VARIANT}_dec${DECODER_EPOCHS}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${DISCRETIZATION_SEED}"
}

SC0_TAG="$(make_tag 00)"
SC1_TAG="$(make_tag 01)"

SC0_DATA="tinystories_vqword_${SC0_TAG}_ids.pt"
SC1_DATA="tinystories_vqword_${SC1_TAG}_ids.pt"
SC0_CODEBOOK="wikitext103_vqword_${SC0_TAG}.pt"
SC1_CODEBOOK="wikitext103_vqword_${SC1_TAG}.pt"
SC0_DICTIONARY="wikitext103_vqword_${SC0_TAG}_dictionary.pt"

SC0_DATA_PATH="/vqword/${SC0_DATA}"
SC1_DATA_PATH="/vqword/${SC1_DATA}"
SC0_CODEBOOK_PATH="/vqword/${SC0_CODEBOOK}"
SC1_CODEBOOK_PATH="/vqword/${SC1_CODEBOOK}"
SC0_DICTIONARY_PATH="/vqword/${SC0_DICTIONARY}"

RUN="ar_sc0_to_sc0_with_sc1_converter_bpe${BPE_VOCAB_LABEL}_center${CENTER_SCALE}_vqcb${VQ_CODEBOOK_LABEL}_arseed${AR_SEED}_conv${CONVERTER_LOSS_WEIGHT}_$(date +%Y%m%d_%H%M%S)"
FINAL_PATH="/vqword/${RUN}.pt"
BEST_PATH="/vqword/${RUN}_best.pt"
LOG_PATH="/vqword/${RUN}.log"

printf '%s\n' \
  "============================================================" \
  "[configuration]" \
  "task                  = past sc0 -> next sc0 + aligned sc1" \
  "sc0 data              = ${SC0_DATA}" \
  "sc1 data              = ${SC1_DATA}" \
  "sc0 codebook          = ${SC0_CODEBOOK}" \
  "sc1 codebook          = ${SC1_CODEBOOK}" \
  "sc0 dictionary        = ${SC0_DICTIONARY}" \
  "converter loss weight = ${CONVERTER_LOSS_WEIGHT}" \
  "AR seed               = ${AR_SEED}" \
  "============================================================"

rm -f "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"

download_file() {
  local remote="$1"
  local local_path="$2"
  if [ -s "${local_path}" ]; then
    echo "[reuse] ${local_path}"
    return
  fi
  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${remote}" -o "${local_path}"
bye
EOF
}

download_file "${SC0_DATA}" "${SC0_DATA_PATH}"
download_file "${SC1_DATA}" "${SC1_DATA_PATH}"
download_file "${SC0_CODEBOOK}" "${SC0_CODEBOOK_PATH}"
download_file "${SC1_CODEBOOK}" "${SC1_CODEBOOK_PATH}"
download_file "${SC0_DICTIONARY}" "${SC0_DICTIONARY_PATH}"

for path in \
  "${SC0_DATA_PATH}" "${SC1_DATA_PATH}" \
  "${SC0_CODEBOOK_PATH}" "${SC1_CODEBOOK_PATH}" \
  "${SC0_DICTIONARY_PATH}"
do
  if [ ! -s "${path}" ]; then
    echo "[error] Missing or empty file: ${path}"
    exit 1
  fi
done

python - "${SC0_DATA_PATH}" "${SC1_DATA_PATH}" "${SC0_CODEBOOK_PATH}" "${SC1_CODEBOOK_PATH}" <<'PYVERIFY'
import sys
import torch

sc0_data_path, sc1_data_path, sc0_cb_path, sc1_cb_path = sys.argv[1:]
sc0 = torch.load(sc0_data_path, map_location="cpu", weights_only=False)
sc1 = torch.load(sc1_data_path, map_location="cpu", weights_only=False)
cb0 = torch.load(sc0_cb_path, map_location="cpu", weights_only=False)
cb1 = torch.load(sc1_cb_path, map_location="cpu", weights_only=False)

for name, data in (("sc0", sc0), ("sc1", sc1)):
    for key in ("samples", "token_ids_flat", "vq_ids_flat"):
        if key not in data:
            raise KeyError(f"{name} data missing {key}")

if not torch.equal(sc0["token_ids_flat"].reshape(-1), sc1["token_ids_flat"].reshape(-1)):
    raise ValueError("sc0/sc1 token_ids_flat mismatch")
if len(sc0["samples"]) != len(sc1["samples"]):
    raise ValueError("sc0/sc1 sample count mismatch")
for i, (a, b) in enumerate(zip(sc0["samples"], sc1["samples"])):
    for key in ("sample_idx", "start", "end", "length"):
        if int(a[key]) != int(b[key]):
            raise ValueError(f"sample metadata mismatch at {i}, key={key}")

for expected, data, cb in ((0, sc0, cb0), (1, sc1, cb1)):
    centers = cb.get("global_centers")
    if centers is None:
        raise KeyError(f"sc{expected} codebook missing global_centers")
    ids = data["vq_ids_flat"].long().reshape(-1)
    if int(ids.min()) < 0 or int(ids.max()) >= centers.size(0):
        raise ValueError(f"sc{expected} IDs out of range")
    print(f"[sc{expected}] centers={tuple(centers.shape)} used={torch.unique(ids).numel():,}")
print("[alignment verification OK]")
PYVERIFY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "${AR_SCRIPT}" \
  --sc0_data "${SC0_DATA_PATH}" \
  --sc1_data "${SC1_DATA_PATH}" \
  --sc0_codebook "${SC0_CODEBOOK_PATH}" \
  --sc1_codebook "${SC1_CODEBOOK_PATH}" \
  --sc0_dictionary "${SC0_DICTIONARY_PATH}" \
  --converter_loss_weight "${CONVERTER_LOSS_WEIGHT}" \
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
  --out "${FINAL_PATH}" \
  2>&1 | tee "${LOG_PATH}"

for path in "${FINAL_PATH}" "${BEST_PATH}" "${LOG_PATH}"; do
  if [ ! -s "${path}" ]; then
    echo "[error] Expected output was not created: ${path}"
    exit 1
  fi
done

python - "${BEST_PATH}" "${FINAL_PATH}" <<'PYCHECK'
import sys
import torch
for path in sys.argv[1:]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model", "args", "history", "sc0_vocab_size", "sc1_vocab_size",
        "sc0_pad_id", "token_vocab_size", "last_valid", "last_test",
    }
    missing = sorted(required - set(ckpt))
    if missing:
        raise KeyError(f"{path}: missing checkpoint keys: {missing}")
    state = ckpt["model"]
    for key in (
        "input_projection.weight", "pos_emb.weight",
        "sc0_head.weight", "sc0_head.bias",
        "sc1_head.weight", "sc1_head.bias",
    ):
        if key not in state:
            raise KeyError(f"{path}: missing model key {key}")
    print("[checkpoint OK]", path)
    print(" last valid:", ckpt["last_valid"])
    print(" last test :", ckpt["last_test"])
PYCHECK

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

printf '%s\n' \
  "============================================================" \
  "[completed]" \
  "TASK  = past sc0 -> next sc0 + aligned sc1" \
  "BEST  = vqword_logs/${RUN}_best.pt" \
  "FINAL = vqword_logs/${RUN}.pt" \
  "LOG   = vqword_logs/${RUN}.log" \
  "============================================================"
