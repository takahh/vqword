#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y lftp

pip install \
  torch \
  datasets \
  transformers \
  scikit-learn \
  tqdm \
  numpy

cd /

if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi

cd /vqword
git pull


# ============================================================
# arguments
# ============================================================

BPE_VOCAB_LABEL=50257

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {local_clusters} {hop}"
  echo
  echo "Example:"
  echo "  $0 5 10"
  exit 1
fi

LOCAL_CLUSTERS="$1"
HOP="$2"

if ! [[ "${LOCAL_CLUSTERS}" =~ ^[0-9]+$ ]]; then
  echo "[error] local_clusters must be positive integer"
  exit 1
fi

if [ "${LOCAL_CLUSTERS}" -lt 1 ]; then
  echo "[error] local_clusters must be >= 1"
  exit 1
fi

if ! [[ "${HOP}" =~ ^[0-9]+$ ]]; then
  echo "[error] hop must be non-negative integer"
  exit 1
fi


# ============================================================
# fixed settings
# ============================================================

CENTER_SCALE=0
DISCRETIZATION_SEED=0

MAX_SAMPLES=20000
SEQ_LEN=256
BATCH_SIZE=512
K_BLOCK=4096

BPE_ARCHIVE="bpe_wikitext103_50257.tar.gz"
TOKENIZER_DIR="/vqword/bpe_wikitext103_50257"

ASSIGN_SCRIPT="/vqword/assign_vqword_ids.py"

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"


# ============================================================
# BPE tokenizer
# ============================================================

rm -f "/vqword/${BPE_ARCHIVE}"
rm -rf "${TOKENIZER_DIR}"

echo "============================================================"
echo "[download BPE tokenizer]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${BPE_ARCHIVE}" \
  -o "/vqword/${BPE_ARCHIVE}"

bye
EOF

tar -xzf \
  "/vqword/${BPE_ARCHIVE}" \
  -C /vqword

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] Tokenizer directory was not created:"
  echo "        ${TOKENIZER_DIR}"
  exit 1
fi

if [ ! -f "${ASSIGN_SCRIPT}" ]; then
  echo "[error] Assignment script was not found:"
  echo "        ${ASSIGN_SCRIPT}"
  exit 1
fi


# ============================================================
# filenames
# ============================================================

HOP2=$(printf "%02d" "${HOP}")

BASE_TAG="bpe${BPE_VOCAB_LABEL}_bilateral${HOP2}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}"

VQ_CKPT="wikitext103_vqword_${BASE_TAG}.pt"
VQ_CKPT_PATH="/vqword/${VQ_CKPT}"

OUT="tinystories_vqword_${BASE_TAG}_ids.pt"
OUT_PATH="/vqword/${OUT}"


echo
echo "============================================================"
echo "[BPE-wise assignment]"
echo "checkpoint     = ${VQ_CKPT}"
echo "output         = ${OUT}"
echo "hop            = ${HOP}"
echo "local clusters = ${LOCAL_CLUSTERS}"
echo "============================================================"

rm -f "${VQ_CKPT_PATH}"
rm -f "${OUT_PATH}"
rm -f "${OUT_PATH}.part"*


# ============================================================
# checkpoint download
# ============================================================

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${VQ_CKPT}" \
  -o "${VQ_CKPT_PATH}"

bye
EOF

if [ ! -f "${VQ_CKPT_PATH}" ]; then
  echo "[error] checkpoint not found:"
  echo "        ${VQ_CKPT_PATH}"
  exit 1
fi


# ============================================================
# checkpoint check
# ============================================================

python - <<PY
import torch

path = "${VQ_CKPT_PATH}"

expected_hop = int("${HOP}")
expected_local_clusters = int("${LOCAL_CLUSTERS}")
expected_bpe_vocab = int("${BPE_VOCAB_LABEL}")

ckpt = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

required = {
    "model",
    "centers_by_bpe",
    "k_by_bpe",
    "args",
    "partition_type",
    "max_local_clusters",
    "id_scheme",
}

missing = sorted(required - set(ckpt.keys()))

if missing:
    raise ValueError(
        f"Missing checkpoint keys: {missing}"
    )

if ckpt["partition_type"] != "bpe_local_kmeans":
    raise ValueError(
        f"Unexpected partition_type: "
        f"{ckpt['partition_type']}"
    )

if ckpt["id_scheme"] != "(bpe_id, local_vq_id)":
    raise ValueError(
        f"Unexpected id_scheme: "
        f"{ckpt['id_scheme']}"
    )

actual_hop = int(
    ckpt.get(
        "hop",
        ckpt["args"]["hop"],
    )
)

actual_center_scale = float(
    ckpt["args"].get(
        "center_scale",
        -1.0,
    )
)

actual_local_clusters = int(
    ckpt["max_local_clusters"]
)

actual_bpe_vocab = int(
    ckpt["model"]["tok_emb.weight"].shape[0]
)

k_by_bpe = ckpt["k_by_bpe"].long()

if actual_hop != expected_hop:
    raise ValueError(
        f"HOP mismatch: "
        f"expected={expected_hop}, "
        f"actual={actual_hop}"
    )

if abs(actual_center_scale - 0.0) > 1e-8:
    raise ValueError(
        f"center_scale mismatch: "
        f"expected=0, "
        f"actual={actual_center_scale}"
    )

if actual_local_clusters != expected_local_clusters:
    raise ValueError(
        f"local cluster mismatch: "
        f"expected={expected_local_clusters}, "
        f"actual={actual_local_clusters}"
    )

if actual_bpe_vocab != expected_bpe_vocab:
    raise ValueError(
        f"BPE vocab mismatch: "
        f"expected={expected_bpe_vocab}, "
        f"actual={actual_bpe_vocab}"
    )

if k_by_bpe.numel() != expected_bpe_vocab:
    raise ValueError(
        f"k_by_bpe length mismatch: "
        f"{k_by_bpe.numel()} != "
        f"{expected_bpe_vocab}"
    )

if int(k_by_bpe.max()) > expected_local_clusters:
    raise ValueError(
        f"k_by_bpe max exceeds local cluster setting: "
        f"{int(k_by_bpe.max())} > "
        f"{expected_local_clusters}"
    )

print("[checkpoint check] OK")
print("hop:", actual_hop)
print("center_scale:", actual_center_scale)
print("max_local_clusters:", actual_local_clusters)
print("bpe_vocab_size:", actual_bpe_vocab)
print("BPEs with centers:", len(ckpt["centers_by_bpe"]))
print("k_by_bpe min/max:", int(k_by_bpe.min()), int(k_by_bpe.max()))
PY


# ============================================================
# TinyStories ID assignment
# ============================================================

python "${ASSIGN_SCRIPT}" \
  --ckpt "${VQ_CKPT_PATH}" \
  --dataset roneneldan/TinyStories \
  --split train \
  --text_col text \
  --max_samples "${MAX_SAMPLES}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --k_block "${K_BLOCK}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --missing_bpe_policy zero \
  --out "${OUT_PATH}"


# ============================================================
# output check
# ============================================================

python - <<PY
import torch

path = "${OUT_PATH}"

expected_hop = int("${HOP}")
expected_local_clusters = int("${LOCAL_CLUSTERS}")

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

required = {
    "samples",
    "vq_ids_flat",
    "local_vq_ids_flat",
    "token_ids_flat",
    "k_by_bpe",
    "vq_vocab_size",
    "vq_pad_id",
    "hop",
    "partition_type",
    "id_scheme",
}

missing = sorted(
    required - set(data.keys())
)

if missing:
    raise ValueError(
        f"Missing output keys: {missing}"
    )

if data["partition_type"] != "bpe_local_kmeans":
    raise ValueError(
        f"Unexpected partition_type: "
        f"{data['partition_type']}"
    )

if data["id_scheme"] != "(bpe_id, local_vq_id)":
    raise ValueError(
        f"Unexpected id_scheme: "
        f"{data['id_scheme']}"
    )

token_ids = (
    data["token_ids_flat"]
    .long()
    .reshape(-1)
)

local_ids = (
    data["local_vq_ids_flat"]
    .long()
    .reshape(-1)
)

vq_ids = (
    data["vq_ids_flat"]
    .long()
    .reshape(-1)
)

k_by_bpe = (
    data["k_by_bpe"]
    .long()
)

if token_ids.numel() != local_ids.numel():
    raise ValueError(
        f"Length mismatch: "
        f"token={token_ids.numel()}, "
        f"local_vq={local_ids.numel()}"
    )

if not torch.equal(
    local_ids,
    vq_ids,
):
    raise ValueError(
        "vq_ids_flat and local_vq_ids_flat differ"
    )

if int(data["hop"]) != expected_hop:
    raise ValueError(
        f"Output HOP mismatch: "
        f"expected={expected_hop}, "
        f"actual={data['hop']}"
    )

if int(data["vq_vocab_size"]) != expected_local_clusters:
    raise ValueError(
        f"VQ vocab mismatch: "
        f"expected={expected_local_clusters}, "
        f"actual={data['vq_vocab_size']}"
    )

if int(data["vq_pad_id"]) != expected_local_clusters:
    raise ValueError(
        f"VQ pad mismatch: "
        f"expected={expected_local_clusters}, "
        f"actual={data['vq_pad_id']}"
    )


# ------------------------------------------------------------
# Strong per-BPE validation
#
# k=0 means the source WikiText checkpoint had no centers for
# that BPE. Those are handled by missing_bpe_policy=zero.
# ------------------------------------------------------------

token_k = k_by_bpe[token_ids]

has_centers = token_k > 0

if has_centers.any():

    invalid = (
        local_ids[has_centers]
        >= token_k[has_centers]
    )

    if invalid.any():
        raise ValueError(
            "Found local VQ ID outside "
            "the BPE-specific cluster range"
        )


missing_mask = token_k == 0

if missing_mask.any():

    if not torch.all(
        local_ids[missing_mask] == 0
    ):
        raise ValueError(
            "Missing-center BPE must have "
            "local_vq_id=0"
        )


print("[output check] OK")

print(
    "samples:",
    len(data["samples"]),
)

print(
    "tokens:",
    token_ids.numel(),
)

print(
    "hop:",
    data["hop"],
)

print(
    "vq_vocab_size:",
    data["vq_vocab_size"],
)

print(
    "local VQ min/max:",
    int(local_ids.min()),
    int(local_ids.max()),
)

print(
    "missing BPE types:",
    len(data.get("missing_bpe_ids", [])),
)

print(
    "missing BPE token count:",
    int(missing_mask.sum()),
)
PY


# ============================================================
# FTP upload
# ============================================================

FILE_SIZE=$(stat -c%s "${OUT_PATH}")

if [ "${FILE_SIZE}" -lt 1800000000 ]; then

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

put "${OUT_PATH}" \
  -o "${OUT}"

bye
EOF

else

  split \
    -b 450M \
    -d \
    -a 3 \
    "${OUT_PATH}" \
    "${OUT_PATH}.part"

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

mput "${OUT_PATH}.part"*

bye
EOF

fi


echo
echo "============================================================"
echo "[completed]"
echo "local_clusters=${LOCAL_CLUSTERS}"
echo "HOP=${HOP}"
echo "${OUT}"
echo "============================================================"

rm -f "${VQ_CKPT_PATH}"