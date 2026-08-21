```
#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y lftp
pip install torch datasets transformers tqdm numpy

cd /
if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi

cd /vqword
git pull

# ============================================================
# Settings
# ============================================================

BPE_VOCAB_LABEL=50257

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {local_clusters}"
  echo "Example: FTP_PASS='...' $0 5"
  exit 1
fi

LOCAL_CLUSTERS="$1"

if ! [[ "${LOCAL_CLUSTERS}" =~ ^[0-9]+$ ]] || [ "${LOCAL_CLUSTERS}" -lt 1 ]; then
  echo "[error] local_clusters must be a positive integer"
  exit 1
fi

DISCRETIZATION_SEED=0
MAX_SAMPLES=20000
SEQ_LEN=256
BATCH_SIZE=512

BPE_ARCHIVE="bpe_wikitext103_50257.tar.gz"
TOKENIZER_DIR="/vqword/bpe_wikitext103_50257"
ASSIGN_SCRIPT="/vqword/assign_vqword_ids.py"

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

CKPT_PREFIX="wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop01to10_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}"

SHARED_MODEL="${CKPT_PREFIX}_shared_model.pt"
SHARED_MODEL_PATH="/vqword/${SHARED_MODEL}"

OUT_PATTERN="/vqword/tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop{hop02}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"

# ============================================================
# Download tokenizer
# ============================================================

rm -f "/vqword/${BPE_ARCHIVE}"
rm -rf "${TOKENIZER_DIR}"

echo "============================================================"
echo "[download tokenizer] ${BPE_ARCHIVE}"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${BPE_ARCHIVE}" -o "/vqword/${BPE_ARCHIVE}"
bye
EOF

tar -xzf "/vqword/${BPE_ARCHIVE}" -C /vqword

[ -d "${TOKENIZER_DIR}" ] || {
  echo "[error] tokenizer missing: ${TOKENIZER_DIR}"
  exit 1
}

[ -f "${ASSIGN_SCRIPT}" ] || {
  echo "[error] assignment script missing: ${ASSIGN_SCRIPT}"
  exit 1
}

# ============================================================
# Shared model
#
# The original generator does not train the GNN before clustering.
# Therefore the tied-GNN weights can be deterministically rebuilt from
# the seed/config stored in the HOP-part signature.  HOP1 is downloaded
# first in the loop below and used to rebuild the one shared model.
# ============================================================

REBUILD_SHARED_SCRIPT="/vqword/rebuild_shared_model_from_hop.py"
[ -f "${REBUILD_SHARED_SCRIPT}" ] || {
  echo "[error] missing helper: ${REBUILD_SHARED_SCRIPT}"
  exit 1
}
rm -f "${SHARED_MODEL_PATH}"

# ============================================================
# HOP1..10
#
# Each HOP file is downloaded, checked, used, then deleted.
#
# Expected HOP-file structure:
#   signature
#   hop
#   local_vq_ids
#   centers_by_bpe
#   k_by_bpe
# ============================================================

for HOP in $(seq 1 10); do

  HOP2=$(printf "%02d" "${HOP}")
  HOP3=$(printf "%03d" "${HOP}")

  VQ_FILE="${CKPT_PREFIX}_hop_${HOP3}_ids.pt"
  VQ_FILE_PATH="/vqword/${VQ_FILE}"

  OUT="tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop${HOP2}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"
  OUT_PATH="/vqword/${OUT}"

  rm -f \
    "${VQ_FILE_PATH}" \
    "${OUT_PATH}" \
    "${OUT_PATH}.part"*

  echo
  echo "============================================================"
  echo "[HOP${HOP}] download: ${VQ_FILE}"
  echo "============================================================"

  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${VQ_FILE}" -o "${VQ_FILE_PATH}"
bye
EOF

  # Rebuild the one common tied-GNN model from HOP1 metadata.
  # No large merged checkpoint and no shared-model FTP file is required.
  if [ "${HOP}" -eq 1 ]; then
    echo "============================================================"
    echo "[HOP1] rebuild shared tied-GNN model"
    echo "============================================================"
    python "${REBUILD_SHARED_SCRIPT}" \
      --hop_file "${VQ_FILE_PATH}" \
      --out "${SHARED_MODEL_PATH}"

    [ -f "${SHARED_MODEL_PATH}" ] || {
      echo "[error] shared model reconstruction failed"
      exit 1
    }
  fi

  # ==========================================================
  # Validate downloaded per-HOP file
  # ==========================================================

  python - "${VQ_FILE_PATH}" "${HOP}" "${LOCAL_CLUSTERS}" <<'PY'
import sys
import torch

path = sys.argv[1]
expected_hop = int(sys.argv[2])
expected_k = int(sys.argv[3])

obj = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

if not isinstance(obj, dict):
    raise ValueError(
        f"HOP{expected_hop}: expected dict, got {type(obj)}"
    )

required = {
    "signature",
    "hop",
    "local_vq_ids",
    "centers_by_bpe",
    "k_by_bpe",
}

missing = required - set(obj.keys())

if missing:
    raise ValueError(
        f"HOP{expected_hop}: missing HOP-file keys: "
        f"{sorted(missing)}"
    )

actual_hop = int(obj["hop"])

if actual_hop != expected_hop:
    raise ValueError(
        f"HOP mismatch: expected {expected_hop}, "
        f"got {actual_hop}"
    )

k = obj["k_by_bpe"]

if not torch.is_tensor(k):
    k = torch.as_tensor(k)

k = k.long().reshape(-1)

if k.numel() == 0:
    raise ValueError(
        f"HOP{expected_hop}: empty k_by_bpe"
    )

if int(k.max()) > expected_k:
    raise ValueError(
        f"HOP{expected_hop}: k_by_bpe max={int(k.max())} "
        f"> local_clusters={expected_k}"
    )

centers = obj["centers_by_bpe"]

try:
    n_centers = len(centers)
except TypeError:
    n_centers = -1

local_ids = obj["local_vq_ids"]

if torch.is_tensor(local_ids):
    num_ids = local_ids.numel()
    id_min = int(local_ids.min()) if num_ids else None
    id_max = int(local_ids.max()) if num_ids else None
else:
    num_ids = len(local_ids)
    id_min = None
    id_max = None

print(
    f"[HOP-file check HOP{expected_hop}] OK"
)
print(
    f"  keys={list(obj.keys())}"
)
print(
    f"  hop={actual_hop}"
)
print(
    f"  k_by_bpe={k.numel():,} entries, "
    f"max_k={int(k.max())}"
)
print(
    f"  centers_by_bpe entries={n_centers}"
)
print(
    f"  local_vq_ids={num_ids:,}, "
    f"range=[{id_min},{id_max}]"
)
PY

  # ==========================================================
  # IMPORTANT:
  #
  # The downloaded file is a per-HOP ID/codebook file, not the
  # old combined GNN checkpoint.
  #
  # assign_vqword_ids.py must therefore support this file format:
  #
  #   signature
  #   hop
  #   local_vq_ids
  #   centers_by_bpe
  #   k_by_bpe
  #
  # If assign_vqword_ids.py still expects:
  #
  #   model
  #   args
  #   hops
  #   centers_by_bpe_by_hop
  #
  # it must be updated separately.
  # ==========================================================

  echo
  echo "============================================================"
  echo "[HOP${HOP}] assign TinyStories"
  echo "============================================================"

  python "${ASSIGN_SCRIPT}" \
    --model_ckpt "${SHARED_MODEL_PATH}" \
    --codebook_ckpt "${VQ_FILE_PATH}" \
    --dataset roneneldan/TinyStories \
    --split train \
    --text_col text \
    --max_samples "${MAX_SAMPLES}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --hops "${HOP}" \
    --missing_bpe_policy zero \
    --out_pattern "${OUT_PATTERN}"

  # ==========================================================
  # Validate generated TinyStories file
  # ==========================================================

  if [ ! -f "${OUT_PATH}" ]; then
    echo "[error] output was not generated:"
    echo "  ${OUT_PATH}"
    exit 1
  fi

  python - "${OUT_PATH}" "${HOP}" "${LOCAL_CLUSTERS}" <<'PY'
import sys
import torch

path = sys.argv[1]
expected_hop = int(sys.argv[2])
expected_k = int(sys.argv[3])

d = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

if not isinstance(d, dict):
    raise ValueError(
        f"output must be dict, got {type(d)}"
    )

required = {
    "hop",
    "vq_vocab_size",
    "vq_pad_id",
    "token_ids_flat",
    "local_vq_ids_flat",
    "vq_ids_flat",
    "k_by_bpe",
}

missing = required - set(d.keys())

if missing:
    raise ValueError(
        f"output missing keys: {sorted(missing)}"
    )

if int(d["hop"]) != expected_hop:
    raise ValueError(
        f"HOP metadata mismatch: "
        f"{d['hop']} != {expected_hop}"
    )

if int(d["vq_vocab_size"]) != expected_k:
    raise ValueError(
        f"vq_vocab_size mismatch: "
        f"{d['vq_vocab_size']} != {expected_k}"
    )

if int(d["vq_pad_id"]) != expected_k:
    raise ValueError(
        f"vq_pad_id mismatch: "
        f"{d['vq_pad_id']} != {expected_k}"
    )

tok = d["token_ids_flat"].long().reshape(-1)
ids = d["local_vq_ids_flat"].long().reshape(-1)
vq_ids = d["vq_ids_flat"].long().reshape(-1)

if tok.numel() != ids.numel():
    raise ValueError(
        "token/local-VQ length mismatch"
    )

if not torch.equal(ids, vq_ids):
    raise ValueError(
        "local_vq_ids_flat != vq_ids_flat"
    )

k = d["k_by_bpe"]

if not torch.is_tensor(k):
    k = torch.as_tensor(k)

k = k.long().reshape(-1)

if tok.numel():
    if int(tok.max()) >= k.numel():
        raise ValueError(
            f"BPE ID exceeds k_by_bpe size: "
            f"max token={int(tok.max())}, "
            f"k size={k.numel()}"
        )

    token_k = k[tok]
    has = token_k > 0

    if has.any():
        if (ids[has] < 0).any():
            raise ValueError(
                "negative local VQ ID"
            )

        if (ids[has] >= token_k[has]).any():
            raise ValueError(
                "local ID exceeds BPE-specific k"
            )

    if (~has).any():
        if not torch.all(ids[~has] == 0):
            raise ValueError(
                "missing-center BPE must map to local ID 0"
            )

print(
    f"[output check HOP{expected_hop}] OK "
    f"tokens={tok.numel():,} "
    f"local=[{int(ids.min())},{int(ids.max())}]"
)
PY

  # ==========================================================
  # Upload
  # ==========================================================

  FILE_SIZE=$(stat -c%s "${OUT_PATH}")

  echo
  echo "[HOP${HOP}] output size: ${FILE_SIZE} bytes"

  if [ "${FILE_SIZE}" -lt 1800000000 ]; then

    lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
put "${OUT_PATH}" -o "${OUT}"
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
set net:timeout 60
set cmd:fail-exit yes
mput "${OUT_PATH}.part"*
bye
EOF

  fi

  echo "[uploaded HOP${HOP}] ${OUT}"

  # Remove downloaded WikiText HOP file before next HOP.
  rm -f "${VQ_FILE_PATH}"

done

echo
echo "============================================================"
echo "[all completed]"
echo "HOP1..10 downloaded and processed separately"
echo "local_clusters=${LOCAL_CLUSTERS}"
echo "============================================================"
```