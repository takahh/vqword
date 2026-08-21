#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y lftp

pip install \
  torch \
  datasets \
  transformers \
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

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {local_clusters}"
  echo "Example: $0 5"
  exit 1
fi

LOCAL_CLUSTERS="$1"
if ! [[ "${LOCAL_CLUSTERS}" =~ ^[0-9]+$ ]] || [ "${LOCAL_CLUSTERS}" -lt 1 ]; then
  echo "[error] local_clusters must be a positive integer"
  exit 1
fi

# ============================================================
# fixed settings
# ============================================================
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

# One shared-GNN checkpoint containing separate HOP1..10 BPE-local codebooks.
VQ_CKPT="wikitext103_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop01to10_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}.pt"
VQ_CKPT_PATH="/vqword/${VQ_CKPT}"

OUT_PATTERN="/vqword/tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop{hop02}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"

# ============================================================
# tokenizer: download once
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
get "${BPE_ARCHIVE}" -o "/vqword/${BPE_ARCHIVE}"
bye
EOF

tar -xzf "/vqword/${BPE_ARCHIVE}" -C /vqword

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] tokenizer directory was not created: ${TOKENIZER_DIR}"
  exit 1
fi
if [ ! -f "${ASSIGN_SCRIPT}" ]; then
  echo "[error] assignment script not found: ${ASSIGN_SCRIPT}"
  exit 1
fi

# ============================================================
# shared checkpoint: download ONCE
# ============================================================
rm -f "${VQ_CKPT_PATH}"

echo "============================================================"
echo "[download tied-GNN HOP1..10 checkpoint]"
echo "${VQ_CKPT}"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${VQ_CKPT}" -o "${VQ_CKPT_PATH}"
bye
EOF

if [ ! -f "${VQ_CKPT_PATH}" ]; then
  echo "[error] checkpoint not found: ${VQ_CKPT_PATH}"
  exit 1
fi

# ============================================================
# checkpoint schema check
# ============================================================
python - <<PY
import torch
p = "${VQ_CKPT_PATH}"
c = torch.load(p, map_location="cpu", weights_only=False)
required = {
    "model", "args", "centers_by_bpe_by_hop", "k_by_bpe_by_hop",
    "hops", "max_local_clusters", "partition_type",
    "shared_gnn_across_hops", "shared_codebook_across_hops",
}
missing = sorted(required - set(c))
if missing:
    raise ValueError(f"missing checkpoint keys: {missing}")
if [int(h) for h in c["hops"]] != list(range(1, 11)):
    raise ValueError(f"expected HOP1..10, got {c['hops']}")
if c["shared_gnn_across_hops"] is not True:
    raise ValueError("shared_gnn_across_hops must be True")
if c["shared_codebook_across_hops"] is not False:
    raise ValueError("shared_codebook_across_hops must be False")
if c["partition_type"] != "bpe_local_kmeans":
    raise ValueError(f"unexpected partition_type={c['partition_type']}")
if int(c["max_local_clusters"]) != int("${LOCAL_CLUSTERS}"):
    raise ValueError(
        f"local cluster mismatch: ckpt={c['max_local_clusters']} "
        f"requested=${LOCAL_CLUSTERS}"
    )
if int(c["model"]["tok_emb.weight"].shape[0]) != int("${BPE_VOCAB_LABEL}"):
    raise ValueError("BPE vocab size mismatch")
print("[checkpoint check] OK")
print("hops:", c["hops"])
print("shared_gnn_across_hops:", c["shared_gnn_across_hops"])
print("shared_codebook_across_hops:", c["shared_codebook_across_hops"])
print("max_local_clusters:", c["max_local_clusters"])
print("k_by_bpe_by_hop shape:", tuple(c["k_by_bpe_by_hop"].shape))
PY

# Remove old outputs before assigning.
for HOP in $(seq 1 10); do
  HOP2=$(printf "%02d" "${HOP}")
  OUT="tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop${HOP2}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"
  rm -f "/vqword/${OUT}" "/vqword/${OUT}.part"*
done

# ============================================================
# TinyStories: tokenize once; assign HOP1..10 from same checkpoint
# ============================================================
python "${ASSIGN_SCRIPT}" \
  --ckpt "${VQ_CKPT_PATH}" \
  --dataset roneneldan/TinyStories \
  --split train \
  --text_col text \
  --max_samples "${MAX_SAMPLES}" \
  --seq_len "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --hops 1 2 3 4 5 6 7 8 9 10 \
  --missing_bpe_policy zero \
  --out_pattern "${OUT_PATTERN}"

# ============================================================
# Validate and upload all 10 outputs
# ============================================================
for HOP in $(seq 1 10); do
  HOP2=$(printf "%02d" "${HOP}")
  OUT="tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop${HOP2}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"
  OUT_PATH="/vqword/${OUT}"

  python - <<PY
import torch
p = "${OUT_PATH}"
d = torch.load(p, map_location="cpu", weights_only=False)
expected_hop = int("${HOP}")
expected_k = int("${LOCAL_CLUSTERS}")
required = {
    "samples", "vq_ids_flat", "local_vq_ids_flat", "token_ids_flat",
    "k_by_bpe", "vq_vocab_size", "vq_pad_id", "hop",
    "partition_type", "id_scheme", "shared_gnn_across_hops",
    "shared_codebook_across_hops",
}
missing = sorted(required - set(d))
if missing:
    raise ValueError(f"HOP{expected_hop}: missing output keys: {missing}")
if int(d["hop"]) != expected_hop:
    raise ValueError(f"HOP metadata mismatch: {d['hop']} != {expected_hop}")
if int(d["vq_vocab_size"]) != expected_k or int(d["vq_pad_id"]) != expected_k:
    raise ValueError("VQ vocabulary/pad mismatch")
if d["partition_type"] != "bpe_local_kmeans":
    raise ValueError("partition_type mismatch")
if d["id_scheme"] != "(bpe_id, local_vq_id)":
    raise ValueError("id_scheme mismatch")
if d["shared_gnn_across_hops"] is not True:
    raise ValueError("shared_gnn_across_hops mismatch")
if d["shared_codebook_across_hops"] is not False:
    raise ValueError("shared_codebook_across_hops mismatch")
tok = d["token_ids_flat"].long().reshape(-1)
ids = d["local_vq_ids_flat"].long().reshape(-1)
if not torch.equal(ids, d["vq_ids_flat"].long().reshape(-1)):
    raise ValueError("vq_ids_flat/local_vq_ids_flat mismatch")
if tok.numel() != ids.numel():
    raise ValueError("token/VQ length mismatch")
k = d["k_by_bpe"].long()
token_k = k[tok]
has = token_k > 0
if has.any() and (ids[has] >= token_k[has]).any():
    raise ValueError("local ID exceeds BPE-specific k")
if (~has).any() and not torch.all(ids[~has] == 0):
    raise ValueError("missing-center BPE must map to local ID 0")
print(f"[output check HOP{expected_hop}] OK tokens={tok.numel():,} local=[{int(ids.min())},{int(ids.max())}]")
PY

  FILE_SIZE=$(stat -c%s "${OUT_PATH}")
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
    split -b 450M -d -a 3 "${OUT_PATH}" "${OUT_PATH}.part"
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
done

echo "============================================================"
echo "[all completed] tied GNN / separate HOP codebooks / HOP1..10"
echo "checkpoint downloaded once: ${VQ_CKPT}"
echo "local_clusters=${LOCAL_CLUSTERS}"
echo "============================================================"

rm -f "${VQ_CKPT_PATH}"
