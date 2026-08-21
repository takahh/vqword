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
OUT_PATTERN="/vqword/tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop{hop02}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"

rm -f "/vqword/${BPE_ARCHIVE}"
rm -rf "${TOKENIZER_DIR}"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${BPE_ARCHIVE}" -o "/vqword/${BPE_ARCHIVE}"
bye
EOF

tar -xzf "/vqword/${BPE_ARCHIVE}" -C /vqword
[ -d "${TOKENIZER_DIR}" ] || { echo "[error] tokenizer missing: ${TOKENIZER_DIR}"; exit 1; }
[ -f "${ASSIGN_SCRIPT}" ] || { echo "[error] assignment script missing: ${ASSIGN_SCRIPT}"; exit 1; }

# Each HOP file is downloaded, checked, used, and removed before the next HOP.
# The broken approximately-2-GiB combined checkpoint is never downloaded.
for HOP in $(seq 1 10); do
  HOP2=$(printf "%02d" "${HOP}")
  HOP3=$(printf "%03d" "${HOP}")
  VQ_CKPT="${CKPT_PREFIX}_hop_${HOP3}_ids.pt"
  VQ_CKPT_PATH="/vqword/${VQ_CKPT}"
  OUT="tinystories_vqword_bpe${BPE_VOCAB_LABEL}_tiedgnn_separatehop${HOP2}_center0_localbpe${LOCAL_CLUSTERS}_seed${DISCRETIZATION_SEED}_ids.pt"
  OUT_PATH="/vqword/${OUT}"

  rm -f "${VQ_CKPT_PATH}" "${OUT_PATH}" "${OUT_PATH}.part"*

  echo "============================================================"
  echo "[HOP${HOP}] download: ${VQ_CKPT}"
  echo "============================================================"
  lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 60
set cmd:fail-exit yes
get "${VQ_CKPT}" -o "${VQ_CKPT_PATH}"
bye
EOF

  python - "${VQ_CKPT_PATH}" "${HOP}" "${LOCAL_CLUSTERS}" "${BPE_VOCAB_LABEL}" <<'PY'
import sys, torch
p, expected_hop, expected_k, expected_vocab = sys.argv[1], *map(int, sys.argv[2:])
c = torch.load(p, map_location="cpu", weights_only=False)
required = {"model", "args", "centers_by_bpe_by_hop", "k_by_bpe_by_hop", "hops",
            "max_local_clusters", "partition_type", "shared_gnn_across_hops",
            "shared_codebook_across_hops"}
missing = sorted(required - set(c))
if missing:
    raise ValueError(f"HOP{expected_hop}: missing checkpoint keys: {missing}")
hops = [int(h) for h in c["hops"]]
if hops != [expected_hop]:
    raise ValueError(f"expected only HOP{expected_hop}, got {hops}")
if c["shared_gnn_across_hops"] is not True or c["shared_codebook_across_hops"] is not False:
    raise ValueError("shared-GNN/separate-codebook metadata mismatch")
if c["partition_type"] != "bpe_local_kmeans":
    raise ValueError(f"unexpected partition_type={c['partition_type']}")
if int(c["max_local_clusters"]) != expected_k:
    raise ValueError("local cluster mismatch")
if int(c["model"]["tok_emb.weight"].shape[0]) != expected_vocab:
    raise ValueError("BPE vocab size mismatch")
print(f"[checkpoint check HOP{expected_hop}] OK")
PY

  # A one-HOP checkpoint is assigned with a one-element --hops list.
  python "${ASSIGN_SCRIPT}" \
    --ckpt "${VQ_CKPT_PATH}" \
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

  python - "${OUT_PATH}" "${HOP}" "${LOCAL_CLUSTERS}" <<'PY'
import sys, torch
p, expected_hop, expected_k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = torch.load(p, map_location="cpu", weights_only=False)
if int(d["hop"]) != expected_hop:
    raise ValueError(f"HOP metadata mismatch: {d['hop']} != {expected_hop}")
if int(d["vq_vocab_size"]) != expected_k or int(d["vq_pad_id"]) != expected_k:
    raise ValueError("VQ vocabulary/pad mismatch")
tok = d["token_ids_flat"].long().reshape(-1)
ids = d["local_vq_ids_flat"].long().reshape(-1)
if not torch.equal(ids, d["vq_ids_flat"].long().reshape(-1)) or tok.numel() != ids.numel():
    raise ValueError("token/local-VQ arrays mismatch")
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
  rm -f "${VQ_CKPT_PATH}"
done

echo "============================================================"
echo "[all completed] HOP1..10 downloaded and processed separately"
echo "local_clusters=${LOCAL_CLUSTERS}"
echo "============================================================"
