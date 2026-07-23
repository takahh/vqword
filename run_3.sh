
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
# Step4: WikiText-103でVQWord tokenizer作成
#
# BPE tokenizer:
#   bpe_wikitext103_50257.tar.gz
#
# VQW:
#   left 20
#   center_scale = 0
#   codebook = 200k
# ============================================================
FTP_PASS='Squat#201k'
# BPEは固定
BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257

# VQW codebookだけ引数で変更
CB_SIZE="${1:-}"

case "${CB_SIZE}" in
  25k)
    VQ_CODEBOOK_LABEL=25k
    VQ_CODEBOOK_SIZE=25000
    ;;

  50k)
    VQ_CODEBOOK_LABEL=50k
    VQ_CODEBOOK_SIZE=50000
    ;;

  100k)
    VQ_CODEBOOK_LABEL=100k
    VQ_CODEBOOK_SIZE=100000
    ;;

  *)
    echo "Usage: $0 {25k|50k|100k}"
    exit 1
    ;;
esac

HOP=20
CENTER_SCALE=0.0
IVF_NLIST=256
SEED=0

D_MODEL=256
N_LAYERS=3

# ============================================================
# ファイル名
# ============================================================
BPE_ARCHIVE="bpe_wikitext103_${BPE_VOCAB_LABEL}.tar.gz"
BPE_ARCHIVE_PATH="/vqword/${BPE_ARCHIVE}"

TOKENIZER_DIR="/vqword/bpe_wikitext103_${BPE_VOCAB_LABEL}"

TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center0_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}_seed${SEED}"

OUT="wikitext103_vqword_${TAG}.pt"
DICTIONARY="wikitext103_vqword_${TAG}_dictionary.pt"
IDS="wikitext103_vqword_${TAG}_ids.pt"

OUT_PATH="/vqword/${OUT}"
DICTIONARY_PATH="/vqword/${DICTIONARY}"
IDS_PATH="/vqword/${IDS}"

# ============================================================
# FTP設定
# ============================================================

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

echo "============================================================"
echo "[configuration]"
echo "BPE vocabulary       = ${BPE_VOCAB_SIZE}"
echo "BPE archive          = ${BPE_ARCHIVE}"
echo "tokenizer directory  = ${TOKENIZER_DIR}"
echo "VQW codebook         = ${VQ_CODEBOOK_SIZE}"
echo "context              = left-only"
echo "hop                  = ${HOP}"
echo "center scale         = ${CENTER_SCALE}"
echo "IVF nlist            = ${IVF_NLIST}"
echo "seed                 = ${SEED}"
echo "output               = ${OUT}"
echo "dictionary           = ${DICTIONARY}"
echo "ids                  = ${IDS}"
echo "============================================================"

# ============================================================
# BPE tokenizer archiveをFTPから取得
# ============================================================

rm -f "${BPE_ARCHIVE_PATH}"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${BPE_ARCHIVE}" \
  -o "${BPE_ARCHIVE_PATH}"

bye
EOF

echo "============================================================"
echo "[downloaded BPE tokenizer archive]"
ls -lh "${BPE_ARCHIVE_PATH}"
echo "============================================================"

# ============================================================
# tokenizerを展開
# ============================================================

rm -rf "${TOKENIZER_DIR}"

tar -xzf "${BPE_ARCHIVE_PATH}" -C /vqword

echo "============================================================"
echo "[extracted tokenizer]"
echo "directory = ${TOKENIZER_DIR}"
echo "============================================================"

if [ ! -d "${TOKENIZER_DIR}" ]; then
  echo "[error] tokenizer directory was not created:"
  echo "        ${TOKENIZER_DIR}"
  echo
  echo "[archive contents]"
  tar -tzf "${BPE_ARCHIVE_PATH}" | head -50
  exit 1
fi

# AutoTokenizerが最低限必要とするファイルを確認
for file in \
  tokenizer.json \
  tokenizer_config.json
do
  if [ ! -f "${TOKENIZER_DIR}/${file}" ]; then
    echo "[error] Missing tokenizer file:"
    echo "        ${TOKENIZER_DIR}/${file}"
    exit 1
  fi
done

ls -lh "${TOKENIZER_DIR}"

# ============================================================
# tokenizerの語彙数を検証
# ============================================================

python - <<PY
from transformers import AutoTokenizer

path = "${TOKENIZER_DIR}"
expected_vocab_size = ${BPE_VOCAB_SIZE}

tok = AutoTokenizer.from_pretrained(path)

print("============================================================")
print("[tokenizer verification]")
print("path:", path)
print("tok.vocab_size:", tok.vocab_size)
print("len(tok):", len(tok))
print("pad_token:", tok.pad_token)
print("pad_token_id:", tok.pad_token_id)
print("unk_token:", tok.unk_token)
print("unk_token_id:", tok.unk_token_id)
print("bos_token:", tok.bos_token)
print("bos_token_id:", tok.bos_token_id)
print("eos_token:", tok.eos_token)
print("eos_token_id:", tok.eos_token_id)

if tok.vocab_size != expected_vocab_size:
    raise ValueError(
        "BPE vocabulary mismatch: "
        f"expected={expected_vocab_size:,}, "
        f"actual={tok.vocab_size:,}"
    )

print("[check] OK")
print("============================================================")
PY

# ============================================================
# WikiText-103でVQWord tokenizerを作成
#
# 注意:
# 現在のtrain_vqword.pyは、Step2のBPE IDファイルを読まず、
# WikiText-103をここで再取得・再tokenizeする。
# ============================================================

echo "============================================================"
echo "[train VQWord]"
echo "dataset    = WikiText-103"
echo "tokenizer  = ${TOKENIZER_DIR}"
echo "context    = past ${HOP} tokens + center"
echo "codebook   = ${VQ_CODEBOOK_SIZE}"
echo "============================================================"

python train_vqword.py \
  --dataset Salesforce/wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --text_col text \
  --tokenizer "${TOKENIZER_DIR}" \
  --max_samples 1000000 \
  --seq_len 256 \
  --hop "${HOP}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --center_scale "${CENTER_SCALE}" \
  --ivf_nlist "${IVF_NLIST}" \
  --ivf_iters 1 \
  --ivf_batch_size 8192 \
  --global_codebook_size "${VQ_CODEBOOK_SIZE}" \
  --global_kmeans_iters 5 \
  --global_batch_size 8192 \
  --batch_size 1024 \
  --k_block 4096 \
  --seed "${SEED}" \
  --out "${OUT_PATH}"

# ============================================================
# 生成物を確認
# ============================================================

echo "============================================================"
echo "[generated files]"
echo "============================================================"

for path in \
  "${OUT_PATH}" \
  "${DICTIONARY_PATH}" \
  "${IDS_PATH}"
do
  if [ ! -f "${path}" ]; then
    echo "[error] Expected output was not generated:"
    echo "        ${path}"
    exit 1
  fi
done

ls -lh \
  "${OUT_PATH}" \
  "${DICTIONARY_PATH}" \
  "${IDS_PATH}"

# ============================================================
# コードブックと辞書をFTPへアップロード
# ============================================================

echo "============================================================"
echo "[upload codebook and dictionary]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

put "${OUT_PATH}" \
  -o "${OUT}"

put "${DICTIONARY_PATH}" \
  -o "${DICTIONARY}"

bye
EOF

# ============================================================
# WikiText VQW IDファイルを分割してアップロード
# ============================================================

echo "============================================================"
echo "[split WikiText-103 VQW ID file]"
echo "============================================================"

rm -f "${IDS_PATH}.part"*

split \
  -b 450M \
  -d \
  -a 3 \
  "${IDS_PATH}" \
  "${IDS_PATH}.part"

ls -lh "${IDS_PATH}.part"*

echo "============================================================"
echo "[upload split ID files]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

mput "${IDS_PATH}.part"*

bye
EOF

echo "============================================================"
echo "[completed]"
echo "BPE tokenizer = ${BPE_ARCHIVE}"
echo "BPE vocabulary = ${BPE_VOCAB_SIZE}"
echo "VQW codebook = ${VQ_CODEBOOK_SIZE}"
echo "context = left ${HOP}"
echo "seed = ${SEED}"
echo "codebook = ${OUT}"
echo "dictionary = ${DICTIONARY}"
echo "IDs = ${IDS}"
echo "============================================================"
