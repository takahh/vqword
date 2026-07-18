#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 使用方法
#
#   bash run_vqword_end_to_end.sh 200000 20
#
# 第1引数:
#   VQW codebook size
#
# 第2引数:
#   HOP数
#
# 例:
#   bash run_vqword_end_to_end.sh 50000 10
#   bash run_vqword_end_to_end.sh 100000 20
#   bash run_vqword_end_to_end.sh 200000 20
#
# 任意の環境変数:
#   WORK_ROOT=/vqword/runs
#   TOKENIZER_DIR=/vqword/tokenizer_wikitext103_bpe50k
#   PRETRAIN_EPOCHS=40
#   FINETUNE_EPOCHS=30
# ============================================================

VQ_CODEBOOK_SIZE="${1:-}"
HOP="${2:-}"

if [[ -z "${VQ_CODEBOOK_SIZE}" || -z "${HOP}" ]]; then
  echo "Usage: bash $0 <VQ_CODEBOOK_SIZE> <HOP>"
  echo "Example: bash $0 200000 20"
  exit 1
fi

if ! [[ "${VQ_CODEBOOK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] VQ_CODEBOOK_SIZE must be a positive integer: ${VQ_CODEBOOK_SIZE}"
  exit 1
fi

if ! [[ "${HOP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] HOP must be a positive integer: ${HOP}"
  exit 1
fi

label_from_size() {
  local size="$1"

  if (( size % 1000 == 0 )); then
    printf '%dk' "$((size / 1000))"
  else
    printf '%s' "${size}"
  fi
}

# ============================================================
# 共通設定
# ============================================================

BPE_VOCAB_LABEL="${BPE_VOCAB_LABEL:-50k}"

VQ_CODEBOOK_LABEL="$(label_from_size "${VQ_CODEBOOK_SIZE}")"

CENTER_SCALE="${CENTER_SCALE:-0.0}"
CENTER_LABEL="${CENTER_LABEL:-0}"
IVF_NLIST="${IVF_NLIST:-256}"

DISCRETIZATION_SEED="${DISCRETIZATION_SEED:-0}"
AR_SEED="${AR_SEED:-0}"

# Discover
WIKITEXT_MAX_SAMPLES="${WIKITEXT_MAX_SAMPLES:-1000000}"
WIKITEXT_SEQ_LEN="${WIKITEXT_SEQ_LEN:-256}"

DISCOVER_D_MODEL="${DISCOVER_D_MODEL:-256}"
DISCOVER_N_LAYERS="${DISCOVER_N_LAYERS:-3}"
DISCOVER_BATCH_SIZE="${DISCOVER_BATCH_SIZE:-1024}"

IVF_ITERS="${IVF_ITERS:-1}"
IVF_BATCH_SIZE="${IVF_BATCH_SIZE:-8192}"
GLOBAL_KMEANS_ITERS="${GLOBAL_KMEANS_ITERS:-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8192}"
K_BLOCK="${K_BLOCK:-4096}"

# TinyStories ID assignment
TINYSTORIES_DATASET="${TINYSTORIES_DATASET:-roneneldan/TinyStories}"
TINYSTORIES_SPLIT="${TINYSTORIES_SPLIT:-train}"
TINYSTORIES_MAX_SAMPLES="${TINYSTORIES_MAX_SAMPLES:-20000}"
TINYSTORIES_SEQ_LEN="${TINYSTORIES_SEQ_LEN:-256}"
ASSIGN_BATCH_SIZE="${ASSIGN_BATCH_SIZE:-512}"

# AR共通
AR_D_MODEL="${AR_D_MODEL:-256}"
AR_N_LAYERS="${AR_N_LAYERS:-6}"
AR_N_HEADS="${AR_N_HEADS:-8}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-16}"
AR_LR="${AR_LR:-3e-4}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
AUX_LAMBDA="${AUX_LAMBDA:-0.05}"

# ============================================================
# リポジトリと実行ディレクトリ
# ============================================================

REPO_DIR="${REPO_DIR:-/vqword}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/runs}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center${CENTER_LABEL}_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}"
RUN_NAME="${RUN_STAMP}_${TAG}"
RUN_DIR="${WORK_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

# ログ全体も保存する
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

echo "============================================================"
echo "[pipeline start]"
echo "run stamp            = ${RUN_STAMP}"
echo "run directory        = ${RUN_DIR}"
echo "tag                  = ${TAG}"
echo "BPE vocabulary       = ${BPE_VOCAB_LABEL}"
echo "VQW codebook label   = ${VQ_CODEBOOK_LABEL}"
echo "VQW codebook size    = ${VQ_CODEBOOK_SIZE}"
echo "context              = left-only"
echo "hop                  = ${HOP}"
echo "center scale         = ${CENTER_SCALE}"
echo "IVF nlist            = ${IVF_NLIST}"
echo "discretization seed  = ${DISCRETIZATION_SEED}"
echo "AR seed              = ${AR_SEED}"
echo "============================================================"

# ============================================================
# 環境準備
# ============================================================

python -m pip install \
  torch \
  datasets \
  transformers \
  scikit-learn \
  tqdm \
  numpy \
  pandas

cd /

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/takahh/vqword.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git pull --ff-only

GIT_COMMIT="$(git rev-parse HEAD)"
echo "[git] commit=${GIT_COMMIT}"

# ============================================================
# BPE tokenizer確認
#
# lftpは使わない。
# tokenizerはリポジトリ内、またはTOKENIZER_DIRで指定した場所に
# すでに存在している必要がある。
# ============================================================

TOKENIZER_DIR="${TOKENIZER_DIR:-${REPO_DIR}/tokenizer_wikitext103_bpe${BPE_VOCAB_LABEL}}"

# 旧名称のディレクトリも自動検出
if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
  LEGACY_TOKENIZER_DIR="${REPO_DIR}/bpe_wikitext103_50k"

  if [[ "${BPE_VOCAB_LABEL}" == "50k" && -f "${LEGACY_TOKENIZER_DIR}/tokenizer.json" ]]; then
    TOKENIZER_DIR="${LEGACY_TOKENIZER_DIR}"
  fi
fi

TOKENIZER_FILES=(
  "vocab.json"
  "merges.txt"
  "tokenizer.json"
  "tokenizer_config.json"
)

for filename in "${TOKENIZER_FILES[@]}"; do
  if [[ ! -s "${TOKENIZER_DIR}/${filename}" ]]; then
    echo "[error] tokenizer file is missing or empty:"
    echo "        ${TOKENIZER_DIR}/${filename}"
    echo
    echo "Set TOKENIZER_DIR to the local BPE tokenizer directory."
    exit 1
  fi
done

echo "============================================================"
echo "[tokenizer]"
echo "directory = ${TOKENIZER_DIR}"
ls -lh \
  "${TOKENIZER_DIR}/vocab.json" \
  "${TOKENIZER_DIR}/merges.txt" \
  "${TOKENIZER_DIR}/tokenizer.json" \
  "${TOKENIZER_DIR}/tokenizer_config.json"
echo "============================================================"

# ============================================================
# 全出力パス
# ============================================================

DISCOVER_CKPT="${RUN_DIR}/wikitext103_vqword_${TAG}.pt"
DISCOVER_DICTIONARY="${RUN_DIR}/wikitext103_vqword_${TAG}_dictionary.pt"
DISCOVER_IDS="${RUN_DIR}/wikitext103_vqword_${TAG}_ids.pt"

TINYSTORIES_IDS="${RUN_DIR}/tinystories_vqword_${TAG}_ids.pt"

PRETRAIN_PREFIX="${RUN_DIR}/ar_vqw2vqw_pretrain_${TAG}_d${AR_D_MODEL}_l${AR_N_LAYERS}_h${AR_N_HEADS}_arseed${AR_SEED}"
PRETRAIN_BEST="${PRETRAIN_PREFIX}.pt"
PRETRAIN_LAST="${PRETRAIN_PREFIX}_last.pt"
PRETRAIN_LOG="${PRETRAIN_PREFIX}.log"

FINETUNE_PREFIX="${RUN_DIR}/ar_bpeplusvqw2bpe_concatft_${TAG}_d${AR_D_MODEL}_l${AR_N_LAYERS}_h${AR_N_HEADS}_arseed${AR_SEED}_aux${AUX_LAMBDA}"
FINETUNE_BEST="${FINETUNE_PREFIX}.pt"
FINETUNE_LAST="${FINETUNE_PREFIX}_last.pt"
FINETUNE_LOG="${FINETUNE_PREFIX}.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_file() {
  local path="$1"

  if [[ ! -s "${path}" ]]; then
    echo "[error] expected output is missing or empty: ${path}"
    exit 1
  fi
}

# ============================================================
# Stage 1: Discover
# WikiText-103 -> VQW codebook / dictionary / WikiText IDs
# ============================================================

echo "============================================================"
echo "[stage 1/4] Discover"
echo "output = ${DISCOVER_CKPT}"
echo "============================================================"

python train_vqword.py \
  --dataset Salesforce/wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --text_col text \
  --tokenizer "${TOKENIZER_DIR}" \
  --max_samples "${WIKITEXT_MAX_SAMPLES}" \
  --seq_len "${WIKITEXT_SEQ_LEN}" \
  --hop "${HOP}" \
  --d_model "${DISCOVER_D_MODEL}" \
  --n_layers "${DISCOVER_N_LAYERS}" \
  --center_scale "${CENTER_SCALE}" \
  --ivf_nlist "${IVF_NLIST}" \
  --ivf_iters "${IVF_ITERS}" \
  --ivf_batch_size "${IVF_BATCH_SIZE}" \
  --global_codebook_size "${VQ_CODEBOOK_SIZE}" \
  --global_kmeans_iters "${GLOBAL_KMEANS_ITERS}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --batch_size "${DISCOVER_BATCH_SIZE}" \
  --k_block "${K_BLOCK}" \
  --seed "${DISCRETIZATION_SEED}" \
  --out "${DISCOVER_CKPT}"

require_file "${DISCOVER_CKPT}"
require_file "${DISCOVER_DICTIONARY}"
require_file "${DISCOVER_IDS}"

ls -lh \
  "${DISCOVER_CKPT}" \
  "${DISCOVER_DICTIONARY}" \
  "${DISCOVER_IDS}"

# Discover checkpointの設定を強制確認
python - "${DISCOVER_CKPT}" "${HOP}" "${CENTER_SCALE}" "${IVF_NLIST}" "${VQ_CODEBOOK_SIZE}" "${DISCRETIZATION_SEED}" <<'PY'
import sys
import torch

path, hop, center_scale, ivf_nlist, vq_size, seed = sys.argv[1:]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
args = ckpt.get("args", {})

expected = {
    "hop": int(hop),
    "center_scale": float(center_scale),
    "ivf_nlist": int(ivf_nlist),
    "global_codebook_size": int(vq_size),
    "seed": int(seed),
}

print("============================================================")
print("[Discover checkpoint verification]")
print("path:", path)
print("keys:", list(ckpt.keys()))

for key, expected_value in expected.items():
    actual = args.get(key)

    if actual is None:
        raise KeyError(f"checkpoint args is missing: {key}")

    if isinstance(expected_value, float):
        matched = abs(float(actual) - expected_value) < 1e-12
    else:
        matched = int(actual) == expected_value

    print(f"{key}: actual={actual}, expected={expected_value}")

    if not matched:
        raise ValueError(
            f"Discover configuration mismatch: "
            f"{key} expected={expected_value}, actual={actual}"
        )

actual_vq_size = ckpt.get("vq_vocab_size")

if actual_vq_size is None:
    raise KeyError("checkpoint is missing vq_vocab_size")

if int(actual_vq_size) != int(vq_size):
    raise ValueError(
        f"vq_vocab_size mismatch: expected={vq_size}, actual={actual_vq_size}"
    )

centers = ckpt.get("global_centers")

if centers is not None:
    print("global_centers:", tuple(centers.shape))

print("[check] Discover checkpoint OK")
print("============================================================")
PY

# ============================================================
# Stage 2: TinyStories ID assignment
# Discover checkpoint -> TinyStories BPE/VQW aligned IDs
# ============================================================

echo "============================================================"
echo "[stage 2/4] Assign TinyStories VQW IDs"
echo "checkpoint = ${DISCOVER_CKPT}"
echo "output     = ${TINYSTORIES_IDS}"
echo "============================================================"

python assign_vqword_ids.py \
  --ckpt "${DISCOVER_CKPT}" \
  --dataset "${TINYSTORIES_DATASET}" \
  --split "${TINYSTORIES_SPLIT}" \
  --text_col text \
  --tokenizer "${TOKENIZER_DIR}" \
  --max_samples "${TINYSTORIES_MAX_SAMPLES}" \
  --seq_len "${TINYSTORIES_SEQ_LEN}" \
  --batch_size "${ASSIGN_BATCH_SIZE}" \
  --k_block "${K_BLOCK}" \
  --out "${TINYSTORIES_IDS}"

require_file "${TINYSTORIES_IDS}"
ls -lh "${TINYSTORIES_IDS}"

python - "${TINYSTORIES_IDS}" "${VQ_CODEBOOK_SIZE}" <<'PY'
import sys
import torch

path = sys.argv[1]
expected_vq_size = int(sys.argv[2])

data = torch.load(path, map_location="cpu", weights_only=False)

required = [
    "samples",
    "token_ids_flat",
    "vq_ids_flat",
    "vq_vocab_size",
]

for key in required:
    if key not in data:
        raise KeyError(
            f"ID data is missing {key}; keys={list(data.keys())}"
        )

samples = data["samples"]
token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)
actual_vq_size = int(data["vq_vocab_size"])

print("============================================================")
print("[TinyStories ID verification]")
print("path:", path)
print("samples:", f"{len(samples):,}")
print("token count:", f"{token_ids.numel():,}")
print("VQ count:", f"{vq_ids.numel():,}")
print("VQ vocabulary:", f"{actual_vq_size:,}")

if not samples:
    raise ValueError("samples is empty")

if token_ids.numel() == 0 or vq_ids.numel() == 0:
    raise ValueError("token_ids or vq_ids is empty")

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        f"Token/VQ length mismatch: "
        f"{token_ids.numel():,} != {vq_ids.numel():,}"
    )

if actual_vq_size != expected_vq_size:
    raise ValueError(
        f"VQ vocabulary mismatch: "
        f"expected={expected_vq_size:,}, actual={actual_vq_size:,}"
    )

vq_min = int(vq_ids.min())
vq_max = int(vq_ids.max())
used_vq = int(torch.unique(vq_ids).numel())

print("token min/max:", int(token_ids.min()), int(token_ids.max()))
print("VQ min/max:", vq_min, vq_max)
print("used VQ IDs:", f"{used_vq:,}")
print("VQ usage ratio:", f"{used_vq / actual_vq_size:.6f}")

if vq_min < 0 or vq_max >= actual_vq_size:
    raise ValueError(
        f"VQ IDs out of range: min={vq_min}, max={vq_max}, "
        f"vocab={actual_vq_size}"
    )

first = samples[0]

if "start" not in first or "end" not in first:
    raise KeyError(
        f"sample requires start/end; first keys={list(first.keys())}"
    )

print("[check] TinyStories IDs OK")
print("============================================================")
PY

# ============================================================
# Stage 3: VQW -> VQW autoregressive pretraining
# ============================================================

echo "============================================================"
echo "[stage 3/4] VQW -> VQW pretraining"
echo "data = ${TINYSTORIES_IDS}"
echo "best = ${PRETRAIN_BEST}"
echo "============================================================"

python ar.py \
  --mode pretrain \
  --data "${TINYSTORIES_IDS}" \
  --epochs "${PRETRAIN_EPOCHS}" \
  --batch_size "${AR_BATCH_SIZE}" \
  --d_model "${AR_D_MODEL}" \
  --n_layers "${AR_N_LAYERS}" \
  --n_heads "${AR_N_HEADS}" \
  --lr "${AR_LR}" \
  --vq_only \
  --main_target vq \
  --aux_lambda 0 \
  --seed "${AR_SEED}" \
  --out "${PRETRAIN_BEST}" \
  2>&1 | tee "${PRETRAIN_LOG}"

require_file "${PRETRAIN_BEST}"
require_file "${PRETRAIN_LAST}"
require_file "${PRETRAIN_LOG}"

ls -lh \
  "${PRETRAIN_BEST}" \
  "${PRETRAIN_LAST}" \
  "${PRETRAIN_LOG}"

python - "${PRETRAIN_BEST}" "${VQ_CODEBOOK_SIZE}" "${AR_D_MODEL}" "${AR_N_LAYERS}" "${AR_N_HEADS}" "${AR_SEED}" <<'PY'
import sys
import torch

path, vq_size, d_model, n_layers, n_heads, seed = sys.argv[1:]
ckpt = torch.load(path, map_location="cpu", weights_only=False)

expected_vq_size = int(vq_size)
actual_vq_size = ckpt.get("vq_vocab_size")

if actual_vq_size is None:
    raise KeyError("pretrain checkpoint is missing vq_vocab_size")

if int(actual_vq_size) != expected_vq_size:
    raise ValueError(
        f"pretrain VQ size mismatch: "
        f"expected={expected_vq_size}, actual={actual_vq_size}"
    )

args = ckpt.get("args", {})
checks = {
    "d_model": int(d_model),
    "n_layers": int(n_layers),
    "n_heads": int(n_heads),
    "seed": int(seed),
}

for key, expected in checks.items():
    actual = args.get(key)

    if actual is not None and int(actual) != expected:
        raise ValueError(
            f"pretrain mismatch: {key} expected={expected}, actual={actual}"
        )

if args.get("main_target") not in (None, "vq"):
    raise ValueError(
        f"pretrain main_target must be vq: {args.get('main_target')}"
    )

print("[check] pretrain checkpoint OK:", path)
PY

# ============================================================
# Stage 4: BPE + VQW -> BPE autoregressive finetuning
# ============================================================

echo "============================================================"
echo "[stage 4/4] BPE + VQW -> BPE finetuning"
echo "data      = ${TINYSTORIES_IDS}"
echo "init_from = ${PRETRAIN_BEST}"
echo "best      = ${FINETUNE_BEST}"
echo "============================================================"

python ar.py \
  --mode finetune \
  --data "${TINYSTORIES_IDS}" \
  --init_from "${PRETRAIN_BEST}" \
  --main_target tok \
  --aux_lambda "${AUX_LAMBDA}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --batch_size "${AR_BATCH_SIZE}" \
  --d_model "${AR_D_MODEL}" \
  --n_layers "${AR_N_LAYERS}" \
  --n_heads "${AR_N_HEADS}" \
  --lr "${AR_LR}" \
  --seed "${AR_SEED}" \
  --out "${FINETUNE_BEST}" \
  2>&1 | tee "${FINETUNE_LOG}"

require_file "${FINETUNE_BEST}"
require_file "${FINETUNE_LAST}"
require_file "${FINETUNE_LOG}"

ls -lh \
  "${FINETUNE_BEST}" \
  "${FINETUNE_LAST}" \
  "${FINETUNE_LOG}"

# ============================================================
# 最終評価・整合性確認
# ============================================================

echo "============================================================"
echo "[evaluation summary]"
grep -E "\[eval\]|\[save\]|\[loss-weight\]" "${FINETUNE_LOG}" || true
echo "============================================================"

python - "${FINETUNE_LOG}" "${FINETUNE_BEST}" "${FINETUNE_LAST}" "${VQ_CODEBOOK_SIZE}" <<'PY'
import re
import sys
import torch

log_path, best_path, last_path, vq_size = sys.argv[1:]
expected_vq_size = int(vq_size)

pattern = re.compile(
    r"\[eval\]\s+ep=(\d+).*?test_tok_ppl=([0-9.]+)"
)

results = []

with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            results.append((int(match.group(1)), float(match.group(2))))

print("============================================================")
print("[best token perplexity]")

if results:
    best_epoch, best_ppl = min(results, key=lambda x: x[1])
    print("best epoch:", best_epoch)
    print("best test_tok_ppl:", best_ppl)

    for epoch, ppl in results:
        marker = " <-- best" if epoch == best_epoch else ""
        print(f"ep={epoch:02d} test_tok_ppl={ppl:.4f}{marker}")
else:
    print("No test_tok_ppl entries found")

for path in (best_path, last_path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actual_vq_size = ckpt.get("vq_vocab_size")

    if actual_vq_size is None:
        raise KeyError(f"{path} is missing vq_vocab_size")

    if int(actual_vq_size) != expected_vq_size:
        raise ValueError(
            f"output VQ size mismatch in {path}: "
            f"expected={expected_vq_size}, actual={actual_vq_size}"
        )

    print("verified checkpoint:", path)
    print("  vq_vocab_size:", actual_vq_size)
    print("  token_vocab_size:", ckpt.get("token_vocab_size"))
    print("  valid_loss:", ckpt.get("valid_loss"))
    print("  test_loss:", ckpt.get("test_loss"))

print("============================================================")
PY

# ============================================================
# 実行情報とSHA256を保存
# ============================================================

cat > "${RUN_DIR}/run_config.txt" <<EOF
run_stamp=${RUN_STAMP}
run_name=${RUN_NAME}
run_dir=${RUN_DIR}
git_commit=${GIT_COMMIT}

bpe_vocab_label=${BPE_VOCAB_LABEL}
vq_codebook_label=${VQ_CODEBOOK_LABEL}
vq_codebook_size=${VQ_CODEBOOK_SIZE}

context=left-only
hop=${HOP}
center_scale=${CENTER_SCALE}
ivf_nlist=${IVF_NLIST}
discretization_seed=${DISCRETIZATION_SEED}
ar_seed=${AR_SEED}

tokenizer_dir=${TOKENIZER_DIR}

discover_ckpt=${DISCOVER_CKPT}
discover_dictionary=${DISCOVER_DICTIONARY}
discover_ids=${DISCOVER_IDS}
tinystories_ids=${TINYSTORIES_IDS}
pretrain_best=${PRETRAIN_BEST}
pretrain_last=${PRETRAIN_LAST}
finetune_best=${FINETUNE_BEST}
finetune_last=${FINETUNE_LAST}
EOF

find "${RUN_DIR}" -maxdepth 1 -type f \
  ! -name "SHA256SUMS" \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${RUN_DIR}/SHA256SUMS"

echo "============================================================"
echo "[completed]"
echo "run directory = ${RUN_DIR}"
echo "tag           = ${TAG}"
echo "Discover      = ${DISCOVER_CKPT}"
echo "TinyStories   = ${TINYSTORIES_IDS}"
echo "pretrain best = ${PRETRAIN_BEST}"
echo "finetune best = ${FINETUNE_BEST}"
echo "config        = ${RUN_DIR}/run_config.txt"
echo "checksums     = ${RUN_DIR}/SHA256SUMS"
echo "pipeline log  = ${RUN_DIR}/pipeline.log"
echo "============================================================"
