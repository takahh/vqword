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
  numpy \
  pandas

cd /

if [ ! -d /vqword ]; then
  git clone https://github.com/takahh/vqword.git
fi

cd /vqword
git pull

# ============================================================
# Step 6: TinyStories BPE-only AR
#
# 入力:
#   BPE[t]
#
# 正解:
#   BPE[t+1]
#
# 入力データには以下の両方が保存されている:
#   token_ids_flat
#   vq_ids_flat
#
# この実験では --token_only により、
# token_ids_flatだけをモデル入力として使用する。
#
# このcheckpointは、
# 後段のBPE+VQWord finetuneの初期値にも使用する。
# ============================================================

# ============================================================
# 実験設定
# ============================================================

BPE_VOCAB_LABEL=50257
BPE_VOCAB_SIZE=50257

VQ_CODEBOOK_LABEL=200k
VQ_CODEBOOK_SIZE=200000

HOP=20
CENTER_SCALE=0.0
IVF_NLIST=256

DISCRETIZATION_SEED=0
AR_SEED=0

# ============================================================
# ARモデル設定
# ============================================================

EPOCHS=40
BATCH_SIZE=32

D_MODEL=256
N_LAYERS=6
N_HEADS=8

DROPOUT=0.1
LR=3e-4

# ============================================================
# ファイル名
#
# Step 4&5で生成したTinyStoriesデータを使用
# ============================================================

TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center0_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}"

DATA="tinystories_vqword_${TAG}_ids.pt"
DATA_PATH="/vqword/${DATA}"

RUN="ar_bpeonly_tinystories_${TAG}_d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_arseed${AR_SEED}_$(date +%Y%m%d_%H%M%S)"

BEST_PATH="/vqword/${RUN}.pt"
LAST_PATH="/vqword/${RUN}_last.pt"
LOG_PATH="/vqword/${RUN}.log"

# ============================================================
# FTP設定
#
# 実行前:
#   export FTP_PASS='FTPパスワード'
# ============================================================

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"

echo "============================================================"
echo "[configuration]"
echo "task                 = TinyStories BPE-only AR"
echo "BPE vocabulary       = ${BPE_VOCAB_SIZE}"
echo "VQW codebook         = ${VQ_CODEBOOK_SIZE}"
echo "VQW context          = left-only"
echo "VQW hop              = ${HOP}"
echo "center scale         = ${CENTER_SCALE}"
echo "IVF nlist            = ${IVF_NLIST}"
echo "discretization seed  = ${DISCRETIZATION_SEED}"
echo "AR seed              = ${AR_SEED}"
echo "epochs               = ${EPOCHS}"
echo "batch size           = ${BATCH_SIZE}"
echo "d_model              = ${D_MODEL}"
echo "n_layers             = ${N_LAYERS}"
echo "n_heads              = ${N_HEADS}"
echo "dropout              = ${DROPOUT}"
echo "learning rate        = ${LR}"
echo "tag                  = ${TAG}"
echo "data                 = ${DATA}"
echo "run                  = ${RUN}"
echo "============================================================"

# ============================================================
# TinyStories BPE + VQWord IDデータをFTPから取得
# ============================================================

rm -f "${DATA_PATH}"

echo "============================================================"
echo "[download data]"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${DATA}" \
  -o "${DATA_PATH}"

bye
EOF

if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file was not downloaded:"
  echo "        ${DATA_PATH}"
  exit 1
fi

echo "============================================================"
echo "[downloaded data]"
echo "============================================================"

ls -lh "${DATA_PATH}"

# ============================================================
# データ内容を学習前に検証
# ============================================================

python - <<PY
import torch

path = "${DATA_PATH}"

expected_token_vocab_size = ${BPE_VOCAB_SIZE}
expected_vq_vocab_size = ${VQ_CODEBOOK_SIZE}
expected_hop = ${HOP}

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

print("============================================================")
print("[data verification]")
print("path:", path)
print("keys:", list(data.keys()))

required_keys = {
    "samples",
    "token_ids_flat",
    "vq_ids_flat",
    "vq_vocab_size",
}

missing = sorted(required_keys - set(data.keys()))

if missing:
    raise KeyError(
        f"Required keys are missing: {missing}. "
        f"Available keys: {list(data.keys())}"
    )

samples = data["samples"]
token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)

vq_vocab_size = int(data["vq_vocab_size"])

print("samples:", f"{len(samples):,}")
print("token_ids shape:", tuple(token_ids.shape))
print("token_ids dtype:", token_ids.dtype)
print("vq_ids shape:", tuple(vq_ids.shape))
print("vq_ids dtype:", vq_ids.dtype)

if len(samples) == 0:
    raise ValueError("samples is empty")

if token_ids.numel() == 0:
    raise ValueError("token_ids_flat is empty")

if vq_ids.numel() == 0:
    raise ValueError("vq_ids_flat is empty")

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        "Token/VQWord length mismatch: "
        f"token={token_ids.numel():,}, "
        f"vq={vq_ids.numel():,}"
    )

token_min = int(token_ids.min())
token_max = int(token_ids.max())

vq_min = int(vq_ids.min())
vq_max = int(vq_ids.max())

used_tokens = int(torch.unique(token_ids).numel())
used_vq = int(torch.unique(vq_ids).numel())

print("token min/max:", token_min, token_max)
print("used BPE IDs:", f"{used_tokens:,}")
print("expected BPE vocabulary:", f"{expected_token_vocab_size:,}")

print("VQ min/max:", vq_min, vq_max)
print("used VQ IDs:", f"{used_vq:,}")
print("vq_vocab_size:", f"{vq_vocab_size:,}")

if token_min < 0:
    raise ValueError(
        f"Negative BPE ID found: {token_min}"
    )

if token_max >= expected_token_vocab_size:
    raise ValueError(
        "BPE ID is out of range: "
        f"max={token_max:,}, "
        f"vocab_size={expected_token_vocab_size:,}"
    )

if vq_min < 0:
    raise ValueError(
        f"Negative VQWord ID found: {vq_min}"
    )

if vq_vocab_size != expected_vq_vocab_size:
    raise ValueError(
        "VQ vocabulary mismatch: "
        f"expected={expected_vq_vocab_size:,}, "
        f"actual={vq_vocab_size:,}"
    )

if vq_max >= vq_vocab_size:
    raise ValueError(
        "VQWord ID is out of range: "
        f"max={vq_max:,}, "
        f"vq_vocab_size={vq_vocab_size:,}"
    )

first_sample = samples[0]

print("first sample keys:", list(first_sample.keys()))

if "start" not in first_sample or "end" not in first_sample:
    raise KeyError(
        "Current ar.py requires samples containing "
        "'start' and 'end'. "
        f"First sample keys: {list(first_sample.keys())}"
    )

start = int(first_sample["start"])
end = int(first_sample["end"])

if start < 0 or end > token_ids.numel() or start >= end:
    raise ValueError(
        "Invalid first sample range: "
        f"start={start}, end={end}, "
        f"total={token_ids.numel():,}"
    )

print("first sample start/end:", start, end)

print(
    "first BPE IDs:",
    token_ids[start:min(end, start + 20)].tolist(),
)

print(
    "first VQWord IDs:",
    vq_ids[start:min(end, start + 20)].tolist(),
)

metadata_hop = data.get("hop")

if metadata_hop is not None:
    print("metadata hop:", metadata_hop)

    if int(metadata_hop) != expected_hop:
        raise ValueError(
            "VQ context mismatch: "
            f"expected hop={expected_hop}, "
            f"actual hop={metadata_hop}"
        )
else:
    print(
        "[note] hop metadata is not stored; "
        "left20 is identified from the filename"
    )

print("pad_token_id:", data.get("pad_token_id"))
print("vq_pad_id:", data.get("vq_pad_id"))
print("tokenizer:", data.get("tokenizer"))
print("checkpoint:", data.get("ckpt"))

print("[check] OK")
print("============================================================")
PY

# ============================================================
# TinyStories BPE-only AR
#
# --mode finetune:
#   現在のar.pyではtoken lossを学習するために必要
#
# --token_only:
#   BPE embeddingだけを入力に使用
#
# --main_target tok:
#   BPEの次トークン予測を主目的にする
#
# --aux_lambda 0:
#   VQ lossを学習目的へ加えない
#
# --token_vocab_size 50257:
#   TinyStories内で未使用のBPE IDがあっても、
#   tokenizer本来の語彙数を維持する
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "[start TinyStories BPE-only AR]"
echo "input       = BPE[t]"
echo "target      = BPE[t+1]"
echo "BPE vocab   = ${BPE_VOCAB_SIZE}"
echo "data        = ${DATA_PATH}"
echo "run         = ${RUN}"
echo "============================================================"

python ar.py \
  --mode finetune \
  --data "${DATA_PATH}" \
  --token_vocab_size "${BPE_VOCAB_SIZE}" \
  --vq_vocab_size "${VQ_CODEBOOK_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --n_heads "${N_HEADS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --token_only \
  --main_target tok \
  --aux_lambda 0 \
  --seed "${AR_SEED}" \
  --out "${BEST_PATH}" \
  2>&1 | tee "${LOG_PATH}"

# ============================================================
# 生成物の存在確認
# ============================================================

echo "============================================================"
echo "[generated files]"
echo "============================================================"

for path in \
  "${BEST_PATH}" \
  "${LAST_PATH}" \
  "${LOG_PATH}"
do
  if [ ! -f "${path}" ]; then
    echo "[error] Expected output was not generated:"
    echo "        ${path}"
    exit 1
  fi
done

ls -lh \
  "${BEST_PATH}" \
  "${LAST_PATH}" \
  "${LOG_PATH}"

# ============================================================
# checkpointを確認
# ============================================================

python - <<PY
import torch

paths = [
    "${BEST_PATH}",
    "${LAST_PATH}",
]

expected_token_vocab_size = ${BPE_VOCAB_SIZE}
expected_vq_vocab_size = ${VQ_CODEBOOK_SIZE}

for path in paths:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print("============================================================")
    print("[checkpoint verification]")
    print("path:", path)
    print("keys:", list(checkpoint.keys()))

    model = checkpoint.get("model")

    if model is None:
        raise KeyError(
            f"Checkpoint does not contain model: {path}"
        )

    if "tok_emb.weight" not in model:
        raise KeyError(
            f"Checkpoint does not contain tok_emb.weight: {path}"
        )

    actual_token_vocab_size = int(
        model["tok_emb.weight"].shape[0]
    )

    actual_vq_vocab_size = checkpoint.get(
        "vq_vocab_size"
    )

    print(
        "tok_emb shape:",
        tuple(model["tok_emb.weight"].shape),
    )

    if "tok_head.weight" in model:
        print(
            "tok_head shape:",
            tuple(model["tok_head.weight"].shape),
        )

    print("token_vocab_size:", actual_token_vocab_size)
    print("vq_vocab_size:", actual_vq_vocab_size)
    print("epoch:", checkpoint.get("epoch"))
    print("data:", checkpoint.get("data"))
    print("valid_loss:", checkpoint.get("valid_loss"))
    print("test_loss:", checkpoint.get("test_loss"))

    if actual_token_vocab_size != expected_token_vocab_size:
        raise ValueError(
            "Checkpoint BPE vocabulary mismatch: "
            f"expected={expected_token_vocab_size:,}, "
            f"actual={actual_token_vocab_size:,}"
        )

    if actual_vq_vocab_size is not None:
        if int(actual_vq_vocab_size) != expected_vq_vocab_size:
            raise ValueError(
                "Checkpoint VQ vocabulary mismatch: "
                f"expected={expected_vq_vocab_size:,}, "
                f"actual={int(actual_vq_vocab_size):,}"
            )

    history = checkpoint.get("history")

    if isinstance(history, dict):
        epochs = history.get("epoch", [])
        valid_tok_ppl = history.get(
            "valid_tok_ppl",
            [],
        )
        test_tok_ppl = history.get(
            "test_tok_ppl",
            [],
        )

        if epochs:
            print("completed epochs:", epochs[-1])

        if valid_tok_ppl:
            print(
                "last valid_tok_ppl:",
                valid_tok_ppl[-1],
            )

        if test_tok_ppl:
            print(
                "last test_tok_ppl:",
                test_tok_ppl[-1],
            )

    elif history is not None:
        print(
            "history type:",
            type(history).__name__,
        )

    args = checkpoint.get("args", {})

    print("mode:", args.get("mode"))
    print("token_only:", args.get("token_only"))
    print("main_target:", args.get("main_target"))
    print("aux_lambda:", args.get("aux_lambda"))

    if args.get("mode") != "finetune":
        raise ValueError(
            f"Unexpected mode: {args.get('mode')}"
        )

    if not args.get("token_only", False):
        raise ValueError(
            "Checkpoint was not trained with --token_only"
        )

print("============================================================")
print("[check] checkpoints OK")
print("============================================================")
PY

# ============================================================
# FTPへアップロード
# ============================================================

echo "============================================================"
echo "[upload files]"
echo "task = TinyStories BPE-only AR"
echo "run  = ${RUN}"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

cd vqword_logs

put "${BEST_PATH}" \
  -o "${RUN}.pt"

put "${LAST_PATH}" \
  -o "${RUN}_last.pt"

put "${LOG_PATH}" \
  -o "${RUN}.log"

bye
EOF

echo "============================================================"
echo "[completed]"
echo "TASK           = TinyStories BPE-only AR"
echo "BPE vocabulary = ${BPE_VOCAB_SIZE}"
echo "VQW codebook   = ${VQ_CODEBOOK_LABEL}"
echo "context        = left ${HOP}"
echo "AR seed        = ${AR_SEED}"
echo "DATA           = ${DATA}"
echo "BEST           = vqword_logs/${RUN}.pt"
echo "LAST           = vqword_logs/${RUN}_last.pt"
echo "LOG            = vqword_logs/${RUN}.log"
echo ""
echo "Next step:"
echo "Use BEST as --init_from with --init_source bpe"
echo "for BPE+VQWord finetuning."
echo "============================================================"