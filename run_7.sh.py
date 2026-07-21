#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Step 7
#
# TinyStories:
#   BPE[t] + VQW[t] → BPE[t+1]
#
# Step 6で学習したBPE-only ARを初期値として使用する。
#
# BPE baselineから引き継ぐ:
#   tok_emb
#   pos_emb
#   Transformer
#   norm
#   tok_head
#
# 新規初期化:
#   vq_emb
#   vq_head
#   input_fusion
#
# input_fusionは初期状態で
#
#   fusion([BPE, VQW]) = BPE
#
# となるため、学習開始時点ではStep 6のBPEモデルと
# 同じ予測を再現する。
# ============================================================


# ============================================================
# Step 6の実際の実行日時
#
# FTP上にあるBPE-onlyベストモデルの日時へ変更する。
#
# 例:
#   BPE_BASELINE_TIMESTAMP=20260721_184530
# ============================================================

BPE_BASELINE_TIMESTAMP=YYYYMMDD_HHMMSS


# ============================================================
# 環境準備
# ============================================================

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
# FTP設定
#
# 実行前:
#
#   export FTP_PASS='FTPパスワード'
#   bash run_7.sh
# ============================================================

FTP_USER="${FTP_USER:-chicappa.jp-wakou}"
FTP_PASS="${FTP_PASS:?Set FTP_PASS before running this script}"
FTP_HOST="${FTP_HOST:-ftp.lolipop.jp}"


# ============================================================
# VQWord・BPE設定
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
#
# Step 6とモデル構造を一致させる。
# ============================================================

D_MODEL=256
N_LAYERS=6
N_HEADS=8
DROPOUT=0.1

EPOCHS=30
BATCH_SIZE=16
LR=3e-4

# 主目的:
#   BPE next-token prediction
#
# 補助目的:
#   VQW next-token prediction
AUX_LAMBDA=0.05


# ============================================================
# ファイル名
# ============================================================

TAG="bpe${BPE_VOCAB_LABEL}_left${HOP}_center0_global_ivf${IVF_NLIST}_vqcb${VQ_CODEBOOK_LABEL}"

# Step 4&5の成果物
DATA="tinystories_vqword_${TAG}_ids.pt"
DATA_PATH="/vqword/${DATA}"

# Step 6のbest checkpoint
BPE_BASELINE="ar_bpeonly_tinystories_${TAG}_d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_arseed${AR_SEED}_${BPE_BASELINE_TIMESTAMP}.pt"
BPE_BASELINE_PATH="/vqword/${BPE_BASELINE}"

# Step 7の実行名
RUN="ar_bpeplusvqw2bpe_bpeinit_${TAG}_d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_arseed${AR_SEED}_aux${AUX_LAMBDA}_$(date +%Y%m%d_%H%M%S)"

BEST_PATH="/vqword/${RUN}.pt"
LAST_PATH="/vqword/${RUN}_last.pt"
LOG_PATH="/vqword/${RUN}.log"


# ============================================================
# 設定表示
# ============================================================

echo "============================================================"
echo "[configuration]"
echo "task                  = TinyStories BPE + VQW to BPE"
echo "initialization        = Step 6 BPE-only baseline"
echo "init source           = bpe"
echo "BPE vocabulary label  = ${BPE_VOCAB_LABEL}"
echo "BPE vocabulary size   = ${BPE_VOCAB_SIZE}"
echo "VQW codebook label    = ${VQ_CODEBOOK_LABEL}"
echo "VQW codebook size     = ${VQ_CODEBOOK_SIZE}"
echo "VQW context           = left-only"
echo "VQW hop               = ${HOP}"
echo "center scale          = ${CENTER_SCALE}"
echo "IVF nlist             = ${IVF_NLIST}"
echo "discretization seed   = ${DISCRETIZATION_SEED}"
echo "AR seed               = ${AR_SEED}"
echo "d_model               = ${D_MODEL}"
echo "n_layers              = ${N_LAYERS}"
echo "n_heads               = ${N_HEADS}"
echo "dropout               = ${DROPOUT}"
echo "epochs                = ${EPOCHS}"
echo "batch size            = ${BATCH_SIZE}"
echo "learning rate         = ${LR}"
echo "aux lambda            = ${AUX_LAMBDA}"
echo "tag                   = ${TAG}"
echo "data                  = ${DATA}"
echo "BPE baseline          = ${BPE_BASELINE}"
echo "run                   = ${RUN}"
echo "============================================================"


# ============================================================
# placeholderのまま実行されるのを防ぐ
# ============================================================

if [ "${BPE_BASELINE_TIMESTAMP}" = "YYYYMMDD_HHMMSS" ]; then
  echo "[error] BPE_BASELINE_TIMESTAMPをStep 6の実際の日時へ変更してください。"
  echo
  echo "Expected filename:"
  echo "ar_bpeonly_tinystories_${TAG}_d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_arseed${AR_SEED}_<TIMESTAMP>.pt"
  exit 1
fi


# ============================================================
# 既存ファイルを削除
# ============================================================

rm -f "${DATA_PATH}"
rm -f "${BPE_BASELINE_PATH}"
rm -f "${BEST_PATH}"
rm -f "${LAST_PATH}"
rm -f "${LOG_PATH}"


# ============================================================
# FTPから入力データとBPE baselineを取得
#
# DATA:
#   FTPルート
#
# BPE baseline:
#   FTP/vqword_logs/
# ============================================================

echo "============================================================"
echo "[download input files]"
echo "DATA         = ${DATA}"
echo "BPE baseline = ${BPE_BASELINE}"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

get "${DATA}" \
  -o "${DATA_PATH}"

cd vqword_logs

get "${BPE_BASELINE}" \
  -o "${BPE_BASELINE_PATH}"

bye
EOF


# ============================================================
# ダウンロード確認
# ============================================================

if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file was not downloaded:"
  echo "        ${DATA_PATH}"
  exit 1
fi

if [ ! -f "${BPE_BASELINE_PATH}" ]; then
  echo "[error] BPE baseline was not downloaded:"
  echo "        ${BPE_BASELINE_PATH}"
  exit 1
fi

echo "============================================================"
echo "[downloaded files]"
echo "============================================================"

ls -lh \
  "${DATA_PATH}" \
  "${BPE_BASELINE_PATH}"


# ============================================================
# データとBPE baselineの整合性確認
# ============================================================

python - <<PY
import torch

data_path = "${DATA_PATH}"
baseline_path = "${BPE_BASELINE_PATH}"

expected_token_vocab_size = ${BPE_VOCAB_SIZE}
expected_vq_vocab_size = ${VQ_CODEBOOK_SIZE}

expected_d_model = ${D_MODEL}
expected_n_layers = ${N_LAYERS}
expected_n_heads = ${N_HEADS}
expected_seed = ${AR_SEED}


# ============================================================
# データ確認
# ============================================================

data = torch.load(
    data_path,
    map_location="cpu",
    weights_only=False,
)

print("============================================================")
print("[data verification]")
print("path:", data_path)
print("keys:", list(data.keys()))

required_data_keys = {
    "samples",
    "token_ids_flat",
    "vq_ids_flat",
    "vq_vocab_size",
}

missing = sorted(required_data_keys - set(data.keys()))

if missing:
    raise KeyError(
        f"Required data keys are missing: {missing}. "
        f"Available keys: {list(data.keys())}"
    )

samples = data["samples"]
token_ids = data["token_ids_flat"].long().reshape(-1)
vq_ids = data["vq_ids_flat"].long().reshape(-1)

data_vq_vocab_size = int(data["vq_vocab_size"])

if len(samples) == 0:
    raise ValueError("samples is empty")

if token_ids.numel() == 0:
    raise ValueError("token_ids_flat is empty")

if vq_ids.numel() == 0:
    raise ValueError("vq_ids_flat is empty")

if token_ids.numel() != vq_ids.numel():
    raise ValueError(
        "Token/VQ length mismatch: "
        f"token={token_ids.numel():,}, "
        f"vq={vq_ids.numel():,}"
    )

token_min = int(token_ids.min())
token_max = int(token_ids.max())

vq_min = int(vq_ids.min())
vq_max = int(vq_ids.max())

print("samples:", f"{len(samples):,}")
print("token count:", f"{token_ids.numel():,}")
print("VQ count:", f"{vq_ids.numel():,}")

print("token min/max:", token_min, token_max)
print("VQ min/max:", vq_min, vq_max)

print(
    "used BPE IDs:",
    f"{torch.unique(token_ids).numel():,}",
)

print(
    "used VQ IDs:",
    f"{torch.unique(vq_ids).numel():,}",
)

print(
    "data vq_vocab_size:",
    f"{data_vq_vocab_size:,}",
)

if token_min < 0:
    raise ValueError(
        f"Negative BPE ID found: {token_min}"
    )

if token_max >= expected_token_vocab_size:
    raise ValueError(
        "BPE ID out of range: "
        f"max={token_max:,}, "
        f"vocab_size={expected_token_vocab_size:,}"
    )

if vq_min < 0:
    raise ValueError(
        f"Negative VQ ID found: {vq_min}"
    )

if data_vq_vocab_size != expected_vq_vocab_size:
    raise ValueError(
        "Data VQ vocabulary mismatch: "
        f"expected={expected_vq_vocab_size:,}, "
        f"actual={data_vq_vocab_size:,}"
    )

if vq_max >= data_vq_vocab_size:
    raise ValueError(
        "VQ ID out of range: "
        f"max={vq_max:,}, "
        f"vq_vocab_size={data_vq_vocab_size:,}"
    )

first_sample = samples[0]

if "start" not in first_sample or "end" not in first_sample:
    raise KeyError(
        "Current ar.py requires samples containing "
        "'start' and 'end'. "
        f"First sample keys: {list(first_sample.keys())}"
    )

print("first sample keys:", list(first_sample.keys()))
print("tokenizer:", data.get("tokenizer"))
print("source VQ checkpoint:", data.get("ckpt"))


# ============================================================
# BPE baseline確認
# ============================================================

baseline = torch.load(
    baseline_path,
    map_location="cpu",
    weights_only=False,
)

print("------------------------------------------------------------")
print("[BPE baseline verification]")
print("path:", baseline_path)
print("keys:", list(baseline.keys()))

if "model" not in baseline:
    raise KeyError(
        "BPE baseline does not contain 'model'"
    )

model = baseline["model"]

required_model_keys = [
    "tok_emb.weight",
    "pos_emb.weight",
    "tok_head.weight",
    "tok_head.bias",
]

for key in required_model_keys:
    if key not in model:
        raise KeyError(
            f"BPE baseline model does not contain: {key}"
        )

tok_emb_shape = tuple(
    model["tok_emb.weight"].shape
)

tok_head_shape = tuple(
    model["tok_head.weight"].shape
)

print("tok_emb shape:", tok_emb_shape)
print("tok_head shape:", tok_head_shape)
print(
    "pos_emb shape:",
    tuple(model["pos_emb.weight"].shape),
)

actual_token_vocab_size = int(
    model["tok_emb.weight"].shape[0]
)

actual_d_model = int(
    model["tok_emb.weight"].shape[1]
)

if actual_token_vocab_size != expected_token_vocab_size:
    raise ValueError(
        "BPE baseline vocabulary mismatch: "
        f"expected={expected_token_vocab_size:,}, "
        f"actual={actual_token_vocab_size:,}"
    )

if actual_d_model != expected_d_model:
    raise ValueError(
        "BPE baseline d_model mismatch: "
        f"expected={expected_d_model}, "
        f"actual={actual_d_model}"
    )

if int(model["tok_head.weight"].shape[0]) != expected_token_vocab_size:
    raise ValueError(
        "BPE baseline tok_head vocabulary mismatch"
    )

if int(model["tok_head.weight"].shape[1]) != expected_d_model:
    raise ValueError(
        "BPE baseline tok_head d_model mismatch"
    )

args = baseline.get("args", {})

print("checkpoint epoch:", baseline.get("epoch"))
print("checkpoint token_vocab_size:", baseline.get("token_vocab_size"))
print("checkpoint vq_vocab_size:", baseline.get("vq_vocab_size"))

print("mode:", args.get("mode"))
print("token_only:", args.get("token_only"))
print("vq_only:", args.get("vq_only"))
print("main_target:", args.get("main_target"))
print("aux_lambda:", args.get("aux_lambda"))
print("d_model:", args.get("d_model"))
print("n_layers:", args.get("n_layers"))
print("n_heads:", args.get("n_heads"))
print("dropout:", args.get("dropout"))
print("seed:", args.get("seed"))

if args:
    if args.get("mode") != "finetune":
        raise ValueError(
            "BPE baseline was not trained in finetune mode: "
            f"mode={args.get('mode')}"
        )

    if not args.get("token_only", False):
        raise ValueError(
            "Checkpoint is not a token-only BPE baseline"
        )

    architecture_checks = [
        ("d_model", expected_d_model),
        ("n_layers", expected_n_layers),
        ("n_heads", expected_n_heads),
        ("seed", expected_seed),
    ]

    for name, expected in architecture_checks:
        actual = args.get(name)

        if actual is not None and int(actual) != expected:
            raise ValueError(
                "BPE baseline architecture mismatch: "
                f"{name} expected={expected}, "
                f"actual={actual}"
            )

print("[check] data and BPE baseline are compatible")
print("============================================================")
PY


# ============================================================
# BPE + VQW → BPE fine-tuning
#
# 入力:
#   BPE[t] + VQW[t]
#
# 主正解:
#   BPE[t+1]
#
# 補助正解:
#   VQW[t+1]
#
# --init_source bpe:
#   Step 6のBPE-only checkpointから初期化
#
# --mode finetune:
#   BPEとVQWをconcatしてinput_fusionを使用
#
# freezeは行わず、モデル全体をfine-tuneする。
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "[start BPE + VQW to BPE fine-tuning]"
echo "input        = BPE[t] + VQW[t]"
echo "main target  = BPE[t+1]"
echo "aux target   = VQW[t+1]"
echo "init source  = BPE-only baseline"
echo "data         = ${DATA_PATH}"
echo "baseline     = ${BPE_BASELINE_PATH}"
echo "run          = ${RUN}"
echo "============================================================"

python ar.py \
  --mode finetune \
  --data "${DATA_PATH}" \
  --token_vocab_size "${BPE_VOCAB_SIZE}" \
  --vq_vocab_size "${VQ_CODEBOOK_SIZE}" \
  --init_from "${BPE_BASELINE_PATH}" \
  --init_source bpe \
  --main_target tok \
  --aux_lambda "${AUX_LAMBDA}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --n_heads "${N_HEADS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --seed "${AR_SEED}" \
  --out "${BEST_PATH}" \
  2>&1 | tee "${LOG_PATH}"


# ============================================================
# 生成物確認
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
# 評価ログ表示
# ============================================================

echo "============================================================"
echo "[evaluation summary]"
echo "============================================================"

grep -E \
  "\[eval\]|\[save\]|\[loss-weight\]|\[init_from\]|\[init_source\]|\[bpe init\]" \
  "${LOG_PATH}" \
  || true


# ============================================================
# 最良 test_tok_ppl を抽出
# ============================================================

python - <<PY
import re

log_path = "${LOG_PATH}"

pattern = re.compile(
    r"\[eval\]\s+"
    r"ep=(\d+).*?"
    r"test_tok_ppl=([0-9.]+)"
)

results = []

with open(
    log_path,
    "r",
    encoding="utf-8",
    errors="replace",
) as f:
    for line in f:
        match = pattern.search(line)

        if match:
            epoch = int(match.group(1))
            test_tok_ppl = float(match.group(2))
            results.append(
                (epoch, test_tok_ppl)
            )

print("============================================================")
print("[test token perplexity]")

if not results:
    print("No test_tok_ppl entries found")
else:
    best_epoch, best_ppl = min(
        results,
        key=lambda item: item[1],
    )

    print("lowest test_tok_ppl epoch:", best_epoch)
    print("lowest test_tok_ppl:", best_ppl)
    print("------------------------------------------------------------")

    for epoch, ppl in results:
        marker = (
            " <-- lowest test PPL"
            if epoch == best_epoch
            else ""
        )

        print(
            f"ep={epoch:02d} "
            f"test_tok_ppl={ppl:.4f}"
            f"{marker}"
        )

print("============================================================")
PY


# ============================================================
# 出力checkpoint確認
# ============================================================

python - <<PY
import torch

paths = [
    "${BEST_PATH}",
    "${LAST_PATH}",
]

expected_token_vocab_size = ${BPE_VOCAB_SIZE}
expected_vq_vocab_size = ${VQ_CODEBOOK_SIZE}
expected_d_model = ${D_MODEL}

for path in paths:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    print("============================================================")
    print("[output checkpoint verification]")
    print("path:", path)
    print("keys:", list(checkpoint.keys()))

    if "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain model: {path}"
        )

    model = checkpoint["model"]

    required_keys = [
        "tok_emb.weight",
        "vq_emb.weight",
        "input_fusion.weight",
        "input_fusion.bias",
        "tok_head.weight",
        "vq_head.weight",
    ]

    for key in required_keys:
        if key not in model:
            raise KeyError(
                f"Output checkpoint does not contain: {key}"
            )

    tok_emb_shape = tuple(
        model["tok_emb.weight"].shape
    )

    vq_emb_shape = tuple(
        model["vq_emb.weight"].shape
    )

    fusion_shape = tuple(
        model["input_fusion.weight"].shape
    )

    tok_head_shape = tuple(
        model["tok_head.weight"].shape
    )

    vq_head_shape = tuple(
        model["vq_head.weight"].shape
    )

    print("tok_emb shape:", tok_emb_shape)
    print("vq_emb shape:", vq_emb_shape)
    print("input_fusion shape:", fusion_shape)
    print("tok_head shape:", tok_head_shape)
    print("vq_head shape:", vq_head_shape)

    actual_token_vocab_size = int(
        model["tok_emb.weight"].shape[0]
    )

    actual_vq_vocab_size = int(
        model["vq_emb.weight"].shape[0]
    )

    if actual_token_vocab_size != expected_token_vocab_size:
        raise ValueError(
            "Output BPE vocabulary mismatch: "
            f"expected={expected_token_vocab_size:,}, "
            f"actual={actual_token_vocab_size:,}"
        )

    # ar.pyはpadding IDを追加する可能性があるため、
    # VQ embeddingはcodebook size以上であることを確認する。
    if actual_vq_vocab_size < expected_vq_vocab_size:
        raise ValueError(
            "Output VQ vocabulary is too small: "
            f"expected at least={expected_vq_vocab_size:,}, "
            f"actual={actual_vq_vocab_size:,}"
        )

    if int(model["tok_emb.weight"].shape[1]) != expected_d_model:
        raise ValueError(
            "Output tok_emb d_model mismatch"
        )

    if fusion_shape != (
        expected_d_model,
        expected_d_model * 2,
    ):
        raise ValueError(
            "Unexpected input_fusion shape: "
            f"expected="
            f"({expected_d_model}, {expected_d_model * 2}), "
            f"actual={fusion_shape}"
        )

    args = checkpoint.get("args", {})

    print("epoch:", checkpoint.get("epoch"))
    print("valid_loss:", checkpoint.get("valid_loss"))
    print("test_loss:", checkpoint.get("test_loss"))
    print("token_vocab_size:", checkpoint.get("token_vocab_size"))
    print("vq_vocab_size:", checkpoint.get("vq_vocab_size"))

    print("mode:", args.get("mode"))
    print("init_source:", args.get("init_source"))
    print("init_from:", args.get("init_from"))
    print("token_only:", args.get("token_only"))
    print("vq_only:", args.get("vq_only"))
    print("main_target:", args.get("main_target"))
    print("aux_lambda:", args.get("aux_lambda"))

    if args:
        if args.get("mode") != "finetune":
            raise ValueError(
                f"Unexpected output mode: {args.get('mode')}"
            )

        if args.get("init_source") != "bpe":
            raise ValueError(
                "Output was not initialized with "
                "--init_source bpe"
            )

        if args.get("token_only", False):
            raise ValueError(
                "Output is unexpectedly token_only"
            )

        if args.get("vq_only", False):
            raise ValueError(
                "Output is unexpectedly vq_only"
            )

print("============================================================")
print("[check] output checkpoints OK")
print("============================================================")
PY


# ============================================================
# FTPへアップロード
# ============================================================

echo "============================================================"
echo "[upload files]"
echo "run = ${RUN}"
echo "============================================================"

lftp -u "${FTP_USER}","${FTP_PASS}" "${FTP_HOST}" <<EOF
set ftp:ssl-allow no
set net:max-retries 5
set net:timeout 30
set cmd:fail-exit yes

mkdir -p vqword_logs
cd vqword_logs

put "${BEST_PATH}" \
  -o "${RUN}.pt"

put "${LAST_PATH}" \
  -o "${RUN}_last.pt"

put "${LOG_PATH}" \
  -o "${RUN}.log"

bye
EOF


# ============================================================
# 完了表示
# ============================================================

echo "============================================================"
echo "[completed]"
echo "TASK           = TinyStories BPE + VQW to BPE"
echo "INIT SOURCE    = BPE-only baseline"
echo "BPE vocabulary = ${BPE_VOCAB_SIZE}"
echo "VQW codebook   = ${VQ_CODEBOOK_LABEL}"
echo "VQ context     = left ${HOP}"
echo "AR seed        = ${AR_SEED}"
echo "AUX lambda     = ${AUX_LAMBDA}"
echo "DATA           = ${DATA}"
echo "BPE BASELINE   = vqword_logs/${BPE_BASELINE}"
echo "BEST           = vqword_logs/${RUN}.pt"
echo "LAST           = vqword_logs/${RUN}_last.pt"
echo "LOG            = vqword_logs/${RUN}.log"
echo "============================================================"