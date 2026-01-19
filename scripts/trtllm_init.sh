#!/usr/bin/env bash
set -euo pipefail

MODEL="${TRTLLM_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ENGINE_NAME="${TRTLLM_ENGINE_NAME:-tinyllama-1.1b-chat}"
OUT_DIR="/trt_engines/${ENGINE_NAME}"

MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-1}"
MAX_INPUT_LEN="${TRTLLM_MAX_INPUT_LEN:-512}"
MAX_SEQ_LEN="${TRTLLM_MAX_SEQ_LEN:-768}"
WORKSPACE_MB="${TRTLLM_WORKSPACE_MB:-512}"
DISABLE_FP8="${TRTLLM_DISABLE_FP8:-1}"

echo "[trtllm-init] model: ${MODEL}"
echo "[trtllm-init] engine dir: ${OUT_DIR}"
echo "[trtllm-init] max_batch=${MAX_BATCH_SIZE} max_input_len=${MAX_INPUT_LEN} max_seq_len=${MAX_SEQ_LEN} workspace_mb=${WORKSPACE_MB} disable_fp8=${DISABLE_FP8}"

# se já tem engine pronta, não refaz
if [ -d "${OUT_DIR}" ] && [ "$(ls -A "${OUT_DIR}" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[trtllm-init] engine already exists, skipping build."
  exit 0
fi

apt-get update -y >/dev/null
apt-get install -y git git-lfs >/dev/null
git lfs install >/dev/null

mkdir -p /work
cd /work

# clone TensorRT-LLM
if [ ! -d "/work/TensorRT-LLM/.git" ]; then
  echo "[trtllm-init] cloning TensorRT-LLM repo..."
  git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git /work/TensorRT-LLM
else
  echo "[trtllm-init] TensorRT-LLM repo already exists"
fi

# clone HF model
rm -rf /work/model
echo "[trtllm-init] cloning HF model repo..."
GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/${MODEL}" /work/model

# token HF (se tiver)
if [ -n "${HF_TOKEN:-}" ]; then
  echo "[trtllm-init] configuring HF token"
  git config --global credential.helper store
  printf "https://user:%s@huggingface.co\n" "$HF_TOKEN" > ~/.git-credentials
fi

echo "[trtllm-init] pulling LFS files..."
cd /work/model
git lfs pull

# achar convert_checkpoint.py (llama)
echo "[trtllm-init] locating llama convert_checkpoint.py..."
CONVERT_SCRIPT="$(find /work/TensorRT-LLM -type f -name convert_checkpoint.py | grep -i llama | head -n 1 || true)"
if [ -z "${CONVERT_SCRIPT}" ]; then
  echo "[trtllm-init] ERROR: could not find convert_checkpoint.py for llama inside TensorRT-LLM repo"
  exit 1
fi
echo "[trtllm-init] using convert script: ${CONVERT_SCRIPT}"

# converter HF -> ckpt TRTLLM
rm -rf /work/ckpt_trt
mkdir -p /work/ckpt_trt

echo "[trtllm-init] converting checkpoint (llama)..."
python3 "${CONVERT_SCRIPT}" \
  --model_dir /work/model \
  --output_dir /work/ckpt_trt \
  --dtype float16

mkdir -p "${OUT_DIR}"

# montar flags opcionais (workspaces/FP8 mudam por versão)
EXTRA_FLAGS=()

# workspace (algumas versões aceitam --workspace_size, outras --workspace_size_mb, outras nada)
if trtllm-build --help 2>&1 | grep -q -- "--workspace"; then
  if trtllm-build --help 2>&1 | grep -q -- "--workspace_size_mb"; then
    EXTRA_FLAGS+=(--workspace_size_mb "${WORKSPACE_MB}")
  elif trtllm-build --help 2>&1 | grep -q -- "--workspace_size"; then
    # muitas vezes é em bytes
    EXTRA_FLAGS+=(--workspace_size "$((WORKSPACE_MB * 1024 * 1024))")
  fi
fi

# FP8 (se você quiser forçar OFF)
if [ "${DISABLE_FP8}" = "1" ]; then
  # tentativa de detectar flags comuns
  if trtllm-build --help 2>&1 | grep -q -- "--fp8"; then
    # se existir algo como --fp8 / --enable_fp8, a gente tenta desligar
    if trtllm-build --help 2>&1 | grep -q -- "--enable_fp8"; then
      EXTRA_FLAGS+=(--enable_fp8 false)
    fi
  fi
fi

echo "[trtllm-init] building TensorRT-LLM engine..."
echo "[trtllm-init] extra flags: ${EXTRA_FLAGS[*]:-(none)}"

trtllm-build \
  --checkpoint_dir /work/ckpt_trt \
  --output_dir "${OUT_DIR}" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_input_len "${MAX_INPUT_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  "${EXTRA_FLAGS[@]}" || {
    echo "[trtllm-init] ERROR: trtllm-build failed. Showing help snippet for debugging:"
    trtllm-build --help | head -n 80 || true
    exit 1
  }

# ============================================================
# ADICIONADO: copiar tokenizer/config HF para o OUT_DIR
# (o serve precisa disso; sem, dá 'architectures None' e cai)
# ============================================================
echo "[trtllm-init] copying HF tokenizer/config into engine dir..."

HF_DIR="/work/model"

# Tokenizer (um desses normalmente existe)
cp -av "${HF_DIR}/tokenizer.json" "${OUT_DIR}/" 2>/dev/null || true
cp -av "${HF_DIR}/tokenizer.model" "${OUT_DIR}/" 2>/dev/null || true
cp -av "${HF_DIR}/tokenizer"* "${OUT_DIR}/" 2>/dev/null || true

# Mapas/configs comuns
cp -av "${HF_DIR}/tokenizer_config.json" "${OUT_DIR}/" 2>/dev/null || true
cp -av "${HF_DIR}/special_tokens_map.json" "${OUT_DIR}/" 2>/dev/null || true
cp -av "${HF_DIR}/generation_config.json" "${OUT_DIR}/" 2>/dev/null || true

# IMPORTANTE: NÃO sobrescrever o config.json do TRT-LLM
if [ -f "${HF_DIR}/config.json" ]; then
  cp -av "${HF_DIR}/config.json" "${OUT_DIR}/hf_config.json"
fi

echo "[trtllm-init] final engine dir contents:"
ls -lah "${OUT_DIR}" || true

echo "[trtllm-init] done. Engine at: ${OUT_DIR}"
