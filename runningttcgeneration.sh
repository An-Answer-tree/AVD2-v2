#!/usr/bin/env bash
set -euo pipefail

source /baai-cwm-vepfs/cwm/cheng.li/.bashrc
conda activate diffsynth

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export HF_HOME="${HF_HOME:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/huggingface_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/modelscope_cache}"
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="${REPO_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/AVD2-v2}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

CSV="${CSV:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/video_ttestttc.csv}"
TTC_JSON="${TTC_JSON:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/outputs/ttcexample.json}"
INIT_SRC_ROOT="${INIT_SRC_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos/}"
OUT="${OUT:-/baai-cwm-backup/cwm/tong.liu/ttcgeneration1/}"

MODEL_ROOT="${MODEL_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B}"
LORA_HIGH="${LORA_HIGH:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors}"
LORA_LOW="${LORA_LOW:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors}"

TTC_CKPT_HIGH="${TTC_CKPT_HIGH:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_high_noise_ttc_embedder141/epoch-4.safetensors}"
TTC_CKPT_LOW="${TTC_CKPT_LOW:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_low_noise_ttc_embedder141/epoch-4.safetensors}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

GEN_SCRIPT="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/generationttc.py"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${GEN_SCRIPT}" \
  --csv "${CSV}" \
  --ttc_json "${TTC_JSON}" \
  --init_src_root "${INIT_SRC_ROOT}" \
  --out "${OUT}" \
  --model_root "${MODEL_ROOT}" \
  --lora_high "${LORA_HIGH}" \
  --lora_low "${LORA_LOW}" \
  --ttc_ckpt_high "${TTC_CKPT_HIGH}" \
  --ttc_ckpt_low "${TTC_CKPT_LOW}"
