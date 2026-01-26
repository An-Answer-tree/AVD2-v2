#!/usr/bin/env bash
set -euo pipefail

source /baai-cwm-vepfs/cwm/cheng.li/.bashrc
conda activate diffsynth

REPO_ROOT="${REPO_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/AVD2-v2}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export HF_HOME="${HF_HOME:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/huggingface_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/modelscope_cache}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#_GPU_ARR[@]}}"
NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/baai-cwm-backup/cwm/tong.liu/newfulldemo}"
GEOMETRY_PATH="${GEOMETRY_PATH:-/baai-cwm-backup/cwm/tong.liu/geodepthnew}"
METADATA_CSV="${METADATA_CSV:-}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-I2V-A14B}"
MODEL_ROOT="${MODEL_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B}"

OUT_ROOT="${OUT_ROOT:-/baai-cwm-backup/cwm/tong.liu/outputckpt}"
OUT_PATH="${OUT_PATH:-${OUT_ROOT}/Wan2.2-I2V-A14B_latent_depth_head_stage3}"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-368}"
NUM_FRAMES="${NUM_FRAMES:-145}"

LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
VAE_TILE_SIZE="${VAE_TILE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/examples/wanvideo/model_training/train_latent_depth_head.py}"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: train script not found: ${SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUT_PATH}"

ARGS=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_geometry_path "${GEOMETRY_PATH}"
  --data_file_keys "video"
  --model_id "${MODEL_ID}"
  --model_root "${MODEL_ROOT}"
  --output_path "${OUT_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -n "${METADATA_CSV}" ]]; then
  if [[ ! -f "${METADATA_CSV}" ]]; then
    echo "WARNING: METADATA_CSV not found: ${METADATA_CSV}. Auto-scan mode will be used." >&2
  else
    ARGS+=( --dataset_metadata_path "${METADATA_CSV}" )
  fi
fi

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  "${SCRIPT}" \
  "${ARGS[@]}"
