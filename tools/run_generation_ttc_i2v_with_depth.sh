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

MODEL_ROOT="${MODEL_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B}"
VIDEO_ROOT="${VIDEO_ROOT:-/baai-cwm-backup/cwm/tong.liu/newfulldemo}"
CSV_PATH="${CSV_PATH:-/baai-cwm-backup/cwm/tong.liu/video_prompts.csv}"
OUT_DIR="${OUT_DIR:-/baai-cwm-backup/cwm/tong.liu/outputs/wangen_with_depth}"

# Optional: JSON mapping {"<video_stem>": [ttc_seq]}
TTC_JSON_PATH="${TTC_JSON_PATH:-}"

# Stage1 LoRA (optional)
LORA_HIGH="${LORA_HIGH:-}"
LORA_LOW="${LORA_LOW:-}"

# Stage2 TTC ckpt (optional)
TTC_CKPT_HIGH="${TTC_CKPT_HIGH:-}"
TTC_CKPT_LOW="${TTC_CKPT_LOW:-}"

# Stage3 depth head ckpt (optional but required when --save_depth)
DEPTH_CKPT="${DEPTH_CKPT:-}"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-368}"
NUM_FRAMES="${NUM_FRAMES:-145}"
STEPS="${STEPS:-50}"
FPS="${FPS:-24}"
SEED="${SEED:-0}"

CMD=(
  torchrun --nproc_per_node "${NUM_GPUS}" \
    "${REPO_ROOT}/tools/generation_ttc_i2v_with_depth.py" \
    --model_root "${MODEL_ROOT}" \
    --video_root "${VIDEO_ROOT}" \
    --csv_path "${CSV_PATH}" \
    --out_dir "${OUT_DIR}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --num_steps "${STEPS}" \
    --fps "${FPS}" \
    --base_seed "${SEED}"
)

if [[ -n "${TTC_JSON_PATH}" ]]; then
  CMD+=( --ttc_json_path "${TTC_JSON_PATH}" )
fi

if [[ -n "${LORA_HIGH}" ]]; then
  CMD+=( --lora_high "${LORA_HIGH}" )
fi
if [[ -n "${LORA_LOW}" ]]; then
  CMD+=( --lora_low "${LORA_LOW}" )
fi

if [[ -n "${TTC_CKPT_HIGH}" ]]; then
  CMD+=( --ttc_ckpt_high "${TTC_CKPT_HIGH}" )
fi
if [[ -n "${TTC_CKPT_LOW}" ]]; then
  CMD+=( --ttc_ckpt_low "${TTC_CKPT_LOW}" )
fi

# Enable depth output
if [[ -n "${DEPTH_CKPT}" ]]; then
  CMD+=( --save_depth --depth_ckpt "${DEPTH_CKPT}" )
fi

mkdir -p "${OUT_DIR}"

"${CMD[@]}"
