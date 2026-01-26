#!/usr/bin/env bash
set -euo pipefail

source /baai-cwm-vepfs/cwm/cheng.li/.bashrc
conda activate diffsynth

REPO_ROOT="${REPO_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/AVD2-v2}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export HF_HOME="${HF_HOME:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/huggingface_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/modelscope_cache}"

export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_DEVICE_MAX_CONNECTIONS=1

CSV="${CSV:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/video_ttestttc.csv}"
TTC_JSON="${TTC_JSON:-/baai-cwm-backup/cwm/tong.liu/ttcnew1.json}"
INIT_SRC_ROOT="${INIT_SRC_ROOT:-/baai-cwm-backup/cwm/tong.liu/newfulldemo/}"

OUT_VIDEO="${OUT_VIDEO:-/baai-cwm-backup/cwm/tong.liu/gen_video_out8}"
OUT_DEPTH="${OUT_DEPTH:-/baai-cwm-backup/cwm/tong.liu/gen_depth_out8}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.2-I2V-A14B}"
MODEL_ROOT="${MODEL_ROOT:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/Wan-AI/Wan2.2-I2V-A14B}"

HIGH_LORA="${HIGH_LORA:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors}"
LOW_LORA="${LOW_LORA:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors}"

TTC_HIGH_CKPT="${TTC_HIGH_CKPT:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_high_noise_ttc_only49_frozen_lora/epoch-4.safetensors}"
TTC_LOW_CKPT="${TTC_LOW_CKPT:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_low_noise_ttc_only49_frozen_lora/epoch-4.safetensors}"

DEPTH_HIGH_CKPT="${DEPTH_HIGH_CKPT:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_high_noise_depth_joint49_frozen_lora/epoch-4.safetensors}"
DEPTH_LOW_CKPT="${DEPTH_LOW_CKPT:-/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_low_noise_depth_joint49_frozen_lora/epoch-4.safetensors}"

WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-320}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_STEPS="${NUM_STEPS:-50}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"

CFG_SCALE="${CFG_SCALE:-1.0}"
CFG_MERGE="${CFG_MERGE:-false}"

SWITCH_DIT_BOUNDARY="${SWITCH_DIT_BOUNDARY:-0.90}"
SIGMA_SHIFT="${SIGMA_SHIFT:-5.0}"

TILED="${TILED:-true}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"

TTC_SAMPLING="${TTC_SAMPLING:-first}"
TTC_SCALE="${TTC_SCALE:-0.05}"

TOKENIZER_PATH="${TOKENIZER_PATH:-}"

SCRIPT="${SCRIPT:-${REPO_ROOT}/wan22_i2v_ttc_depth_generate.py}"

mkdir -p "${OUT_VIDEO}" "${OUT_DEPTH}"

ARGS=(
  --csv "${CSV}"
  --ttc_json "${TTC_JSON}"
  --init_src_root "${INIT_SRC_ROOT}"
  --out_video "${OUT_VIDEO}"
  --out_depth "${OUT_DEPTH}"
  --model_id "${MODEL_ID}"
  --model_root "${MODEL_ROOT}"
  --lora_high "${HIGH_LORA}"
  --lora_low "${LOW_LORA}"
  --ttc_ckpt_high "${TTC_HIGH_CKPT}"
  --ttc_ckpt_low "${TTC_LOW_CKPT}"
  --depth_ckpt_high "${DEPTH_HIGH_CKPT}"
  --depth_ckpt_low "${DEPTH_LOW_CKPT}"
  --lora_alpha "${LORA_ALPHA}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --num_frames "${NUM_FRAMES}"
  --num_steps "${NUM_STEPS}"
  --fps "${FPS}"
  --quality "${QUALITY}"
  --cfg_scale "${CFG_SCALE}"
  --cfg_merge "${CFG_MERGE}"
  --switch_dit_boundary "${SWITCH_DIT_BOUNDARY}"
  --sigma_shift "${SIGMA_SHIFT}"
  --tiled "${TILED}"
  --ttc_sampling "${TTC_SAMPLING}"
  --ttc_scale "${TTC_SCALE}"
)

if [[ -n "${TOKENIZER_PATH}" ]]; then
  ARGS+=(--tokenizer_path "${TOKENIZER_PATH}")
fi

NUM_SHARDS=2

CUDA_VISIBLE_DEVICES=0,1 \
python "${SCRIPT}" "${ARGS[@]}" \
  --device_high "cuda:0" --device_low "cuda:1" --decode_device "cuda:0" \
  --shard_index 0 --num_shards ${NUM_SHARDS} &

CUDA_VISIBLE_DEVICES=2,3 \
python "${SCRIPT}" "${ARGS[@]}" \
  --device_high "cuda:0" --device_low "cuda:1" --decode_device "cuda:0" \
  --shard_index 1 --num_shards ${NUM_SHARDS} &

wait