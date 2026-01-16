#!/usr/bin/env bash
set -euo pipefail

source /baai-cwm-vepfs/cwm/cheng.li/.bashrc
conda activate diffsynth

REPO_ROOT="/baai-cwm-vepfs/cwm/cheng.li/AVD2-v2"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/huggingface_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/modelscope_cache}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

NUM_WORKERS="${NUM_WORKERS:-8}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#_GPU_ARR[@]}"
fi

DATASET_BASE_PATH="/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos"
GEOMETRY_PATH="/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine"
METADATA_CSV="/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/video1.csv"
TTC_JSON="/baai-cwm-vepfs/cwm/cheng.li/qwen3vl_workspace/calculate_ttc_logs/ttc_results_20251222_113546.json"

HIGH_LORA="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors"
LOW_LORA="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors"

OUT_ROOT="${OUT_ROOT:-/baai-cwm-backup/cwm/tong.liu/outputckpt}"
OUT_HIGH="${OUT_ROOT}/Wan2.2-I2V-A14B_high_noise_ttc_embedderfree"
OUT_LOW="${OUT_ROOT}/Wan2.2-I2V-A14B_low_noise_ttc_embedderfree"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-368}"
NUM_FRAMES="${NUM_FRAMES:-49}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"

TASK="${TASK:-dual_head_sft}"
TRAINABLE_MODELS="${TRAINABLE_MODELS:-dit.ttc_embedder,dit.depth_head}"

FREEZE_PRESET_LORA="${FREEZE_PRESET_LORA:-1}"

if [[ -z "${REMOVE_PREFIX_IN_CKPT:-}" ]]; then
  if [[ "${TRAINABLE_MODELS}" == *","* ]]; then
    REMOVE_PREFIX_IN_CKPT="pipe.dit."
  else
    REMOVE_PREFIX_IN_CKPT="pipe.dit.ttc_embedder."
  fi
fi

TRAIN_SCRIPT="${REPO_ROOT}/examples/wanvideo/model_training/train.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: train script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_BASE_PATH}" ]]; then
  echo "ERROR: DATASET_BASE_PATH not found: ${DATASET_BASE_PATH}" >&2
  exit 1
fi
if [[ ! -d "${GEOMETRY_PATH}" ]]; then
  echo "ERROR: GEOMETRY_PATH not found: ${GEOMETRY_PATH}" >&2
  exit 1
fi
if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "ERROR: METADATA_CSV not found: ${METADATA_CSV}" >&2
  exit 1
fi
if [[ ! -f "${TTC_JSON}" ]]; then
  echo "ERROR: TTC_JSON not found: ${TTC_JSON}" >&2
  exit 1
fi
if [[ ! -f "${HIGH_LORA}" ]]; then
  echo "ERROR: HIGH_LORA not found: ${HIGH_LORA}" >&2
  exit 1
fi
if [[ ! -f "${LOW_LORA}" ]]; then
  echo "ERROR: LOW_LORA not found: ${LOW_LORA}" >&2
  exit 1
fi

mkdir -p "${OUT_HIGH}" "${OUT_LOW}"

COMMON_ARGS=(
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_geometry_path "${GEOMETRY_PATH}"
  --dataset_metadata_path "${METADATA_CSV}"
  --data_file_keys "video"
  --ttc_json_path "${TTC_JSON}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --dataset_repeat "${DATASET_REPEAT}"
  --trainable_models "${TRAINABLE_MODELS}"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --remove_prefix_in_ckpt "${REMOVE_PREFIX_IN_CKPT}"
  --extra_inputs "input_image,ttc"
  --use_gradient_checkpointing
  --use_gradient_checkpointing_offload
  --task "${TASK}"
  --dataset_num_workers "${NUM_WORKERS}"
)

if [[ "${FREEZE_PRESET_LORA}" == "1" ]]; then
  COMMON_ARGS+=(--freeze_preset_lora)
fi

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "${HIGH_LORA}" \
  --preset_lora_model "dit" \
  --output_path "${OUT_HIGH}" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "$((MASTER_PORT + 1))" \
  "${TRAIN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "${LOW_LORA}" \
  --preset_lora_model "dit" \
  --output_path "${OUT_LOW}" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
