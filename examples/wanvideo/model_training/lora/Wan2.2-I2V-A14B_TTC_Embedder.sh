#!/usr/bin/env bash
set -euo pipefail

# Added by cheng li, tong liu

DATASET_BASE_PATH="/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos"
GEOMETRY_PATH="/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine"
METADATA_CSV="/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/video1.csv"
TTC_JSON="/baai-cwm-vepfs/cwm/cheng.li/qwen3vl_workspace/calculate_ttc_logs/ttc_results_20251222_113546.json"

HIGH_LORA="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors"
LOW_LORA="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors"

OUT_HIGH="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_ttc_embedder"
OUT_LOW="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_ttc_embedder"

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_geometry_path "${GEOMETRY_PATH}" \
  --dataset_metadata_path "${METADATA_CSV}" \
  --data_file_keys "video" \
  --ttc_json_path "${TTC_JSON}" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "${HIGH_LORA}" \
  --preset_lora_model "dit" \
  --trainable_models "dit.ttc_embedder" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --output_path "${OUT_HIGH}" \
  --remove_prefix_in_ckpt "pipe.dit.ttc_embedder." \
  --extra_inputs "input_image,ttc" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_geometry_path "${GEOMETRY_PATH}" \
  --dataset_metadata_path "${METADATA_CSV}" \
  --data_file_keys "video" \
  --ttc_json_path "${TTC_JSON}" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "${LOW_LORA}" \
  --preset_lora_model "dit" \
  --trainable_models "dit.ttc_embedder" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --output_path "${OUT_LOW}" \
  --remove_prefix_in_ckpt "pipe.dit.ttc_embedder." \
  --extra_inputs "input_image,ttc" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358
