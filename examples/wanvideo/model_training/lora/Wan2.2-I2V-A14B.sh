source /baai-cwm-vepfs/cwm/cheng.li/.bashrc
conda activate avd2v2
cd /baai-cwm-vepfs/cwm/cheng.li/AVD2-v2

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HOME="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/huggingface_cache"
export MODELSCOPE_CACHE="/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/modelscope_cache"

accelerate launch --num_processes 8 --main_process_port 29500 examples/wanvideo/model_training/train.py \
  --dataset_base_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos" \
  --dataset_geometry_path "/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine" \
  --dataset_metadata_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/video1.csv" \
  --ttc_json_path "/baai-cwm-vepfs/cwm/cheng.li/qwen3vl_workspace/calculate_ttc_logs/ttc_results_20251222_113546.json" \
  --height 224 \
  --width 224 \
  --num_frames  \
  --dataset_num_workers 16 \
  --task "dual_head_sft" \
  --trainable_models "dit.ttc_embedder,dit.depth_head" \
  --extra_inputs "input_image,ttc" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors" \
  --preset_lora_model "dit" \
  --output_path "/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_high_noise_ttc_embedder" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0

accelerate launch --num_processes 8 --main_process_port 29501 examples/wanvideo/model_training/train.py \
  --dataset_base_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos" \
  --dataset_geometry_path "/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine" \
  --dataset_metadata_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/video1.csv" \
  --ttc_json_path "/baai-cwm-vepfs/cwm/cheng.li/qwen3vl_workspace/calculate_ttc_logs/ttc_results_20251222_113546.json" \
  --height 224 \
  --width 224 \
  --num_frames 300 \
  --dataset_num_workers 16 \
  --task "dual_head_sft" \
  --trainable_models "dit.ttc_embedder,dit.depth_head" \
  --extra_inputs "input_image,ttc" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth" \
  --preset_lora_path "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors" \
  --preset_lora_model "dit" \
  --output_path "/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_low_noise_ttc_embedder" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.358