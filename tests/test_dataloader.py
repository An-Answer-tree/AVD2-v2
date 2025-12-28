import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os

# --- Imports ---
try:
    from diffsynth.core.data.unified_dataset_with_ttc import UnifiedDatasetWithTTC, UnifiedDataset
except ImportError:
    # 这里的 import 路径请根据实际情况调整
    print("Error: Could not import dataset classes. Make sure PYTHONPATH is correct.")
    sys.exit(1)

# --- Configuration ---
DATASET_BASE_PATH = "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos"
GEOMETRY_PATH = "/baai-cwm-backup/cwm/tong.liu/Geo_Out_Fine"
METADATA_CSV = "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/video1.csv"
TTC_JSON = "/baai-cwm-vepfs/cwm/cheng.li/qwen3vl_workspace/calculate_ttc_logs/ttc_results_20251222_113546.json"

HEIGHT = 480
WIDTH = 832 
NUM_FRAMES = 49 
BATCH_SIZE = 1  # 强制为 1，配合 lambda x: x[0]

# --- Helper: Convert List[PIL] -> Tensor (C, F, H, W) ---
def process_video_output(video_frames):
    """
    Takes the output of LoadVideo (List of PIL Images) and converts 
    it to a normalized PyTorch tensor [C, F, H, W] in range [-1, 1].
    """
    if not isinstance(video_frames, list):
        return video_frames 
    
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply to all frames and stack
    tensors = [tf(frame) for frame in video_frames]
    if len(tensors) == 0:
        return torch.zeros(3, NUM_FRAMES, HEIGHT, WIDTH)
        
    video_tensor = torch.stack(tensors) # (F, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3) # -> (C, F, H, W)
    
    return video_tensor

def test_dataloader():
    print(f"--- Starting Dataloader Test ---")

    # 1. Define Base Operator
    base_video_op = UnifiedDataset.default_video_operator(
        base_path=DATASET_BASE_PATH,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        time_division_factor=4, 
        time_division_remainder=1, 
    )

    # 2. Define Final Operator
    final_video_op = lambda x: process_video_output(base_video_op(x))

    # 3. Initialize Dataset
    try:
        dataset = UnifiedDatasetWithTTC(
            ttc_json_path=TTC_JSON,
            base_path=DATASET_BASE_PATH,
            geometry_path=GEOMETRY_PATH,
            metadata_path=METADATA_CSV,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            data_file_keys=("video",),
            main_data_operator=final_video_op,
            filter_missing_ttc=True,
            filter_first_ttc_zero=False,
        )
    except Exception as e:
        print(f"\n[Fatal Error] Failed to initialize dataset: {e}")
        return

    print(f"Dataset initialized. Total samples: {len(dataset)}")

    # 4. Initialize DataLoader
    # 使用你指定的 lambda collate_fn，直接提取单个样本，不进行 Stack
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, # 必须为 1
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: x[0] 
    )

    # 5. Iterate and Inspect
    try:
        print("\n--- Fetching first sample (Batch Size=1) ---\n")
        for i, sample in enumerate(loader):
            # 注意：sample 是字典，不是 Batch 后的 Tensor
            # 形状应该是 [C, F, H, W] 而不是 [B, C, F, H, W]

            # Inspect Video
            if "video" in sample:
                video = sample["video"]
                print(f"Video Shape: {video.shape} | Type: {video.dtype}")
                # 期望: (3, 49, 480, 832)
                
            # Inspect Depth
            if "depth" in sample:
                depth = sample["depth"]
                print(f"Depth Shape: {depth.shape} | Type: {depth.dtype}")
                # 期望: (1, 49, 480, 832)

                # Verify Alignment (比较 F, H, W)
                # Video: (C, F, H, W) -> shape[1]=F, shape[2]=H, shape[3]=W
                if "video" in sample:
                    # 检查 F (帧数)
                    if video.shape[1] != depth.shape[1]:
                        print(f"[FAIL] Frame count mismatch! Video: {video.shape[1]} vs Depth: {depth.shape[1]}")
                    else:
                        print(f"[PASS] Frame count aligns: {video.shape[1]}")
                    
                    # 检查 H, W (空间尺寸)
                    if video.shape[2:] != depth.shape[2:]:
                        print(f"[FAIL] Spatial mismatch! Video: {video.shape[2:]} vs Depth: {depth.shape[2:]}")
                    else:
                        print(f"[PASS] Spatial dimensions align: {video.shape[2:]}")

            # Inspect TTC
            if "ttc" in sample:
                ttc = sample["ttc"]
                print(f"TTC Value (List): {ttc}")
                print(f"TTC Length: {len(ttc)}")

            print("\n--- Test Finished successfully ---")
            break 

    except Exception as e:
        print(f"\n[Error] Runtime error during iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()