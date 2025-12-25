import torch
import os
import sys
from torch.utils.data import DataLoader
from PIL import Image

# 尝试引入 diffsynth 库
try:
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import LoadVideo, LoadAudio, ToAbsolutePath
except ImportError:
    print("❌ 错误: 找不到 diffsynth 库，请确保环境变量设置正确。")
    exit()

# ==========================================
# 1. 配置区域 (基于你的真实环境)
# ==========================================
class MockArgs:
    def __init__(self):
        # [真实路径]
        self.dataset_base_path = "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos"
        self.dataset_geometry_path = "/baai-cwm-backup/cwm/tong.liu/Geo_Out"
        self.dataset_metadata_path = "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/metadata.csv"
        
        # [视频参数]
        self.height = 480
        self.width = 832
        self.num_frames = 49
        
        # Resize 参数 (通常保持默认或根据显存调整)
        self.max_pixels = 512 * 512 
        
        # 数据集参数
        self.dataset_repeat = 1
        self.data_file_keys = "video" 

args = MockArgs()

# ==========================================
# 2. 主测试逻辑
# ==========================================
def test_dataloader_final():
    print(f"🚀 开始测试 DataLoader (Metadata 模式)...")
    print(f"📂 Video Path: {args.dataset_base_path}")
    print(f"📂 Geo Path:   {args.dataset_geometry_path}")
    print(f"📄 Metadata:   {args.dataset_metadata_path}")

    # 1. 初始化 UnifiedDataset
    # 这次我们会传入 metadata_path，让它自己去读 CSV
    try:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            geometry_path=args.dataset_geometry_path, # [你的 Depth 代码生效处]
            metadata_path=args.dataset_metadata_path, # [读取 CSV]
            
            # 尺寸参数
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            
            # 视频加载算子 (只做 Resize, 不转 Tensor)
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            ),
        )
        print(f"✅ Dataset 初始化成功，总数据量: {len(dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset 初始化失败: {e}")
        # 常见错误提示
        if "No such file" in str(e):
            print("   -> 请检查 metadata.csv 文件路径是否正确。")
        return

    # 2. 初始化 DataLoader
    # 【关键】collate_fn=lambda x: x[0]
    # 意味着 DataLoader 取出一个样本后，直接把该样本(dict)传出来，不进行任何 Tensor 打包
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=lambda x: x[0] 
    )

    print("\n🔄 开始读取前 2 个样本...")
    
    try:
        for i, batch in enumerate(dataloader):
            if i >= 2: break
            
            print(f"\n--- Sample {i} ---")
            # 此时 batch 就是一个普通的 python dict
            
            # 检查 Video (预期: List of PIL)
            if "video" in batch:
                video_data = batch["video"]
                print(f"  🎬 Key: 'video'")
                print(f"     Type: {type(video_data)}") # <class 'list'>
                
                if isinstance(video_data, list) and len(video_data) > 0:
                    first_frame = video_data[0]
                    print(f"     Content: List of {type(first_frame)}") # <class 'PIL.Image.Image'>
                    print(f"     Length: {len(video_data)} frames")
                    # PIL size 是 (Width, Height)
                    print(f"     Size: {first_frame.size} (Expected: ({args.width}, {args.height}))")
            
            # 检查 Depth (预期: Tensor)
            if "depth" in batch:
                depth_data = batch["depth"]
                print(f"  🧊 Key: 'depth'")
                print(f"     Type: {type(depth_data)}") # <class 'torch.Tensor'>
                
                if isinstance(depth_data, torch.Tensor):
                    print(f"     Shape: {depth_data.shape}") 
                    # 预期: [1, 1, 49, 480, 832] (如果你的代码带batch dim) 
                    # 或者 [1, 49, 480, 832] (如果你的代码不带batch dim)
                    
                    print(f"     Range: min={depth_data.min():.2f}, max={depth_data.max():.2f}")
                    
                    # 简单验证一下数值是否合理
                    if depth_data.max() > 1.1 or depth_data.min() < -1.1:
                        print("     ⚠️ 警告: Depth 数值范围似乎没有归一化到 [-1, 1]")
                    else:
                        print("     ✅ 数值范围正常 (Normalized)")

            # 检查 Prompt
            if "prompt" in batch:
                print(f"  📝 Key: 'prompt' | Content: {str(batch['prompt'])[:50]}...")

    except Exception as e:
        import traceback
        print("\n❌ 迭代过程报错:")
        traceback.print_exc()

    print("\n✅ 测试结束。")

if __name__ == "__main__":
    test_dataloader_final()