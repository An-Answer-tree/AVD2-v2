from .operators import *
import torch, json, pandas

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import torch.nn.functional as F


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, geometry_path=None, metadata_path=None,
        ######## Tong Liu ADD for depth procession ########
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=None,
        time_division_factor=4,
        time_division_remainder=1,
        ######## Tong Liu ADD for depth procession ########
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.geometry_path = geometry_path  # 2025_12/22_17:25 [Tong Liu ADD] Geometry Directory
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None

        ######## Tong Liu ADD for depth procession ########
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        ######## Tong Liu ADD for depth procession ########

        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
    
    def _process_raw_points(
        self,
        point_map: np.ndarray,
        mask: np.ndarray,
        max_dist: float = 20.0,
        fill_holes: bool = True
    ) -> np.ndarray:
        """Cleans raw point cloud data and converts it to a normalized uint8 depth map.

        This process includes extracting the Z-coordinate, handling invalid regions based
        on the mask, clipping outliers, inverting (so near is bright), and filling holes.

        Args:
            point_map: A numpy array of shape (F, H, W, 3) containing raw XYZ coordinates.
            mask: A boolean numpy array of shape (F, H, W) indicating valid pixels.
            max_dist: The cutoff distance in meters. Points further than this are clipped
                to this value. Defaults to 20.0.
            fill_holes: If True, applies a median blur to fill sparse holes typically
                caused by point cloud projection. Defaults to True.

        Returns:
            A numpy array of shape (F, H, W) with dtype uint8. Values are normalized
            inverse depth where 255 represents 0m (Near) and 0 represents max_dist (Far).
        """
        # 1. Extract Z-depth. Use copy() to ensure memory independence and continuity.
        depth = point_map[..., 2].copy()

        # 2. Handle invalid regions by setting them to the maximum distance.
        depth[~mask] = max_dist

        # 3. Clip outliers to the defined range to prevent normalization skew.
        depth = np.clip(depth, 0, max_dist)

        # 4. Invert & Normalize: Distance 0 -> 1.0 (White), Distance max_dist -> 0.0 (Black).
        depth_normalized = 1.0 - (depth / max_dist)

        # 5. Convert to uint8 range [0, 255] for efficient storage.
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # 6. Denoise and fill holes using Median Blur.
        if fill_holes:
            processed_frames = []
            for i in range(depth_uint8.shape[0]):
                # Median blur is effective for removing salt-and-pepper noise (holes)
                # while preserving edges better than Gaussian blur.
                processed_frames.append(cv2.medianBlur(depth_uint8[i], 3))
            depth_uint8 = np.array(processed_frames)

        return depth_uint8

    def get_depth_tensor(
        self,
        raw_point_map: np.ndarray,
        mask: np.ndarray
    ) -> torch.Tensor:
        """Converts raw geometry data into a normalized training tensor.

        This function ensures strict alignment with the RGB video loader by:
        1. Trimming frames to satisfy the VAE's temporal requirement (frame_count % 4 == 1).
        2. Applying 'Scale Shortest Edge' and 'Center Crop' resizing logic instead of simple
           interpolation, ensuring spatial alignment with RGB frames.

        Args:
            raw_point_map: Raw numpy array of shape (F_total, H, W, 3).
            mask: Raw numpy mask of shape (F_total, H, W).

        Returns:
            A torch.Tensor of shape (1, F, H, W) with dtype float32.
            The values are normalized to the range [-1, 1].
        """
        # --- Step 1: Clean and obtain uint8 depth ---
        # Shape: (F_total, Raw_H, Raw_W)
        depth_uint8 = self._process_raw_points(raw_point_map, mask)
        total_frames, raw_h, raw_w = depth_uint8.shape

        # --- Step 2: Temporal Alignment (Match LoadVideo logic) ---
        # Ensure the number of frames satisfies the VAE requirement (e.g., % 4 == 1).
        # We start with self.num_frames (target) or total available frames, whichever is smaller.
        num_frames = self.num_frames or 49
        actual_num_frames = min(num_frames, total_frames)
        
        # Iteratively reduce frame count until it matches the temporal divisor rule.
        # Logic mirrors: LoadVideo.get_num_frames
        factor = self.time_division_factor or 4
        remainder = self.time_division_remainder or 1
        
        while (actual_num_frames > 1 and 
               actual_num_frames % factor != remainder):
            actual_num_frames -= 1
        
        # Slice the frames strictly from the beginning (assuming sequential read).
        depth_sampled = depth_uint8[:actual_num_frames] 
        
        # --- Step 3: To Tensor & Normalize ---
        # Convert to float [0, 1] and add Channel dim: (F, 1, H, W)
        depth_tensor = torch.from_numpy(depth_sampled).float() / 255.0
        depth_tensor = depth_tensor.unsqueeze(1)

        # --- Step 4: Spatial Alignment (Match ImageCropAndResize logic) ---
        # Target dimensions
        target_h = self.height or 480
        target_w = self.width or 832

        h_div = self.height_division_factor or 16
        w_div = self.width_division_factor or 16

        # Set target height / width to be the multiple of 16
        if target_h is not None:
            target_h = target_h // h_div * h_div
        if target_w is not None:
            target_w = target_w // w_div * w_div

        # Calculate scale to fit the shortest edge (filling the target area).
        scale = max(target_w / raw_w, target_h / raw_h)
        scaled_h = round(raw_h * scale)
        scaled_w = round(raw_w * scale)

        # Resize (Bilinear interpolation)
        if (scaled_h, scaled_w) != (raw_h, raw_w):
            depth_tensor = F.interpolate(
                depth_tensor,
                size=(scaled_h, scaled_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Center Crop to target dimensions
        if scaled_h != target_h or scaled_w != target_w:
            h_start = (scaled_h - target_h) // 2
            w_start = (scaled_w - target_w) // 2
            depth_tensor = depth_tensor[
                :, :, h_start : h_start + target_h, w_start : w_start + target_w
            ]

        # --- Step 5: Final Normalization ---
        # Scale to [-1, 1] to match standard RGB normalization.
        depth_tensor = (depth_tensor * 2.0) - 1.0
        
        # Permute to (C, F, H, W) -> (1, F, H, W)
        depth_tensor = depth_tensor.permute(1, 0, 2, 3)

        return depth_tensor

    def _get_depth(self, data: Dict[str, str]) -> torch.Tensor:
        """Loads and processes depth data corresponding to the video.

        Args:
            data: Metadata dictionary containing the 'video' file path.

        Returns:
            A processed depth tensor of shape (1, F, H, W).

        Raises:
            FileNotFoundError: If the geometry .npz file is missing.
            KeyError: If the .npz file is missing required keys ('point_map', 'mask').
        """
        # 1. Resolve paths using pathlib.
        video_dir = Path(self.base_path)
        video_rel_path = video_dir / data["video"]
        # Assuming geometry files share the same stem as video files.
        geometry_path = Path(self.geometry_path) / f"{video_rel_path.stem}.npz"

        if not geometry_path.exists():
            raise FileNotFoundError(f"Geometry file not found: {geometry_path}")

        # 2. Load Geometry Data.
        try:
            with np.load(geometry_path) as geometry:
                # We call get_depth_tensor directly. It handles sampling and resizing internally
                # by accessing self.height, self.width, etc.
                depth_tensor = self.get_depth_tensor(
                    raw_point_map=geometry["point_map"],
                    mask=geometry["mask"]
                )
        except KeyError as e:
            raise KeyError(f"Missing keys in {geometry_path}: {e}")

        return depth_tensor

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()

            # [Tong Liu ADD] Load Depth information.
            # Depth is loaded before the main operator converts the video path string to a Tensor,
            # ensuring _get_depth can still access the filename string.
            if "video" in data:
                try:
                    data["depth"] = self._get_depth(data)
                except Exception as e:
                    raise RuntimeError(f"Error: Failed to load depth for {data['video']}: {e}")

            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
