# Added by Cheng Li
"""Unified Dataset with Time-To-Collision (TTC) support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .unified_dataset import UnifiedDataset


class UnifiedDatasetWithTTC(UnifiedDataset):
    """Dataset class that extends UnifiedDataset to include TTC data.

    This class loads TTC information from a JSON file, filters the dataset based
    on TTC availability and properties, and aligns the TTC data length with
    video and depth frames.

    Attributes:
        ttc_json_path: Path to the JSON file containing TTC data.
        ttc_video_key: Key in the metadata to identify the video.
        ttc_field_name: Key name to store TTC data in the returned dictionary.
        filter_missing_ttc: Whether to filter out samples with missing TTC data.
        filter_first_ttc_zero: Whether to filter out samples where the first TTC value is zero.
        filter_no_accident_frame: Whether to filter out samples containing no accident frame (0.0).
        ttc_map: Dictionary mapping video IDs to TTC lists.
    """

    def __init__(
        self,
        ttc_json_path: str,
        base_path: Optional[str] = None,
        geometry_path: Optional[str] = None,  # 2025_12/22_17:25 [Tong Liu ADD] Geometry Directory
        metadata_path: Optional[str] = None,
        # -------- Tong Liu ADD for depth procession --------
        height: Optional[int] = None,
        width: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        num_frames: Optional[int] = None,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        # -------- Tong Liu ADD for depth procession --------
        repeat: int = 1,
        data_file_keys: Sequence[str] = (),
        main_data_operator=lambda x: x,
        special_operator_map: Optional[dict] = None,
        ttc_video_key: str = "video",
        ttc_field_name: str = "ttc",
        filter_missing_ttc: bool = True,
        filter_first_ttc_zero: bool = True,
        filter_no_accident_frame: bool = False,
    ):
        """Initializes the UnifiedDatasetWithTTC.

        Args:
            ttc_json_path: Path to the JSON file containing TTC data.
            base_path: Base directory for data.
            geometry_path: Directory for geometry/depth data.
            metadata_path: Path to the metadata file.
            height: Target height for images/videos.
            width: Target width for images/videos.
            height_division_factor: Factor to ensure height is divisible by.
            width_division_factor: Factor to ensure width is divisible by.
            num_frames: Number of frames to retrieve.
            time_division_factor: Factor for temporal alignment/downsampling.
            time_division_remainder: Remainder for temporal alignment.
            repeat: Number of times to repeat the dataset.
            data_file_keys: Keys identifying data files in the metadata.
            main_data_operator: Operator to apply to main data fields.
            special_operator_map: Map of special operators for specific keys.
            ttc_video_key: Metadata key used to link to TTC data (default: "video").
            ttc_field_name: Output key name for TTC data (default: "ttc").
            filter_missing_ttc: Filter samples without TTC data.
            filter_first_ttc_zero: Filter samples where first TTC is 0.
            filter_no_accident_frame: Filter samples with no accident frame (0.0).

        Raises:
            ValueError: If ttc_json_path is invalid.
        """
        if ttc_json_path is None or str(ttc_json_path).strip() == "":
            raise ValueError("ttc_json_path must be a valid path")

        self.ttc_json_path = str(ttc_json_path)
        self.ttc_video_key = str(ttc_video_key)
        self.ttc_field_name = str(ttc_field_name)
        self.filter_missing_ttc = bool(filter_missing_ttc)
        self.filter_first_ttc_zero = bool(filter_first_ttc_zero)
        self.filter_no_accident_frame = bool(filter_no_accident_frame)

        self.ttc_map: Dict[str, List[float]] = self._load_ttc_map(self.ttc_json_path)

        # Ensure we don't treat TTC as a file path to be loaded by the base class.
        if self.ttc_field_name in data_file_keys:
            data_file_keys = [k for k in data_file_keys if k != self.ttc_field_name]

        super().__init__(
            base_path=base_path,
            geometry_path=geometry_path,
            metadata_path=metadata_path,
            # -------- Tong Liu ADD for depth procession --------
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            # -------- Tong Liu ADD for depth procession --------
            repeat=repeat,
            data_file_keys=tuple(data_file_keys),
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
        )

        if not self.load_from_cache:
            self._attach_and_filter_ttc()

    def _load_ttc_map(self, path: str) -> Dict[str, List[float]]:
        """Loads the TTC mapping from a JSON file.

        Args:
            path: Path to the TTC JSON file.

        Returns:
            A dictionary mapping video IDs to lists of TTC values.

        Raises:
            ValueError: If the loaded JSON is not a dictionary.
        """
        with open(path, "r") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"TTC json must be a dict, got {type(raw)}")

        out: Dict[str, List[float]] = {}
        for k, v in raw.items():
            if v is None:
                continue
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                continue
            key = str(k)
            try:
                out[key] = [int(x) for x in v]
            except Exception:
                continue
        return out

    def _video_id_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        """Extracts the video ID from a data record.

        Args:
            record: A single data record dictionary.

        Returns:
            The video ID string (stem of the path), or None if not found.
        """
        if self.ttc_video_key not in record:
            return None
        v = record.get(self.ttc_video_key, None)
        if v is None:
            return None
        return Path(str(v)).stem

    def _attach_and_filter_ttc(self) -> None:
        """Attaches TTC data to records and filters the dataset.

        Iterates through the dataset, attaches TTC lists based on video IDs,
        and applies configured filters (missing, zero-start, no-accident).
        Updates self.data with the filtered list.
        """
        filtered: List[Dict[str, Any]] = []
        for record in self.data:
            vid_id = self._video_id_from_record(record)
            if vid_id is None:
                continue

            ttc = self.ttc_map.get(vid_id, None)
            if ttc is None:
                if self.filter_missing_ttc:
                    continue
                record[self.ttc_field_name] = None
                filtered.append(record)
                continue

            if len(ttc) == 0:
                continue

            if self.filter_first_ttc_zero:
                try:
                    if float(ttc[0]) == 0.0:
                        continue
                except Exception:
                    continue

            if self.filter_no_accident_frame:
                if 0.0 not in ttc:
                    continue

            record[self.ttc_field_name] = ttc
            filtered.append(record)

        self.data = filtered

    def _get_ttc_tensor(self, ttc_list: List[float]) -> torch.Tensor:
        """Converts TTC list to a Tensor and crops it to align with video/depth.

        Ensures the TTC tensor length matches the video and depth frames by
        applying the same temporal constraints (num_frames and VAE division factors).

        Args:
            ttc_list: List of raw TTC values.

        Returns:
            A torch.Tensor containing the aligned and cropped TTC values.
        """
        ttc_tensor = torch.tensor(ttc_list, dtype=torch.float32)

        # 1. Calculate target crop length.
        total_frames = len(ttc_tensor)
        # Prioritize configured num_frames, default to 49 if None.
        target_num_frames = self.num_frames or 49

        # Take the minimum: truncate if longer, use actual length if shorter.
        actual_num_frames = min(target_num_frames, total_frames)

        # 2. VAE constraint alignment (e.g., frame_count % 4 == 1).
        # Must strictly match LoadVideo logic.
        factor = self.time_division_factor or 4
        remainder = self.time_division_remainder or 1

        while (actual_num_frames > 1 and
               actual_num_frames % factor != remainder):
            actual_num_frames -= 1

        # 3. Crop data (sequential crop of the first N frames).
        return ttc_tensor[:actual_num_frames]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieves a data sample.

        Loads the base data (Video, Depth, Metadata) using the parent class,
        then intercepts and processes the TTC field to return an aligned Tensor.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionary containing the data sample with aligned TTC tensor.
        """
        # 1. Call parent method to get base data.
        data = super().__getitem__(index)

        # 2. Intercept and process TTC data.
        if self.ttc_field_name in data and data[self.ttc_field_name] is not None:
            # Replace raw long list with aligned tensor.
            data[self.ttc_field_name] = self._get_ttc_tensor(
                data[self.ttc_field_name]
            )

        return data