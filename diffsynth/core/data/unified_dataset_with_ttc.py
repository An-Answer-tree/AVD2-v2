# Added by Cheng Li
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .unified_dataset import UnifiedDataset


class UnifiedDatasetWithTTC(UnifiedDataset):
    def __init__(
        self,
        ttc_json_path: str,
        base_path: Optional[str] = None,
        geometry_path: Optional[str] = None,  # 2025_12/22_17:25 [Tong Liu ADD] Geometry Directory
        metadata_path: Optional[str] = None,
        ######## Tong Liu ADD for depth procession ########
        height: Optional[int] = None,
        width: Optional[int] = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        num_frames: Optional[int] = None,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        ######## Tong Liu ADD for depth procession ########
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
        if ttc_json_path is None or str(ttc_json_path).strip() == "":
            raise ValueError("ttc_json_path must be a valid path")

        self.ttc_json_path = str(ttc_json_path)
        self.ttc_video_key = str(ttc_video_key)
        self.ttc_field_name = str(ttc_field_name)
        self.filter_missing_ttc = bool(filter_missing_ttc)
        self.filter_first_ttc_zero = bool(filter_first_ttc_zero)
        self.filter_no_accident_frame = bool(filter_no_accident_frame)

        self.ttc_map: Dict[str, List[float]] = self._load_ttc_map(self.ttc_json_path)

        if self.ttc_field_name in data_file_keys:
            data_file_keys = [k for k in data_file_keys if k != self.ttc_field_name]

        super().__init__(
            base_path=base_path,
            geometry_path=geometry_path,
            metadata_path=metadata_path,
            ######## Tong Liu ADD for depth procession ########
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            ######## Tong Liu ADD for depth procession ########
            repeat=repeat,
            data_file_keys=tuple(data_file_keys),
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
        )

        if not self.load_from_cache:
            self._attach_and_filter_ttc()

    def _load_ttc_map(self, path: str) -> Dict[str, List[float]]:
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
                out[key] = [float(x) for x in v]
            except Exception:
                continue
        return out

    def _video_id_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        if self.ttc_video_key not in record:
            return None
        v = record.get(self.ttc_video_key, None)
        if v is None:
            return None
        return Path(str(v)).stem

    def _attach_and_filter_ttc(self) -> None:
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
