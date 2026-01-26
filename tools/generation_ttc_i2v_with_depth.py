#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch I2V generation with Wan 2.2 (DiffSynth) with TTC conditioning (optional) and
optional latent depth inference.

This script:
- Generates I2V videos using Wan 2.2 (high/low noise DiT experts + T5 + VAE)
- Loads optional LoRA adapters for high/low experts (stage1)
- Loads optional TTC conditioning weights into high/low experts (stage2)
- Optionally loads a latent depth head checkpoint (stage3) and predicts depth
  from the generated video latents, saving:
    - <output>.mp4
    - <output>_depth.mp4
    - <output>_depth.npz   (keys: point_map, mask)

CSV format:
  video_name.mp4,prompt
  video_name.mp4,prompt,ttc

Prompts may contain commas; quoting is supported.

Multi-GPU:
  torchrun --standalone --nproc_per_node=N generation_ttc_i2v_with_depth.py ...

Rows are sharded by (row_index % world_size == rank).
"""

from __future__ import annotations

import os
import re
import csv
import json
import glob
import argparse
import secrets
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_latent_depth_head import WanLatentDepthHead


DEFAULT_NEGATIVE_PROMPT = (
    "over-saturated colors, overexposed, static, blurry details, subtitles, "
    "painterly style, artwork, still image, grayish overall, worst quality, "
    "low quality, JPEG artifacts, ugly, broken parts, malformed limbs, "
    "bad hands, bad face, fused fingers, motionless frame, cluttered background"
)


def _sorted_ckpts(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found for pattern: {pattern}")

    def _key(p: str) -> Tuple[int, str]:
        m = re.search(r"-(\d+)-of-(\d+)", p)
        return (int(m.group(1)) if m else 0, p)

    return sorted(files, key=_key)


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _infer_runtime_args_from_env() -> Tuple[str, int, int]:
    local_rank = _env_int("LOCAL_RANK", None)
    rank = _env_int("RANK", None)
    world_size = _env_int("WORLD_SIZE", None)
    if local_rank is not None and world_size is not None:
        device = f"cuda:{local_rank}"
        shard_index = rank if rank is not None else local_rank
        num_shards = world_size
        return device, shard_index, num_shards
    return "cuda:0", 0, 1


def _normalize_video_key(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = s.replace("\\\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        xs = str(x).strip().strip('"').strip()
        if xs == "":
            return None
        return float(xs)
    except Exception:
        return None


def _read_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.lstrip("\ufeff").strip()
            line = line.rstrip(";").strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("video,prompt") or lower.startswith("filename,prompt") or lower.startswith("video_name,prompt"):
                continue
            parsed = next(csv.reader([line]))
            if not parsed:
                continue
            name = parsed[0].strip().strip('"').strip()
            if not name:
                continue

            ttc_val: Optional[float] = None
            prompt: str = ""

            if len(parsed) >= 3:
                maybe_ttc = _safe_float(parsed[-1])
                if maybe_ttc is not None:
                    ttc_val = float(maybe_ttc)
                    prompt = ",".join(parsed[1:-1]).strip().strip('"')
                else:
                    prompt = ",".join(parsed[1:]).strip().strip('"')
            else:
                prompt = ",".join(parsed[1:]).strip().strip('"')

            if not prompt:
                continue

            rows.append({"video": _normalize_video_key(name), "prompt": prompt, "ttc": ttc_val})
    return rows


def _try_extract_ttc_record(rec: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    video = (
        rec.get("video")
        or rec.get("video_name")
        or rec.get("filename")
        or rec.get("file")
        or rec.get("path")
        or rec.get("video_path")
        or rec.get("name")
    )
    ttc = rec.get("ttc") or rec.get("TTC") or rec.get("time_to_collision") or rec.get("timeToCollision")
    if video is not None and ttc is not None:
        return str(video), ttc

    for nested_key in ["metrics", "result", "outputs", "pred", "prediction"]:
        obj = rec.get(nested_key, None)
        if isinstance(obj, dict):
            ttc2 = obj.get("ttc") or obj.get("TTC") or obj.get("time_to_collision") or obj.get("timeToCollision")
            if video is not None and ttc2 is not None:
                return str(video), ttc2
    return None


def load_ttc_map(ttc_json_path: str) -> Dict[str, Union[float, List[float]]]:
    with open(ttc_json_path, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)

    mapping: Dict[str, Union[float, List[float]]] = {}

    def add(video_key: str, value: Union[float, List[float]]) -> None:
        k = _normalize_video_key(video_key)
        b = os.path.basename(k)
        if k not in mapping:
            mapping[k] = value
        if b and b not in mapping:
            mapping[b] = value

    def normalize_value(v: Any) -> Optional[Union[float, List[float]]]:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, list):
            out_seq: List[float] = []
            for x in v:
                fx = _safe_float(x)
                if fx is None:
                    return None
                out_seq.append(float(fx))
            return out_seq
        # try string to float
        fv = _safe_float(v)
        if fv is not None:
            return float(fv)
        return None

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                pair = _try_extract_ttc_record(item)
                if pair is not None:
                    vv = normalize_value(pair[1])
                    if vv is not None:
                        add(pair[0], vv)
        return mapping

    if isinstance(obj, dict):
        # {name: scalar|seq}
        for k, v in obj.items():
            vv = normalize_value(v)
            if vv is not None:
                add(str(k), vv)

        if mapping:
            return mapping

        for key in ["results", "data", "items", "records", "outputs"]:
            if key in obj:
                inner = obj[key]
                if isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict):
                            pair = _try_extract_ttc_record(item)
                            if pair is not None:
                                vv = normalize_value(pair[1])
                                if vv is not None:
                                    add(pair[0], vv)
                elif isinstance(inner, dict):
                    for k, v in inner.items():
                        vv = normalize_value(v)
                        if vv is not None:
                            add(str(k), vv)
                if mapping:
                    return mapping

    return mapping


def _compute_occurrence_ordinals(names: List[str]) -> List[int]:
    counts: Dict[str, int] = defaultdict(int)
    ordinals: List[int] = []
    for n in names:
        ordinals.append(counts[n])
        counts[n] += 1
    return ordinals


def _compose_out_path(base_name: str, ordinal: int, out_root: str) -> str:
    rel_dir = os.path.dirname(base_name)
    base = os.path.basename(base_name)
    stem, ext = os.path.splitext(base)
    suffix = "" if ordinal == 0 else f"_{ordinal}"
    rel_out = os.path.join(rel_dir, f"{stem}{suffix}{ext}") if rel_dir else f"{stem}{suffix}{ext}"
    out_path = os.path.join(out_root, rel_out)
    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return out_path


def _next_available_out_path(base_name: str, start_ordinal: int, out_root: str) -> str:
    n = max(0, int(start_ordinal))
    while True:
        candidate = _compose_out_path(base_name, n, out_root)
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _first_frame_as_pil(video_path: str, target_size: Tuple[int, int]) -> Image.Image:
    try:
        from decord import VideoReader, cpu  # type: ignore

        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(frame)
        return img.resize(target_size, Image.BICUBIC)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("OpenCV failed to read the first frame.")
        frame = frame[:, :, ::-1]
        img = Image.fromarray(frame)
        return img.resize(target_size, Image.BICUBIC)
    except Exception:
        pass

    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(video_path, index=0)
        if frame is None:
            raise RuntimeError("imageio failed to read the first frame.")
        img = Image.fromarray(frame)
        return img.resize(target_size, Image.BICUBIC)
    except Exception as e:
        raise RuntimeError(f"Failed to read first frame from {video_path}: {e}")


def _select_ckpt_file(path_or_dir: str) -> str:
    p = Path(path_or_dir)
    if p.is_file():
        return str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path_or_dir}")
    epoch_candidates = list(p.glob("epoch-*.safetensors"))
    if epoch_candidates:
        def _epoch_num(x: Path) -> int:
            m = re.search(r"epoch-(\d+)\.safetensors$", x.name)
            return int(m.group(1)) if m else -1

        epoch_candidates.sort(key=_epoch_num)
        return str(epoch_candidates[-1])

    any_candidates = list(p.glob("*.safetensors")) + list(p.glob("*.pth")) + list(p.glob("*.pt"))
    if any_candidates:
        any_candidates.sort(key=lambda x: x.stat().st_mtime)
        return str(any_candidates[-1])

    raise FileNotFoundError(f"No checkpoint files found in directory: {path_or_dir}")


def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                break
        out[nk] = v
    return out


def _choose_best_state_dict_for_target(
    raw_sd: Dict[str, torch.Tensor],
    target: torch.nn.Module,
) -> Tuple[str, Dict[str, torch.Tensor], int]:
    target_keys = set(target.state_dict().keys())
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [
        ("as_is", raw_sd),
        ("strip_module", _strip_prefixes(raw_sd, ["module."])),
        ("strip_pipe_dit", _strip_prefixes(raw_sd, ["pipe.dit.", "pipe.dit2."])),
        ("strip_pipe", _strip_prefixes(raw_sd, ["pipe."])),
        ("strip_dit", _strip_prefixes(raw_sd, ["dit.", "dit2."])),
        ("strip_pipe_and_module", _strip_prefixes(_strip_prefixes(raw_sd, ["module."]), ["pipe.dit.", "pipe.dit2.", "pipe."])),
    ]
    best_name = "as_is"
    best_sd = raw_sd
    best_match = -1
    for name, sd in candidates:
        match = len(set(sd.keys()) & target_keys)
        if match > best_match:
            best_match = match
            best_name = name
            best_sd = sd
    return best_name, best_sd, best_match


def _load_trainable_ckpt_into_model(target_model: torch.nn.Module, ckpt_path_or_dir: str, label: str) -> None:
    ckpt_file = _select_ckpt_file(ckpt_path_or_dir)
    sd = load_state_dict(ckpt_file)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(f"Invalid state_dict loaded from: {ckpt_file}")

    best_name, best_sd, best_match = _choose_best_state_dict_for_target(sd, target_model)
    if best_match <= 0:
        sample_keys = list(sd.keys())[:20]
        raise RuntimeError(
            f"No matching keys found when loading {ckpt_file} into {label}. "
            f"Sample keys: {sample_keys}"
        )

    missing, unexpected = target_model.load_state_dict(best_sd, strict=False)
    print(
        f"[CKPT] {label}: loaded from {ckpt_file} | mapping={best_name} | "
        f"keys_in_ckpt={len(best_sd)} | matched={best_match} | missing={len(missing)} | unexpected={len(unexpected)}"
    )
    if unexpected:
        print(f"[CKPT] {label}: unexpected keys (first 20): {unexpected[:20]}")


def _validate_pipe_accepts_ttc(pipe: WanVideoPipeline) -> None:
    try:
        sig = inspect.signature(pipe.__call__)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if ("ttc" not in sig.parameters) and (not has_var_kw):
            raise RuntimeError(
                "WanVideoPipeline.__call__ does not accept 'ttc'. "
                "Make sure you are running a codebase that supports TTC conditioning."
            )
    except (ValueError, TypeError):
        return


def build_pipeline(
    device: str,
    model_id: str,
    model_root: Optional[str],
    lora_high: Optional[str],
    lora_low: Optional[str],
    lora_alpha: float,
    ttc_ckpt_high: Optional[str],
    ttc_ckpt_low: Optional[str],
) -> WanVideoPipeline:
    if model_root:
        model_root_p = Path(model_root)
        if not model_root_p.exists():
            raise FileNotFoundError(f"model_root not found: {model_root}")
        _ = _sorted_ckpts(str(model_root_p / "high_noise_model" / "diffusion_pytorch_model*.safetensors"))
        _ = _sorted_ckpts(str(model_root_p / "low_noise_model" / "diffusion_pytorch_model*.safetensors"))
        if not (model_root_p / "models_t5_umt5-xxl-enc-bf16.pth").exists():
            raise FileNotFoundError(f"Missing T5 file under model_root: {model_root_p/'models_t5_umt5-xxl-enc-bf16.pth'}")
        if not (model_root_p / "Wan2.1_VAE.pth").exists():
            raise FileNotFoundError(f"Missing VAE file under model_root: {model_root_p/'Wan2.1_VAE.pth'}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id=model_id, origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management(
            num_persistent_param_in_dit=0,
        )

    if lora_high:
        if not Path(lora_high).is_file():
            raise FileNotFoundError(f"High LoRA not found: {lora_high}")
        pipe.load_lora(pipe.dit, lora_high, alpha=lora_alpha)

    if lora_low:
        if not Path(lora_low).is_file():
            raise FileNotFoundError(f"Low LoRA not found: {lora_low}")
        pipe.load_lora(pipe.dit2, lora_low, alpha=lora_alpha)

    if ttc_ckpt_high:
        _load_trainable_ckpt_into_model(pipe.dit, ttc_ckpt_high, label="high_noise_dit")
    if ttc_ckpt_low:
        _load_trainable_ckpt_into_model(pipe.dit2, ttc_ckpt_low, label="low_noise_dit")

    _validate_pipe_accepts_ttc(pipe)
    return pipe


def _load_depth_head(depth_ckpt_path_or_dir: str, in_channels: int, out_channels: int, device: str) -> WanLatentDepthHead:
    ckpt_file = _select_ckpt_file(depth_ckpt_path_or_dir)
    sd = load_state_dict(ckpt_file)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(f"Invalid state_dict loaded from: {ckpt_file}")

    model = WanLatentDepthHead(in_channels=in_channels, out_channels=out_channels)
    best_name, best_sd, best_match = _choose_best_state_dict_for_target(sd, model)
    if best_match <= 0:
        sample_keys = list(sd.keys())[:20]
        raise RuntimeError(
            f"No matching keys found when loading {ckpt_file} into latent_depth_head. "
            f"Sample keys: {sample_keys}"
        )

    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    print(
        f"[CKPT] latent_depth_head: loaded from {ckpt_file} | mapping={best_name} | "
        f"matched={best_match} | missing={len(missing)} | unexpected={len(unexpected)}"
    )
    model = model.to(device)
    model.eval()
    return model


def _colorize_depth(depth01: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """depth01: (H, W) float in [0,1] -> (H,W,3) uint8"""
    depth01 = np.clip(depth01, 0.0, 1.0)
    try:
        import matplotlib
        import matplotlib.cm

        cm = matplotlib.cm.get_cmap(cmap_name)
        rgba = cm(depth01)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        return rgb
    except Exception:
        gray = (depth01 * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def _depth_to_point_map_npz(depth01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map (T,H,W) in [0,1] to (point_map, mask) like the provided .npz.

    point_map: (T,H,W,3) float16
    mask:     (T,H,W) bool

    This is a lightweight, self-contained representation that preserves the required
    npz structure. X/Y are a simple normalized projection scaled by depth.
    """
    if depth01.ndim != 3:
        raise ValueError(f"depth01 must be (T,H,W), got {depth01.shape}")
    t, h, w = depth01.shape

    u = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    point_map = np.zeros((t, h, w, 3), dtype=np.float32)
    mask = np.isfinite(depth01) & (depth01 > 0.0)

    for i in range(t):
        z = depth01[i]
        point_map[i, ..., 0] = uu * z
        point_map[i, ..., 1] = vv * z
        point_map[i, ..., 2] = z

    return point_map.astype(np.float16), mask.astype(bool)


def _make_ttc_tensor(
    ttc_val: Union[float, List[float]],
    num_frames: int,
    device: str,
    dtype: torch.dtype,
    seq_mode: str = "clip",
) -> torch.Tensor:
    if isinstance(ttc_val, list):
        seq = [float(x) for x in ttc_val]
        if len(seq) == 0:
            raise ValueError("Empty TTC sequence.")
        if len(seq) < num_frames:
            seq = seq + [seq[-1]] * (num_frames - len(seq))
        if len(seq) > num_frames:
            if seq_mode == "clip":
                seq = seq[:num_frames]
            elif seq_mode == "tail":
                seq = seq[-num_frames:]
            elif seq_mode == "center":
                start = (len(seq) - num_frames) // 2
                seq = seq[start : start + num_frames]
            elif seq_mode == "random":
                start = secrets.randbelow(len(seq) - num_frames + 1)
                seq = seq[start : start + num_frames]
            else:
                raise ValueError(f"Unknown seq_mode: {seq_mode}")
        t = torch.tensor(seq, device=device, dtype=dtype).unsqueeze(0)
        return t

    ttc_scalar = float(ttc_val)
    return torch.full((1, num_frames), ttc_scalar, device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--ttc_json_path", type=str, default=None)
    parser.add_argument("--strict_ttc", action="store_true", help="If set, missing TTC will raise an error.")
    parser.add_argument("--ttc_scale", type=float, default=1.0)
    parser.add_argument("--ttc_seq_mode", type=str, default="clip", choices=["clip", "tail", "center", "random"])

    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-I2V-A14B")
    parser.add_argument("--model_root", type=str, default=None)

    parser.add_argument("--lora_high", type=str, default=None)
    parser.add_argument("--lora_low", type=str, default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)

    parser.add_argument("--ttc_ckpt_high", type=str, default=None)
    parser.add_argument("--ttc_ckpt_low", type=str, default=None)

    parser.add_argument("--depth_ckpt", type=str, default=None)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--depth_latent_scale", type=float, default=1.0)
    parser.add_argument("--depth_cmap", type=str, default="turbo")

    parser.add_argument("--width", type=int, default=368)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=145)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.9)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)

    parser.add_argument("--base_seed", type=int, default=-1)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    infer_device, infer_shard, infer_world = _infer_runtime_args_from_env()
    device = args.device if args.device is not None else infer_device
    shard_index = args.shard_index if args.shard_index is not None else infer_shard
    num_shards = args.num_shards if args.num_shards is not None else infer_world

    num_shards = max(1, int(num_shards))
    shard_index = int(shard_index)

    if device.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rows_all = _read_csv_rows(args.csv_path)
    if not rows_all:
        raise RuntimeError(f"No rows parsed from CSV: {args.csv_path}")

    ttc_map: Dict[str, Union[float, List[float]]] = {}
    if args.ttc_json_path:
        if not Path(args.ttc_json_path).is_file():
            raise FileNotFoundError(f"TTC JSON not found: {args.ttc_json_path}")
        ttc_map = load_ttc_map(args.ttc_json_path)
        print(f"[TTC] Loaded TTC mapping entries: {len(ttc_map)} from {args.ttc_json_path}")

    names_all = [r["video"] for r in rows_all]
    ordinals_all = _compute_occurrence_ordinals(names_all)

    if args.base_seed >= 0:
        base_seed = int(args.base_seed) & 0x7FFFFFFF
    else:
        base_seed = secrets.randbits(31)

    seeds_all: List[int] = [((base_seed + i * 1_000_003) & 0x7FFFFFFF) for i in range(len(rows_all))]

    indexed_rows = [(i, rows_all[i]) for i in range(len(rows_all)) if i % num_shards == shard_index]
    print(f"Total jobs: {len(rows_all)} | Shard {shard_index}/{num_shards} -> {len(indexed_rows)} jobs on {device}")
    print(f"Base seed: {base_seed}")

    print("Building Wan 2.2 I2V pipeline...")
    pipe = build_pipeline(
        device=device,
        model_id=args.model_id,
        model_root=args.model_root,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        lora_alpha=float(args.lora_alpha),
        ttc_ckpt_high=args.ttc_ckpt_high,
        ttc_ckpt_low=args.ttc_ckpt_low,
    )

    depth_head: Optional[WanLatentDepthHead] = None
    if args.save_depth:
        if not args.depth_ckpt:
            raise RuntimeError("--save_depth was set but --depth_ckpt is not provided.")
        z_dim = int(getattr(pipe.vae.model, "z_dim", 16))
        depth_head = _load_depth_head(args.depth_ckpt, in_channels=z_dim, out_channels=z_dim, device=device)
        print(f"[DEPTH] latent depth head ready (channels={z_dim})")

    print("Pipeline ready. Starting batch generation...\n")

    w = int(args.width)
    h = int(args.height)
    ttc_dtype = torch.float32

    for j, (row_idx, row) in enumerate(indexed_rows, 1):
        mp4_name = row["video"]
        prompt = row["prompt"]

        ttc_val: Optional[Union[float, List[float]]] = None
        if row.get("ttc") is not None:
            ttc_val = float(row["ttc"])
        else:
            key1 = _normalize_video_key(mp4_name)
            key2 = os.path.basename(key1)
            if key1 in ttc_map:
                ttc_val = ttc_map[key1]
            elif key2 in ttc_map:
                ttc_val = ttc_map[key2]

        ttc_tensor: Optional[torch.Tensor] = None
        if ttc_val is not None:
            ttc_tensor = _make_ttc_tensor(ttc_val, num_frames=int(args.num_frames), device=device, dtype=ttc_dtype, seq_mode=args.ttc_seq_mode)
            if args.ttc_scale != 1.0:
                ttc_tensor = ttc_tensor * float(args.ttc_scale)
        else:
            if args.strict_ttc:
                raise RuntimeError(f"Missing TTC for {mp4_name}")

        src_video = os.path.join(args.video_root, mp4_name)
        base_ordinal = ordinals_all[row_idx]
        out_video = _next_available_out_path(mp4_name, base_ordinal, out_dir)

        if not os.path.isfile(src_video):
            print(f"[{device} | {j}/{len(indexed_rows)}] MISSING source: {src_video}")
            continue

        row_seed = int(seeds_all[row_idx])
        try:
            torch.manual_seed(row_seed)

            input_image = _first_frame_as_pil(src_video, target_size=(w, h))

            save_depth = bool(args.save_depth)
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=row_seed,
                tiled=bool(args.tiled),
                input_image=input_image,
                ttc=ttc_tensor,
                switch_DiT_boundary=float(args.switch_dit_boundary),
                height=h,
                width=w,
                num_frames=int(args.num_frames),
                num_inference_steps=int(args.num_steps),
                return_latents=save_depth,
            )

            if save_depth:
                assert isinstance(result, dict) and "video" in result and "latents" in result
                video = result["video"]
                latents_final = result["latents"]
            else:
                video = result
                latents_final = None

            save_video(video, out_video, fps=int(args.fps), quality=int(args.quality))

            rel_out = os.path.relpath(out_video, out_dir)
            ttc_str = "None" if ttc_val is None else "seq" if isinstance(ttc_val, list) else f"{float(ttc_val):.6f}"
            print(f"[{device} | {j}/{len(indexed_rows)}] DONE: {rel_out} (seed={row_seed}, ttc={ttc_str})")

            if save_depth:
                if depth_head is None:
                    raise RuntimeError("Internal error: save_depth requested but depth_head is None")
                if latents_final is None:
                    raise RuntimeError("Internal error: latents_final is None")

                lat = latents_final.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    depth_latents = depth_head(lat)
                    if args.depth_latent_scale != 1.0:
                        depth_latents = depth_latents * float(args.depth_latent_scale)

                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    depth_rgb = pipe.vae.decode(
                        depth_latents.to(dtype=torch.bfloat16),
                        device=device,
                        tiled=bool(args.tiled),
                        tile_size=getattr(pipe, "tile_size", 256),
                    )
                pipe.load_models_to_device([])

                depth_gray = depth_rgb.mean(dim=1)
                depth01 = ((depth_gray + 1.0) / 2.0).clamp(0.0, 1.0)
                depth_np = depth01[0].detach().cpu().numpy().astype(np.float32)

                # Improve visualization contrast for depth video only
                vmin = float(np.percentile(depth_np, 2))
                vmax = float(np.percentile(depth_np, 98))
                if vmax > vmin:
                    depth_vis = (depth_np - vmin) / (vmax - vmin)
                    depth_vis = np.clip(depth_vis, 0.0, 1.0)
                else:
                    depth_vis = depth_np

                point_map, mask = _depth_to_point_map_npz(depth_np)

                depth_npz_path = os.path.splitext(out_video)[0] + "_depth.npz"
                np.savez_compressed(depth_npz_path, point_map=point_map, mask=mask)

                depth_video_path = os.path.splitext(out_video)[0] + "_depth.mp4"
                frames_depth: List[np.ndarray] = []
                for t in range(depth_np.shape[0]):
                    frames_depth.append(_colorize_depth(depth_vis[t], cmap_name=args.depth_cmap))
                save_video(frames_depth, depth_video_path, fps=int(args.fps), quality=int(args.quality))

        except TypeError as e:
            raise RuntimeError(f"Pipeline call failed. Error: {e}") from e
        except Exception as e:
            print(f"[{device} | {j}/{len(indexed_rows)}] FAIL: {mp4_name} | {e}")

    print("\nAll jobs finished for this shard.")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
