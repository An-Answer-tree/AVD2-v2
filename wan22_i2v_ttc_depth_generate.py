#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import glob
import argparse
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core.loader import ModelConfig


DEFAULT_NEGATIVE_PROMPT = (
    "over-saturated colors, overexposed, static, blurry details, subtitles, "
    "painterly style, artwork, still image, grayish overall, worst quality, "
    "low quality, JPEG artifacts, ugly, broken parts, malformed limbs, "
    "bad hands, bad face, fused fingers, motionless frame, cluttered background"
)


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


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
        return device, int(shard_index), int(num_shards)
    return "cuda:0", 0, 1


def _sorted_ckpts(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found for pattern: {pattern}")

    def _key(p: str):
        m = re.search(r"-(\d+)-of-(\d+)", p)
        return (int(m.group(1)) if m else 0, p)

    return sorted(files, key=_key)


def _pil_bicubic_resample() -> int:
    if hasattr(Image, "Resampling"):
        return int(Image.Resampling.BICUBIC)
    return int(Image.BICUBIC)


def _first_frame_as_pil(video_path: str, target_size: Tuple[int, int]) -> Image.Image:
    resample = _pil_bicubic_resample()

    try:
        from decord import VideoReader, cpu  # type: ignore

        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(frame).convert("RGB")
        return img.resize(target_size, resample)
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
        img = Image.fromarray(frame).convert("RGB")
        return img.resize(target_size, resample)
    except Exception:
        pass

    try:
        import imageio.v3 as iio  # type: ignore

        frame = iio.imread(video_path, index=0)
        if frame is None:
            raise RuntimeError("imageio failed to read the first frame.")
        img = Image.fromarray(frame).convert("RGB")
        return img.resize(target_size, resample)
    except Exception as e:
        raise RuntimeError(f"Failed to read first frame from {video_path}: {e}")


def _read_csv_rows(csv_path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.lstrip("\ufeff").rstrip().rstrip(";")
            if not line:
                continue
            lower = line.lower().strip()
            if lower.startswith("video,prompt") or lower.startswith("filename,prompt"):
                continue
            row = next(csv.reader([line]))
            if not row:
                continue
            name = row[0].strip().strip('"').strip()
            if not name:
                continue
            if len(row) >= 2:
                prompt = ",".join(row[1:]).strip().strip('"').strip()
            else:
                idx = line.find(",")
                if idx < 0:
                    continue
                prompt = line[idx + 1 :].strip().strip('"').strip()
            if name and prompt:
                out.append((name, prompt))
    return out


def _compute_occurrence_ordinals(names: List[str]) -> List[int]:
    counts: Dict[str, int] = defaultdict(int)
    ordinals: List[int] = []
    for n in names:
        ordinals.append(counts[n])
        counts[n] += 1
    return ordinals


def _compose_out_path(base_name: str, ordinal: int, out_root: str, suffix: str = "") -> str:
    rel_dir = os.path.dirname(base_name)
    base = os.path.basename(base_name)
    stem, ext = os.path.splitext(base)
    sfx = "" if ordinal == 0 else f"_{ordinal}"
    rel_out = os.path.join(rel_dir, f"{stem}{sfx}{suffix}{ext}") if rel_dir else f"{stem}{sfx}{suffix}{ext}"
    out_path = os.path.join(out_root, rel_out)
    parent = os.path.dirname(out_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return out_path


def _next_available_out_path(base_name: str, start_ordinal: int, out_root: str, suffix: str = "") -> str:
    n = max(0, int(start_ordinal))
    while True:
        candidate = _compose_out_path(base_name, n, out_root, suffix=suffix)
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _normalize_ttc_sequence(seq: Any) -> List[float]:
    if seq is None:
        return []
    if isinstance(seq, (int, float)):
        return [float(seq)]
    if not isinstance(seq, list):
        return []
    if len(seq) == 0:
        return []

    first = seq[0]
    if isinstance(first, dict):
        value_key = None
        for k in ("ttc", "TTC", "value", "val"):
            if k in first:
                value_key = k
                break
        if value_key is None:
            return []

        frame_key = None
        for k in ("frame", "frame_id", "idx", "index"):
            if k in first:
                frame_key = k
                break

        items = seq
        if frame_key is not None:
            try:
                items = sorted(items, key=lambda x: int(x.get(frame_key, 0)))
            except Exception:
                pass

        out: List[float] = []
        for it in items:
            if not isinstance(it, dict) or value_key not in it:
                continue
            try:
                out.append(float(it[value_key]))
            except Exception:
                continue
        return out

    out2: List[float] = []
    for x in seq:
        try:
            out2.append(float(x))
        except Exception:
            continue
    return out2


def _load_ttc_json(ttc_json_path: str) -> Dict[str, List[float]]:
    with open(ttc_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[float]] = {}

    def _register(video_key: str, seq: Any) -> None:
        name = os.path.basename(str(video_key))
        stem = os.path.splitext(name)[0]
        ttc_seq = _normalize_ttc_sequence(seq)
        if not ttc_seq:
            return
        mapping[str(video_key)] = ttc_seq
        mapping[name] = ttc_seq
        mapping[stem] = ttc_seq

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                _register(k, v)
            elif isinstance(v, dict):
                for kk in ("ttc", "TTC", "ttc_frames", "values", "ttc_list"):
                    if kk in v:
                        _register(k, v[kk])
                        break
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = item.get("video") or item.get("video_name") or item.get("filename") or item.get("file") or item.get("path")
            seq = item.get("ttc") or item.get("TTC") or item.get("ttc_frames") or item.get("values")
            if vid is not None and seq is not None:
                _register(str(vid), seq)
    else:
        raise ValueError(f"Unsupported TTC JSON top-level type: {type(data)}")

    return mapping


def _sample_ttc(ttc_seq: List[float], num_frames: int, mode: str) -> List[float]:
    if not ttc_seq:
        return [0.0] * num_frames

    vals = [float(x) for x in ttc_seq]
    L = len(vals)

    if L == num_frames:
        return vals

    if L > num_frames:
        if mode == "first":
            return vals[:num_frames]
        if mode == "center":
            start = max(0, (L - num_frames) // 2)
            return vals[start : start + num_frames]
        if mode == "uniform":
            if num_frames == 1:
                return [vals[0]]
            out: List[float] = []
            for i in range(num_frames):
                idx = int(round(i * (L - 1) / (num_frames - 1)))
                out.append(vals[idx])
            return out
        raise ValueError(f"Unknown ttc_sampling: {mode}")

    pad_val = vals[-1]
    return vals + [pad_val] * (num_frames - L)


def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # type: ignore

        return load_file(path, device="cpu")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")
    return obj


def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _load_ckpt_into_module(module: torch.nn.Module, ckpt_path: str, tag: str) -> None:
    sd_raw = _load_state_dict_any(ckpt_path)
    sd = _strip_prefixes(
        sd_raw,
        prefixes=[
            "module.",
            "pipe.",
            "pipe.dit.",
            "pipe.dit2.",
            "dit.",
            "dit2.",
        ],
    )
    incompatible = module.load_state_dict(sd, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    if os.getenv("RANK", "0") == "0":
        print(f"[ckpt:{tag}] {ckpt_path} | missing={missing} unexpected={unexpected}")


def _to_uint8_video(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        x = x.detach().cpu()
    x = x.float()
    if x.ndim == 5:
        x = x[0]
        x = x.permute(1, 2, 3, 0).contiguous()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    return x.to(torch.uint8).numpy()


def _depth_to_uint8(depth_1cthw: torch.Tensor) -> np.ndarray:
    if depth_1cthw.is_cuda:
        depth_1cthw = depth_1cthw.detach().cpu()
    d = depth_1cthw.float()
    if d.ndim == 5:
        d = d[0]
    if d.shape[0] >= 1:
        d = d[0]
    d = torch.clamp(d, -1.0, 1.0)
    d = (d + 1.0) * 0.5
    d = (d * 255.0).to(torch.uint8)
    return d.numpy()


def _set_seed(seed: int) -> None:
    seed = int(seed) & 0x7FFFFFFF
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_required_file(path: str, alt: Optional[str] = None) -> str:
    if os.path.isfile(path):
        return path
    if alt is not None and os.path.isfile(alt):
        return alt
    raise FileNotFoundError(path)


def _gather_wan_model_configs(model_root: str, model_id: str) -> List[ModelConfig]:
    high_paths = _sorted_ckpts(os.path.join(model_root, "high_noise_model", "diffusion_pytorch_model*.safetensors"))
    low_paths = _sorted_ckpts(os.path.join(model_root, "low_noise_model", "diffusion_pytorch_model*.safetensors"))

    t5_path = _find_required_file(
        os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
        alt=os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.safetensors"),
    )

    vae_path = _find_required_file(
        os.path.join(model_root, "Wan2.1_VAE.pth"),
        alt=os.path.join(model_root, "Wan2.1_VAE.safetensors"),
    )

    return [
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            path=high_paths,
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            path=low_paths,
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern=os.path.basename(t5_path),
            path=t5_path,
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern=os.path.basename(vae_path),
            path=vae_path,
            offload_device="cpu",
        ),
    ]


def build_pipeline(
    device: str,
    model_id: str,
    model_root: str,
    tokenizer_path: Optional[str],
    lora_high: str,
    lora_low: str,
    ttc_ckpt_high: str,
    ttc_ckpt_low: str,
    depth_ckpt_high: str,
    depth_ckpt_low: str,
    lora_alpha: float,
) -> WanVideoPipeline:
    model_root = str(model_root)
    model_configs = _gather_wan_model_configs(model_root, model_id=str(model_id))

    tokenizer_config = None
    if tokenizer_path:
        tokenizer_config = ModelConfig(path=str(tokenizer_path))
    else:
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        audio_processor_config=None,
    )

    if hasattr(pipe, "enable_vram_management") and callable(getattr(pipe, "enable_vram_management")):
        try:
            pipe.enable_vram_management()
        except Exception:
            pass

    pipe.load_lora(pipe.dit, lora_high, alpha=float(lora_alpha))
    if getattr(pipe, "dit2", None) is not None:
        pipe.load_lora(pipe.dit2, lora_low, alpha=float(lora_alpha))

    _load_ckpt_into_module(pipe.dit, ttc_ckpt_high, tag="ttc_high")
    _load_ckpt_into_module(pipe.dit, depth_ckpt_high, tag="depth_high")

    if getattr(pipe, "dit2", None) is not None:
        _load_ckpt_into_module(pipe.dit2, ttc_ckpt_low, tag="ttc_low")
        _load_ckpt_into_module(pipe.dit2, depth_ckpt_low, tag="depth_low")

    return pipe


@torch.no_grad()
def generate_one(
    pipe: WanVideoPipeline,
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    ttc_tensor: torch.Tensor,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    num_steps: int,
    cfg_scale: float,
    cfg_merge: bool,
    switch_boundary: float,
    tiled: bool,
    tile_size: Tuple[int, int],
    tile_stride: Tuple[int, int],
    sigma_shift: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipe.scheduler.set_timesteps(num_steps, denoising_strength=1.0, shift=sigma_shift)

    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared: Dict[str, Any] = {
        "input_image": input_image,
        "seed": int(seed),
        "rand_device": "cpu",
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "cfg_scale": float(cfg_scale),
        "cfg_merge": bool(cfg_merge),
        "switch_DiT_boundary": float(switch_boundary),
        "num_inference_steps": int(num_steps),
        "sigma_shift": float(sigma_shift),
        "tiled": bool(tiled),
        "tile_size": tuple(tile_size),
        "tile_stride": tuple(tile_stride),
        "ttc": ttc_tensor,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "vace_scale": 1.0,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    if "latents" not in inputs_shared:
        raise RuntimeError("Pipeline units did not produce 'latents'.")

    _set_seed(seed)
    depth_latents = torch.zeros_like(inputs_shared["latents"], device=inputs_shared["latents"].device, dtype=inputs_shared["latents"].dtype)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {
        name: getattr(pipe, name)
        for name in pipe.in_iteration_models
        if getattr(pipe, name, None) is not None
    }

    t_max = float(pipe.scheduler.timesteps.max().item())
    boundary_ts = float(switch_boundary)
    if 0.0 <= boundary_ts <= 1.0:
        boundary_ts = boundary_ts * t_max

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=pipe.torch_dtype)
        if str(pipe.device).startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )

    for progress_id, ts in enumerate(pipe.scheduler.timesteps):
        ts_val = float(ts.item())

        if getattr(pipe, "dit2", None) is not None and ts_val < boundary_ts and models.get("dit") is not pipe.dit2:
            if hasattr(pipe, "in_iteration_models_2"):
                pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            if getattr(pipe, "vace2", None) is not None:
                models["vace"] = pipe.vace2

        timestep = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        with autocast_ctx:
            out_pos = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                depth_latents=depth_latents,
                timestep=timestep,
                progress_id=progress_id,
            )

            if cfg_scale != 1.0:
                if cfg_merge:
                    if not isinstance(out_pos, dict):
                        raise TypeError("cfg_merge=True requires dict output for dual-head generation")
                    vp, vn = out_pos["video"].chunk(2, dim=0)
                    dp, dn = out_pos["depth"].chunk(2, dim=0)
                else:
                    out_neg = pipe.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        depth_latents=depth_latents,
                        timestep=timestep,
                        progress_id=progress_id,
                    )
                    if isinstance(out_pos, dict) and isinstance(out_neg, dict):
                        vp, dp = out_pos["video"], out_pos.get("depth")
                        vn, dn = out_neg["video"], out_neg.get("depth")
                    else:
                        raise TypeError("Dual-head generation expects dict outputs from model_fn")

                v = vn + float(cfg_scale) * (vp - vn)
                d = dn + float(cfg_scale) * (dp - dn)
            else:
                if not isinstance(out_pos, dict):
                    raise TypeError("Dual-head generation expects dict output from model_fn")
                v = out_pos["video"]
                d = out_pos["depth"]

        inputs_shared["latents"] = pipe.scheduler.step(v, ts, inputs_shared["latents"])
        depth_latents = pipe.scheduler.step(d, ts, depth_latents)

        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    for unit in getattr(pipe, "post_units", []):
        inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(["vae"])

    video_latents = inputs_shared["latents"]
    depth_latents_out = depth_latents

    video = pipe.vae.decode(video_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    depth_vid = pipe.vae.decode(depth_latents_out, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

    video_u8 = _to_uint8_video(video)
    depth_u8 = _depth_to_uint8(depth_vid)

    depth_f = depth_vid.detach().float().cpu()
    if depth_f.ndim == 5:
        depth_f = depth_f[0]
    depth_f = depth_f[0].permute(0, 1, 2).contiguous().numpy()

    pipe.load_models_to_device([])

    return video_u8, depth_u8, depth_f


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--ttc_json", type=str, required=True)
    ap.add_argument("--init_src_root", type=str, required=True)
    ap.add_argument("--out_video", type=str, required=True)
    ap.add_argument("--out_depth", type=str, required=True)

    ap.add_argument("--model_root", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None)

    ap.add_argument("--lora_high", type=str, required=True)
    ap.add_argument("--lora_low", type=str, required=True)
    ap.add_argument("--ttc_ckpt_high", type=str, required=True)
    ap.add_argument("--ttc_ckpt_low", type=str, required=True)
    ap.add_argument("--depth_ckpt_high", type=str, required=True)
    ap.add_argument("--depth_ckpt_low", type=str, required=True)
    ap.add_argument("--lora_alpha", type=float, default=1.0)

    ap.add_argument("--width", type=int, default=368)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=int, default=5)

    ap.add_argument("--cfg_scale", type=float, default=1.0)
    ap.add_argument("--cfg_merge", type=_str2bool, default=False)

    ap.add_argument("--switch_dit_boundary", type=float, default=0.90)
    ap.add_argument("--sigma_shift", type=float, default=5.0)

    ap.add_argument("--tiled", type=_str2bool, default=True)
    ap.add_argument("--tile_size_h", type=int, default=30)
    ap.add_argument("--tile_size_w", type=int, default=52)
    ap.add_argument("--tile_stride_h", type=int, default=15)
    ap.add_argument("--tile_stride_w", type=int, default=26)

    ap.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)

    ap.add_argument("--ttc_sampling", type=str, default="first", choices=["first", "center", "uniform"])
    ap.add_argument("--ttc_scale", type=float, default=1.0)

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--shard_index", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)

    ap.add_argument("--base_seed", type=int, default=-1)

    args = ap.parse_args()

    if args.base_seed is None or int(args.base_seed) < 0:
        base_seed = secrets.randbits(31)
    else:
        base_seed = int(args.base_seed) & 0x7FFFFFFF

    infer_device, infer_shard, infer_world = _infer_runtime_args_from_env()
    device = args.device if args.device is not None else infer_device
    shard_index = int(args.shard_index if args.shard_index is not None else infer_shard)
    num_shards = int(args.num_shards if args.num_shards is not None else infer_world)
    num_shards = max(1, num_shards)

    if device.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass

    os.makedirs(args.out_video, exist_ok=True)
    os.makedirs(args.out_depth, exist_ok=True)

    rows_all = _read_csv_rows(args.csv)
    if not rows_all:
        raise RuntimeError(f"No valid rows parsed from CSV: {args.csv}")

    names_all = [r[0] for r in rows_all]
    ordinals_all = _compute_occurrence_ordinals(names_all)
    seeds_all = [((base_seed + i * 1_000_003) & 0x7FFFFFFF) for i in range(len(rows_all))]

    ttc_map = _load_ttc_json(args.ttc_json)

    indexed_rows = [(i, rows_all[i]) for i in range(len(rows_all)) if (i % num_shards) == shard_index]

    if os.getenv("RANK", "0") == "0":
        print(f"Device: {device} | Shard: {shard_index}/{num_shards} | Jobs: {len(indexed_rows)}/{len(rows_all)}")
        print(f"Base seed: {base_seed}")

    pipe = build_pipeline(
        device=device,
        model_id=args.model_id,
        model_root=args.model_root,
        tokenizer_path=args.tokenizer_path,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        ttc_ckpt_high=args.ttc_ckpt_high,
        ttc_ckpt_low=args.ttc_ckpt_low,
        depth_ckpt_high=args.depth_ckpt_high,
        depth_ckpt_low=args.depth_ckpt_low,
        lora_alpha=float(args.lora_alpha),
    )

    tile_size = (int(args.tile_size_h), int(args.tile_size_w))
    tile_stride = (int(args.tile_stride_h), int(args.tile_stride_w))

    for j, (row_idx, (mp4_name, prompt)) in enumerate(indexed_rows, 1):
        src_video = mp4_name if os.path.isabs(mp4_name) else os.path.join(args.init_src_root, mp4_name)
        if not os.path.isfile(src_video):
            if os.getenv("RANK", "0") == "0":
                print(f"[{device} | {j}/{len(indexed_rows)}] MISSING source: {src_video}")
            continue

        ttc_seq = (
            ttc_map.get(mp4_name)
            or ttc_map.get(os.path.basename(mp4_name))
            or ttc_map.get(os.path.splitext(os.path.basename(mp4_name))[0])
        )
        if ttc_seq is None:
            if os.getenv("RANK", "0") == "0":
                print(f"[{device} | {j}/{len(indexed_rows)}] MISSING TTC: {mp4_name}")
            continue

        ttc_list = _sample_ttc(ttc_seq, int(args.num_frames), mode=str(args.ttc_sampling))
        ttc_tensor = torch.tensor(ttc_list, dtype=torch.float32).unsqueeze(0)
        if float(args.ttc_scale) != 1.0:
            ttc_tensor = ttc_tensor * float(args.ttc_scale)
        if device.startswith("cuda"):
            ttc_tensor = ttc_tensor.to(device=device)

        ordinal = ordinals_all[row_idx]
        base_name_for_out = mp4_name if not os.path.isabs(mp4_name) else os.path.basename(mp4_name)
        out_video_path = _next_available_out_path(base_name_for_out, ordinal, args.out_video, suffix="")
        out_depth_path = _next_available_out_path(base_name_for_out, ordinal, args.out_depth, suffix="_depth")
        out_depth_npy = os.path.splitext(out_depth_path)[0] + ".npy"

        row_seed = int(seeds_all[row_idx])

        try:
            input_image = _first_frame_as_pil(src_video, target_size=(int(args.width), int(args.height)))

            video_u8, depth_u8, depth_f = generate_one(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                input_image=input_image,
                ttc_tensor=ttc_tensor,
                seed=row_seed,
                height=int(args.height),
                width=int(args.width),
                num_frames=int(args.num_frames),
                num_steps=int(args.num_steps),
                cfg_scale=float(args.cfg_scale),
                cfg_merge=bool(args.cfg_merge),
                switch_boundary=float(args.switch_dit_boundary),
                tiled=bool(args.tiled),
                tile_size=tile_size,
                tile_stride=tile_stride,
                sigma_shift=float(args.sigma_shift),
            )

            save_video(video_u8, out_video_path, fps=int(args.fps), quality=int(args.quality))

            depth_rgb = np.repeat(depth_u8[..., None], 3, axis=3)
            save_video(depth_rgb, out_depth_path, fps=int(args.fps), quality=int(args.quality))

            np.save(out_depth_npy, depth_f.astype(np.float32, copy=False))

            if os.getenv("RANK", "0") == "0":
                rel_v = os.path.relpath(out_video_path, args.out_video)
                rel_d = os.path.relpath(out_depth_path, args.out_depth)
                print(f"[{device} | {j}/{len(indexed_rows)}] DONE: {rel_v} | {rel_d} | seed={row_seed}")

        except Exception as e:
            if os.getenv("RANK", "0") == "0":
                print(f"[{device} | {j}/{len(indexed_rows)}] FAIL: {mp4_name} | {e}")

    if os.getenv("RANK", "0") == "0":
        print(f"Outputs video: {args.out_video}")
        print(f"Outputs depth: {args.out_depth}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
