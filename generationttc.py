# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import glob
import argparse
import secrets
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def _sorted_ckpts(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found for pattern: {pattern}")

    def _key(p: str):
        m = re.search(r"-(\d+)-of-(\d+)", p)
        return (int(m.group(1)) if m else 0, p)

    return sorted(files, key=_key)


def _read_csv_pairs(csv_path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.lstrip("\ufeff").rstrip().rstrip(";")
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("video,prompt") or lower.startswith("filename,prompt"):
                continue

            row = next(csv.reader([line]))
            if not row:
                continue

            name = row[0].strip().strip('"').strip()
            if not name or not name.endswith(".mp4"):
                idx = line.find(",")
                if idx == -1:
                    continue
                name = line[:idx].strip().strip('"')
                prompt = line[idx + 1 :].strip().strip('"')
            else:
                prompt = ",".join(row[1:]).strip().strip('"')

            if name and prompt:
                out.append((name, prompt))
    return out


def _first_frame_as_pil(video_path: str, target_size: Tuple[int, int]) -> Image.Image:
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(frame)
        return img.resize(target_size, Image.BICUBIC)
    except Exception:
        pass

    try:
        import cv2
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
        import imageio.v3 as iio
        frame = iio.imread(video_path, index=0)
        if frame is None:
            raise RuntimeError("imageio failed to read the first frame.")
        img = Image.fromarray(frame)
        return img.resize(target_size, Image.BICUBIC)
    except Exception as e:
        raise RuntimeError(f"Failed to read first frame from {video_path}: {e}")


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


def _compute_occurrence_ordinals(pairs: List[Tuple[str, str]]) -> List[int]:
    counts: Dict[str, int] = defaultdict(int)
    ordinals: List[int] = []
    for name, _ in pairs:
        ordinals.append(counts[name])
        counts[name] += 1
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


def _load_ttc_json(ttc_json_path: str) -> Dict[str, List[float]]:
    with open(ttc_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, List[float]] = {}

    def _register(video_key: str, seq: Any):
        if video_key is None:
            return
        name = os.path.basename(str(video_key))
        stem = os.path.splitext(name)[0]

        ttc_seq = _normalize_ttc_sequence(seq)
        if not ttc_seq:
            return

        mapping[video_key] = ttc_seq
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
                _register(vid, seq)
    else:
        raise ValueError(f"Unsupported TTC JSON top-level type: {type(data)}")

    return mapping


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
            if not isinstance(it, dict):
                continue
            if value_key not in it:
                continue
            try:
                out.append(float(it[value_key]))
            except Exception:
                continue
        return out

    out: List[float] = []
    for x in seq:
        try:
            out.append(float(x))
        except Exception:
            continue
    return out


def _make_ttc_tensor(ttc_seq: List[float], num_frames: int, ttc_scale: float) -> torch.Tensor:
    if len(ttc_seq) >= num_frames:
        clip = ttc_seq[:num_frames]
    else:
        pad_val = ttc_seq[-1] if len(ttc_seq) > 0 else 0.0
        clip = ttc_seq + [pad_val] * (num_frames - len(ttc_seq))
    t = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
    if ttc_scale != 1.0:
        t = t * float(ttc_scale)
    return t


def _load_state_dict_any(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path, device=device)
        return sd
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _load_ttc_ckpt_into_dit(dit: torch.nn.Module, ckpt_path: str) -> None:
    sd = _load_state_dict_any(ckpt_path, device="cpu")
    keys = list(sd.keys())

    if any(k.startswith("ttc_embedder.") or k.startswith("depth_head.") for k in keys):
        dit.load_state_dict(sd, strict=False)
        return

    if hasattr(dit, "ttc_embedder"):
        dit.ttc_embedder.load_state_dict(sd, strict=False)
        return

    raise AttributeError("dit has no attribute 'ttc_embedder' but TTC checkpoint was provided.")


def build_pipeline(
    device: str,
    model_root: str,
    lora_high: str,
    lora_low: str,
    lora_alpha: float,
    ttc_ckpt_high: Optional[str],
    ttc_ckpt_low: Optional[str],
) -> WanVideoPipeline:
    assert os.path.isfile(lora_high), f"High-noise LoRA not found: {lora_high}"
    assert os.path.isfile(lora_low), f"Low-noise LoRA not found: {lora_low}"

    _ = _sorted_ckpts(os.path.join(model_root, "high_noise_model", "diffusion_pytorch_model*.safetensors"))
    _ = _sorted_ckpts(os.path.join(model_root, "low_noise_model", "diffusion_pytorch_model*.safetensors"))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )

    pipe.enable_vram_management()

    pipe.load_lora(pipe.dit, lora_high, alpha=lora_alpha)
    pipe.load_lora(pipe.dit2, lora_low, alpha=lora_alpha)

    if ttc_ckpt_high:
        assert os.path.isfile(ttc_ckpt_high), f"TTC high ckpt not found: {ttc_ckpt_high}"
        _load_ttc_ckpt_into_dit(pipe.dit, ttc_ckpt_high)

    if ttc_ckpt_low:
        assert os.path.isfile(ttc_ckpt_low), f"TTC low ckpt not found: {ttc_ckpt_low}"
        _load_ttc_ckpt_into_dit(pipe.dit2, ttc_ckpt_low)

    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=os.getenv("CSV", "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/video_ttestttc.csv"))
    parser.add_argument("--ttc_json", type=str, default=os.getenv("TTC_JSON", "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/outputs/ttcexample.json"))
    parser.add_argument("--init_src_root", type=str, default=os.getenv("INIT_SRC_ROOT", "/baai-cwm-vepfs/cwm/cheng.li/liutong/MM-AU/full_demos/"))
    parser.add_argument("--out", type=str, default=os.getenv("OUT", "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/outputs/ttc_generation"))

    parser.add_argument("--model_root", type=str, default=os.getenv("MODEL_ROOT", "models/Wan-AI/Wan2.2-I2V-A14B"))
    parser.add_argument("--lora_high", type=str, default=os.getenv("LORA_HIGH", "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_high_noise_lora/epoch-4.safetensors"))
    parser.add_argument("--lora_low", type=str, default=os.getenv("LORA_LOW", "/baai-cwm-vepfs/cwm/cheng.li/liutong/DiffSynth-Studio/models/train/Wan2.2-I2V-A14B_low_noise_lora/epoch-4.safetensors"))
    parser.add_argument("--ttc_ckpt_high", type=str, default=os.getenv("TTC_CKPT_HIGH", ""))
    parser.add_argument("--ttc_ckpt_low", type=str, default=os.getenv("TTC_CKPT_LOW", ""))

    parser.add_argument("--width", type=int, default=int(os.getenv("WIDTH", "960")))
    parser.add_argument("--height", type=int, default=int(os.getenv("HEIGHT", "512")))
    parser.add_argument("--num_frames", type=int, default=int(os.getenv("NUM_FRAMES", "49")))
    parser.add_argument("--num_steps", type=int, default=int(os.getenv("NUM_STEPS", "50")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("FPS", "15")))
    parser.add_argument("--quality", type=int, default=int(os.getenv("QUALITY", "5")))
    parser.add_argument("--switch_dit_boundary", type=float, default=float(os.getenv("SWITCH_DIT_BOUNDARY", "0.90")))
    parser.add_argument("--tiled", action="store_true", default=(os.getenv("TILED", "1") == "1"))
    parser.add_argument("--lora_alpha", type=float, default=float(os.getenv("LORA_ALPHA", "1.0")))
    parser.add_argument("--ttc_scale", type=float, default=float(os.getenv("TTC_SCALE", "1.0")))
    parser.add_argument("--skip_if_ttc0", action="store_true", default=True)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)

    args = parser.parse_args()

    base_seed = secrets.randbits(31)

    infer_device, infer_shard, infer_world = _infer_runtime_args_from_env()
    device = args.device if args.device is not None else infer_device
    shard_index = args.shard_index if args.shard_index is not None else infer_shard
    num_shards = args.num_shards if args.num_shards is not None else infer_world
    num_shards = max(1, int(num_shards))

    os.makedirs(args.out, exist_ok=True)

    pairs_all = _read_csv_pairs(args.csv)
    if not pairs_all:
        raise RuntimeError(f"No (video, prompt) rows parsed from CSV: {args.csv}")

    ordinals_all = _compute_occurrence_ordinals(pairs_all)
    seeds_all: List[int] = [((base_seed + i * 1_000_003) & 0x7FFFFFFF) for i in range(len(pairs_all))]

    ttc_map = _load_ttc_json(args.ttc_json)

    indexed_pairs = [(i, pairs_all[i]) for i in range(len(pairs_all)) if i % num_shards == shard_index]

    print(f"Total jobs: {len(pairs_all)} | Shard {shard_index}/{num_shards} -> {len(indexed_pairs)} jobs on {device}")
    print(f"Base random seed: {base_seed}")
    print(f"CSV: {args.csv}")
    print(f"TTC_JSON: {args.ttc_json}")
    print(f"INIT_SRC_ROOT: {args.init_src_root}")
    print(f"OUT: {args.out}")

    if device.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass

    model_root_abs = args.model_root
    if not os.path.isabs(model_root_abs):
        model_root_abs = os.path.abspath(model_root_abs)

    print("Building Wan 2.2 I2V pipeline...")
    pipe = build_pipeline(
        device=device,
        model_root=model_root_abs,
        lora_high=args.lora_high,
        lora_low=args.lora_low,
        lora_alpha=args.lora_alpha,
        ttc_ckpt_high=(args.ttc_ckpt_high or None),
        ttc_ckpt_low=(args.ttc_ckpt_low or None),
    )
    print("Pipeline ready. Starting batch generation...\n")

    negative_prompt = (
        "over-saturated colors, overexposed, static, blurry details, subtitles, "
        "painterly style, artwork, still image, grayish overall, worst quality, "
        "low quality, JPEG artifacts, ugly, broken parts, malformed limbs, "
        "bad hands, bad face, fused fingers, motionless frame, cluttered background"
    )
    
    for j, (row_idx, (mp4_name, prompt)) in enumerate(indexed_pairs, 1):
        src_video = os.path.join(args.init_src_root, mp4_name)

        base_ordinal = ordinals_all[row_idx]
        out_video = _next_available_out_path(mp4_name, base_ordinal, args.out)

        if not os.path.isfile(src_video):
            print(f"[{device} | {j}/{len(indexed_pairs)}] MISSING source: {src_video}")
            continue

        row_seed = int(seeds_all[row_idx])
        torch.manual_seed(row_seed)

        ttc_seq = ttc_map.get(mp4_name) or ttc_map.get(os.path.basename(mp4_name)) or ttc_map.get(os.path.splitext(os.path.basename(mp4_name))[0])
        if ttc_seq is None:
            print(f"[{device} | {j}/{len(indexed_pairs)}] MISSING TTC: {mp4_name}")
            continue

        ttc_tensor = _make_ttc_tensor(ttc_seq, args.num_frames, args.ttc_scale)

        if args.skip_if_ttc0:
            try:
                if float(ttc_tensor[0, 0].item()) == 0.0:
                    print(f"[{device} | {j}/{len(indexed_pairs)}] SKIP TTC0: {mp4_name}")
                    continue
            except Exception:
                pass

        try:
            input_image = _first_frame_as_pil(src_video, target_size=(args.width, args.height))
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ttc=ttc_tensor,
                seed=row_seed,
                tiled=args.tiled,
                input_image=input_image,
                switch_DiT_boundary=args.switch_dit_boundary,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
            )
            save_video(video, out_video, fps=args.fps, quality=args.quality)
            rel_out = os.path.relpath(out_video, args.out)
            print(f"[{device} | {j}/{len(indexed_pairs)}] DONE: {rel_out} (seed={row_seed})")
        except Exception as e:
            print(f"[{device} | {j}/{len(indexed_pairs)}] FAIL: {mp4_name} | {e}")

    print("\nAll jobs finished for this shard.")
    print(f"Outputs: {args.out}")


if __name__ == "__main__":
    main()
