from safetensors.torch import load_file
import re

paths = [
  "/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_high_noise_ttc_embedder141/epoch-4.safetensors",
  "/baai-cwm-backup/cwm/tong.liu/outputckpt/Wan2.2-I2V-A14B_low_noise_ttc_embedder141/epoch-4.safetensors",
]
for p in paths:
    sd = load_file(p, device="cpu")
    keys = list(sd.keys())
    print("\n===", p, "===")
    print("num_keys:", len(keys))
    hit = [k for k in keys if "ttc" in k.lower() or "embed" in k.lower() or "depth" in k.lower()]
    print("sample_hits:", hit[:30])
    print("has_ttc_embedder_prefix:", any(k.startswith("ttc_embedder.") for k in keys))
    print("has_pipe_dit_ttc_prefix:", any(k.startswith("pipe.dit.ttc_embedder.") for k in keys))