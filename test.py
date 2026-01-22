#!/usr/bin/env python3
import inspect
import diffsynth.pipelines.wan_video as wv

print("wan_video.py path:", getattr(wv, "__file__", "unknown"))
print("model_fn_wan_video:", wv.model_fn_wan_video)
print("signature:", inspect.signature(wv.model_fn_wan_video))

sig = inspect.signature(wv.model_fn_wan_video)
has_depth_latents = ("depth_latents" in sig.parameters)
print("has depth_latents param:", has_depth_latents)

# Quick check of expected return type is not possible without running a forward,
# but we can at least confirm whether model_fn was modified to return dict
src = inspect.getsource(wv.model_fn_wan_video)
print("returns dict keywords:", ("{\"video\"" in src) or ("'video'" in src and "'depth'" in src))