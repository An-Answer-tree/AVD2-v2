import os
import argparse
import warnings
import torch
import accelerate

from diffsynth.core.data.unified_dataset_with_ttc import UnifiedDatasetWithTTC
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
        return [v for v in items if v]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _resolve_attr_path(root, path):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            raise AttributeError(
                f"Invalid trainable_models entry '{path}': '{type(obj).__name__}' has no attribute '{part}'"
            )
        obj = getattr(obj, part)
    return obj


def _set_requires_grad(obj, requires_grad):
    if obj is None:
        return
    if isinstance(obj, torch.nn.Parameter):
        obj.requires_grad = requires_grad
        return
    if torch.is_tensor(obj):
        try:
            obj.requires_grad_(requires_grad)
        except Exception:
            obj.requires_grad = requires_grad
        return
    if isinstance(obj, torch.nn.Module):
        for p in obj.parameters():
            p.requires_grad = requires_grad
        return
    raise TypeError(f"Unsupported object type for requires_grad control: {type(obj)}")


def _freeze_lora_parameters(module):
    if module is None or not isinstance(module, torch.nn.Module):
        return
    for name, param in module.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = False


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        freeze_preset_lora=False,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()

        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.2-S2V-14B",
                origin_file_pattern="wav2vec2-large-xlsr-53-english/",
            )
            if audio_processor_path is None
            else ModelConfig(audio_processor_path)
        )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self._enforce_trainability(
            trainable_models=trainable_models,
            preset_lora_path=preset_lora_path,
            preset_lora_model=preset_lora_model,
            freeze_preset_lora=freeze_preset_lora,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "dual_head_sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTDualHeadLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _freeze_all_parameters_in_pipe(self):
        if hasattr(self.pipe, "parameters"):
            try:
                for p in self.pipe.parameters():
                    p.requires_grad = False
                return
            except Exception:
                pass

        for attr in (
            "text_encoder",
            "image_encoder",
            "vae",
            "motion_controller",
            "vace",
            "vace2",
            "vap",
            "animate_adapter",
            "audio_encoder",
            "dit",
            "dit2",
        ):
            m = getattr(self.pipe, attr, None)
            if isinstance(m, torch.nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

    def _expand_trainable_paths(self, paths):
        expanded = []
        seen = set()
        for p in paths:
            if p and p not in seen:
                expanded.append(p)
                seen.add(p)

            if getattr(self.pipe, "dit2", None) is not None:
                if p == "dit":
                    alt = "dit2"
                    if alt not in seen:
                        expanded.append(alt)
                        seen.add(alt)
                elif p.startswith("dit."):
                    alt = "dit2." + p[len("dit.") :]
                    if alt not in seen:
                        expanded.append(alt)
                        seen.add(alt)

            if getattr(self.pipe, "vace2", None) is not None:
                if p == "vace":
                    alt = "vace2"
                    if alt not in seen:
                        expanded.append(alt)
                        seen.add(alt)
                elif p.startswith("vace."):
                    alt = "vace2." + p[len("vace.") :]
                    if alt not in seen:
                        expanded.append(alt)
                        seen.add(alt)

        return expanded

    def _get_preset_lora_target_modules(self, preset_lora_model):
        if preset_lora_model is None:
            names = ("dit", "dit2", "vace", "vace2", "vap", "motion_controller", "animate_adapter")
        else:
            key = str(preset_lora_model).lower()
            if key == "dit":
                names = ("dit", "dit2")
            elif key == "vace":
                names = ("vace", "vace2")
            elif key == "vap":
                names = ("vap",)
            elif key == "motion_controller":
                names = ("motion_controller",)
            elif key == "animate_adapter":
                names = ("animate_adapter",)
            else:
                names = (preset_lora_model,)

        modules = []
        for n in names:
            m = getattr(self.pipe, n, None)
            if m is not None:
                modules.append(m)
        return modules

    def _enforce_trainability(self, trainable_models, preset_lora_path, preset_lora_model, freeze_preset_lora):
        trainable_paths = _parse_csv_list(trainable_models)
        if not trainable_paths:
            return

        trainable_paths = self._expand_trainable_paths(trainable_paths)

        self._freeze_all_parameters_in_pipe()

        for path in trainable_paths:
            obj = _resolve_attr_path(self.pipe, path)
            _set_requires_grad(obj, True)

        if freeze_preset_lora and preset_lora_path is not None:
            for m in self._get_preset_lora_target_modules(preset_lora_model):
                _freeze_lora_parameters(m)

        trainable_numel = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable_numel += p.numel()
        if trainable_numel == 0:
            raise RuntimeError(
                "No trainable parameters found after enforcing trainable_models/freeze_preset_lora. "
                "Please check --trainable_models and whether the target modules exist."
            )

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "ground_truth_depth": data["depth"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


class DualHeadWanTrainingModule(WanTrainingModule):
    def __init__(self, *args, depth_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_loss_weight = depth_loss_weight


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.",
    )
    parser.add_argument("--ttc_json_path", type=str, default=None, help="Path to TTC json file.")
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--freeze_preset_lora",
        default=False,
        action="store_true",
        help="Freeze parameters of preset LoRA adapters loaded via --preset_lora_path.",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Whether to initialize models on CPU.",
    )
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    dataset = UnifiedDatasetWithTTC(
        ttc_json_path=args.ttc_json_path,
        base_path=args.dataset_base_path,
        geometry_path=args.dataset_geometry_path,
        metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDatasetWithTTC.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        },
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        freeze_preset_lora=args.freeze_preset_lora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "dual_head_sft": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
