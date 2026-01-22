from .base_pipeline import BasePipeline
import torch
import os

_LOSS_PRINT_STEP = 0


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTDualHeadLoss(
    pipe: BasePipeline,
    depth_loss_weight: float = 1.0,
    **inputs
) -> torch.Tensor:
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_outputs = pipe.model_fn(**models, **inputs, timestep=timestep)

    if isinstance(model_outputs, dict):
        noise_pred = model_outputs.get("video")
        depth_pred = model_outputs.get("depth")
    else:
        noise_pred = model_outputs
        depth_pred = None

    loss_video = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss_video = loss_video * pipe.scheduler.training_weight(timestep)

    loss_depth = torch.tensor(0.0, device=loss_video.device, dtype=loss_video.dtype)

    if depth_pred is not None:
        target_depth = inputs.get("input_depth_latents")
        if target_depth is None:
            raise ValueError(
                "Depth prediction exists but 'input_depth_latents' is missing from inputs."
            )

        if depth_pred.shape[2] != target_depth.shape[2]:
            diff = target_depth.shape[2] - depth_pred.shape[2]
            if diff > 0:
                target_depth = target_depth[:, :, diff:]
            elif diff < 0:
                depth_pred = depth_pred[:, :, (-diff):]

        loss_depth = torch.nn.functional.mse_loss(depth_pred.float(), target_depth.float())

    total_loss = loss_video + (loss_depth * depth_loss_weight)
    return total_loss


def FlowMatchSFTDualHeadLossWithDecayFactor(
    pipe: BasePipeline,
    depth_loss_weight: float = 1.0,
    **inputs
) -> torch.Tensor:
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)

    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_outputs = pipe.model_fn(**models, **inputs, timestep=timestep)

    if isinstance(model_outputs, dict):
        noise_pred = model_outputs.get("video")
        depth_pred = model_outputs.get("depth")
    else:
        noise_pred = model_outputs
        depth_pred = None

    loss_video = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss_video = loss_video * pipe.scheduler.training_weight(timestep)

    loss_depth = torch.tensor(0.0, device=loss_video.device, dtype=loss_video.dtype)

    if depth_pred is not None:
        target_depth = inputs.get("input_depth_latents")
        if target_depth is None:
            raise ValueError(
                "Depth prediction exists but 'input_depth_latents' is missing from inputs."
            )

        if depth_pred.shape[2] != target_depth.shape[2]:
            diff = target_depth.shape[2] - depth_pred.shape[2]
            if diff > 0:
                target_depth = target_depth[:, :, diff:]
            elif diff < 0:
                depth_pred = depth_pred[:, :, (-diff):]

        loss_depth = torch.nn.functional.mse_loss(depth_pred.float(), target_depth.float())

    t_max = pipe.scheduler.timesteps.max()
    decay_factor = torch.clamp((1 - timestep.float() / t_max) ** 2, min=0.0)

    total_loss = loss_video + (loss_depth * depth_loss_weight * decay_factor)

    global _LOSS_PRINT_STEP
    _LOSS_PRINT_STEP += 1
    log_every = int(inputs.get("log_every_steps", 100) or 100)

    if log_every > 0 and (_LOSS_PRINT_STEP % log_every == 0) and os.getenv("RANK", "0") == "0":
        print(
            f"[Loss Info] step={_LOSS_PRINT_STEP} T={timestep.item():.0f} | "
            f"VideoLoss: {loss_video.item():.5f} | "
            f"DepthLoss: {loss_depth.item():.5f} | "
            f"DecayFactor: {decay_factor.item():.4f} | "
            f"TotalLoss: {total_loss.item():.5f}",
            flush=True,
        )

    return total_loss


def FlowMatchSFTDualHeadJointLoss(pipe, **inputs):
    input_latents = inputs["input_latents"]
    input_depth_latents = inputs.get("input_depth_latents", None)
    if input_depth_latents is None:
        return FlowMatchSFTLoss(pipe, **inputs)

    depth_loss_weight = float(inputs.get("depth_loss_weight", 1.0))
    depth_latent_scale = float(inputs.get("depth_latent_scale", 1.0))

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    video_noise = torch.randn_like(input_latents)
    inputs["latents"] = pipe.scheduler.add_noise(input_latents, video_noise, timestep)

    depth_clean = input_depth_latents
    if depth_latent_scale != 1.0:
        depth_clean = depth_clean * depth_latent_scale

    depth_noise = torch.randn_like(depth_clean)
    inputs["depth_latents"] = pipe.scheduler.add_noise(depth_clean, depth_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_output = pipe.model_fn(**models, **inputs, timestep=timestep)

    if not isinstance(model_output, dict) or "video" not in model_output:
        raise TypeError(
            "Dual-head joint loss expects pipe.model_fn to return a dict with keys 'video' and 'depth'."
        )

    video_pred = model_output["video"]
    depth_pred = model_output.get("depth", None)

    video_target = pipe.scheduler.training_target(input_latents, video_noise, timestep)
    depth_target = pipe.scheduler.training_target(depth_clean, depth_noise, timestep)

    if video_pred.shape[2] != video_target.shape[2]:
        diff = video_target.shape[2] - video_pred.shape[2]
        if diff > 0:
            video_target = video_target[:, :, diff:]
        elif diff < 0:
            video_pred = video_pred[:, :, (-diff):]

    loss_video = torch.nn.functional.mse_loss(video_pred.float(), video_target.float())
    w = pipe.scheduler.training_weight(timestep)
    loss_video = loss_video * w

    if depth_pred is None:
        return loss_video

    if depth_pred.shape[2] != depth_target.shape[2]:
        diff = depth_target.shape[2] - depth_pred.shape[2]
        if diff > 0:
            depth_target = depth_target[:, :, diff:]
        elif diff < 0:
            depth_pred = depth_pred[:, :, (-diff):]

    loss_depth = torch.nn.functional.mse_loss(depth_pred.float(), depth_target.float())
    loss_depth = loss_depth * w

    return loss_video + depth_loss_weight * loss_depth


def FlowMatchSFTDualHeadJointLossWithDecayFactor(pipe, **inputs):
    input_latents = inputs["input_latents"]
    input_depth_latents = inputs.get("input_depth_latents", None)
    if input_depth_latents is None:
        return FlowMatchSFTLoss(pipe, **inputs)

    depth_loss_weight = float(inputs.get("depth_loss_weight", 1.0))
    depth_latent_scale = float(inputs.get("depth_latent_scale", 1.0))

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    video_noise = torch.randn_like(input_latents)
    inputs["latents"] = pipe.scheduler.add_noise(input_latents, video_noise, timestep)

    depth_clean = input_depth_latents
    if depth_latent_scale != 1.0:
        depth_clean = depth_clean * depth_latent_scale

    depth_noise = torch.randn_like(depth_clean)
    inputs["depth_latents"] = pipe.scheduler.add_noise(depth_clean, depth_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_output = pipe.model_fn(**models, **inputs, timestep=timestep)

    if not isinstance(model_output, dict) or "video" not in model_output:
        raise TypeError(
            "Dual-head joint loss expects pipe.model_fn to return a dict with keys 'video' and 'depth'."
        )

    video_pred = model_output["video"]
    depth_pred = model_output.get("depth", None)

    video_target = pipe.scheduler.training_target(input_latents, video_noise, timestep)
    depth_target = pipe.scheduler.training_target(depth_clean, depth_noise, timestep)

    if video_pred.shape[2] != video_target.shape[2]:
        diff = video_target.shape[2] - video_pred.shape[2]
        if diff > 0:
            video_target = video_target[:, :, diff:]
        elif diff < 0:
            video_pred = video_pred[:, :, (-diff):]

    w = pipe.scheduler.training_weight(timestep)

    loss_video = torch.nn.functional.mse_loss(video_pred.float(), video_target.float(), reduction="none")
    loss_video = loss_video.mean(dim=[1, 2, 3, 4])
    loss_video = loss_video * w

    loss_depth = None
    if depth_pred is not None:
        if depth_pred.shape[2] != depth_target.shape[2]:
            diff = depth_target.shape[2] - depth_pred.shape[2]
            if diff > 0:
                depth_target = depth_target[:, :, diff:]
            elif diff < 0:
                depth_pred = depth_pred[:, :, (-diff):]

        loss_depth = torch.nn.functional.mse_loss(depth_pred.float(), depth_target.float(), reduction="none")
        loss_depth = loss_depth.mean(dim=[1, 2, 3, 4])
        loss_depth = loss_depth * w

    if loss_depth is None:
        loss = loss_video
    else:
        loss = loss_video + depth_loss_weight * loss_depth

    ttc = inputs.get("ttc", None)
    if ttc is None:
        loss_decay_factor = torch.ones(loss.shape[0], device=loss.device, dtype=loss.dtype)
    else:
        if not torch.is_tensor(ttc):
            ttc = torch.as_tensor(ttc, device=loss.device, dtype=loss.dtype)
        else:
            ttc = ttc.to(device=loss.device, dtype=loss.dtype)

        if ttc.ndim == 0:
            ttc = ttc.view(1, 1)
        elif ttc.ndim == 1:
            ttc = ttc.unsqueeze(1)

        loss_decay_factor = (ttc.abs().mean(dim=1) + 1).to(device=loss.device, dtype=loss.dtype)
        if loss_decay_factor.numel() == 1 and loss.shape[0] > 1:
            loss_decay_factor = loss_decay_factor.expand(loss.shape[0])

    total_loss = (loss * loss_decay_factor).mean() / loss_decay_factor.mean()

    global _LOSS_PRINT_STEP
    _LOSS_PRINT_STEP += 1
    log_every = int(inputs.get("log_every_steps", 100) or 100)

    if log_every > 0 and (_LOSS_PRINT_STEP % log_every == 0) and os.getenv("RANK", "0") == "0":
        lv = loss_video.mean().item()
        ld = loss_depth.mean().item() if loss_depth is not None else 0.0
        df = loss_decay_factor.mean().item()
        print(
            f"[Loss Info] step={_LOSS_PRINT_STEP} T={timestep.item():.0f} | "
            f"VideoLoss: {lv:.5f} | "
            f"DepthLoss: {ld:.5f} | "
            f"DecayFactorMean: {df:.4f} | "
            f"TotalLoss: {total_loss.item():.5f}",
            flush=True,
        )

    return total_loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, device):
        import lpips
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)
            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory

    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]

            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss

    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
