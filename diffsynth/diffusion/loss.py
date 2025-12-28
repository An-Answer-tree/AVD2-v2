from .base_pipeline import BasePipeline
import torch


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
    """Calculates the combined loss for the Dual-Head WanVideo model.

    This function computes the Flow Matching loss for the video generation head
    and the Mean Squared Error (MSE) loss for the depth prediction head.
    It handles the separate outputs returned by the modified model_fn.

    Args:
        pipe: The training pipeline instance containing the model and scheduler.
        depth_loss_weight: The weighting factor for the depth loss component.
            Defaults to 1.0.
        **inputs: A dictionary containing batch data, including 'input_latents',
            'input_depth_latents', and configuration parameters.

    Returns:
        A scalar Tensor representing the weighted sum of video and depth losses.
    """
    # 1. Timestep Sampling
    # Calculate boundaries for timestep sampling based on input config.
    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
    )

    # Sample a random timestep index and convert to the correct device/dtype.
    timestep_id = torch.randint(
        min_timestep_boundary, max_timestep_boundary, (1,)
    )
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )

    # 2. Video Target Preparation (Flow Matching)
    # Generate noise and add it to the clean video latents.
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(
        inputs["input_latents"], noise, timestep
    )
    
    # Calculate the velocity target (v) for the video head.
    training_target = pipe.scheduler.training_target(
        inputs["input_latents"], noise, timestep
    )

    # 3. Model Forward Pass
    # Gather necessary sub-models.
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    
    # Perform inference. This calls your modified model_fn_wan_video.
    # Expected output: {"video": Tensor, "depth": Tensor} or Tensor
    model_outputs = pipe.model_fn(**models, **inputs, timestep=timestep)

    # 4. Output Parsing
    # Handle both dictionary output (Dual Head) and tensor output (Legacy/Single Head).
    if isinstance(model_outputs, dict):
        noise_pred = model_outputs.get("video")
        depth_pred = model_outputs.get("depth")
    else:
        noise_pred = model_outputs
        depth_pred = None

    # 5. Calculate Video Loss
    # Standard MSE between predicted velocity and target velocity.
    loss_video = torch.nn.functional.mse_loss(
        noise_pred.float(), training_target.float()
    )
    loss_video = loss_video * pipe.scheduler.training_weight(timestep)

    # 6. Calculate Depth Loss
    loss_depth = torch.tensor(0.0, device=loss_video.device, dtype=loss_video.dtype)
    
    if depth_pred is not None:
        target_depth = inputs.get("input_depth_latents")
        
        if target_depth is None:
            raise ValueError(
                "Depth prediction exists but 'input_depth_latents' is missing "
                "from inputs. Check your Dataset and Unit configuration."
            )

        # Shape Alignment:
        # If the model sliced off reference frames (I2V), depth_pred will have 
        # fewer frames than the ground truth (target_depth).
        # We must align the ground truth to match the prediction.
        if depth_pred.shape[2] != target_depth.shape[2]:
            diff = target_depth.shape[2] - depth_pred.shape[2]
            # Slice the target to remove the initial reference frames.
            target_depth = target_depth[:, :, diff:]

        # Compute MSE Loss against Clean Ground Truth (Reconstruction).
        loss_depth = torch.nn.functional.mse_loss(
            depth_pred.float(), target_depth.float()
        )

    # 7. Aggregate Total Loss (adaptive loss weight)
    total_loss = loss_video + (loss_depth * depth_loss_weight)

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
        import lpips # TODO: remove it
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
