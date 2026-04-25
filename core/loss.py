"""Loss functions for the optimization loop.

All rendered images are (H, W, C) float32 tensors in [0, 1].

Two modes
---------
MSE  : Mean-squared error between rendered pixels and a target image.
       Used for both views when a target image is available.

SDS  : Score Distillation Sampling (Poole et al., DreamFusion 2022).
       Used for View 2 when no target image exists — a frozen diffusion
       model supplies the gradient signal instead.

       Setup::
           pip install diffusers transformers accelerate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass   # diffusers types are lazily imported


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _match_size(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bilinearly resize ``target`` to match ``rendered`` spatial dims if needed."""
    if rendered.shape[:2] == target.shape[:2]:
        return target
    # (H, W, C) → (1, C, H, W) → resize → (H, W, C)
    t = target.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(
        t,
        size=(rendered.shape[0], rendered.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return t.squeeze(0).permute(1, 2, 0)


# ---------------------------------------------------------------------------
# MSE loss
# ---------------------------------------------------------------------------


def mse_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared error between a rendered image and a target.

    Args:
        rendered: (H, W, C) float32 — output from the differentiable renderer.
                  May have 4 channels (RGBA); the alpha channel is ignored.
        target:   (H, W, 3) float32 — ground-truth RGB image in [0, 1].
                  Resized automatically if spatial dims differ.
        mask:     (H, W) or (H, W, 1) float32 optional weight map.
                  Useful for ignoring background pixels.

    Returns:
        Scalar loss tensor, differentiable w.r.t. ``rendered``.
    """
    r = rendered[..., :3]              # drop alpha if present → (H, W, 3)
    t = _match_size(r, target.to(r.device))

    diff = (r - t) ** 2               # (H, W, 3)

    if mask is not None:
        m = mask.to(r.device)
        if m.dim() == 2:
            m = m.unsqueeze(-1)        # (H, W, 1)  broadcast over C
        diff = diff * m
        return diff.sum() / (m.sum() * 3.0 + 1e-8)

    return diff.mean()


def silhouette_loss(
    rendered: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error between rendered alpha and a target foreground mask."""
    if rendered.shape[-1] >= 4:
        alpha = rendered[..., 3:4]
    else:
        alpha = rendered[..., :3].amax(dim=-1, keepdim=True)
    mask = _match_size(alpha, target_mask.to(alpha.device))
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    return ((alpha - mask.clamp(0.0, 1.0)) ** 2).mean()


def masked_rgb_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """RGB loss weighted to the target foreground."""
    if target_mask.dim() == 2:
        target_mask = target_mask.unsqueeze(-1)
    mask = _match_size(rendered[..., :1], target_mask.to(rendered.device)).clamp(0.0, 1.0)
    return mse_loss(rendered, target, mask)


# ---------------------------------------------------------------------------
# SDS loss
# ---------------------------------------------------------------------------


def sds_loss(
    rendered: torch.Tensor,
    prompt: str,
    pipe,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    t_range: tuple[float, float] = (0.02, 0.98),
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """Score Distillation Sampling loss (DreamFusion, Poole et al. 2022).

    Computes a gradient signal from a frozen diffusion model and attaches
    it to ``rendered`` so that backprop updates the scene parameters directly.
    The model never generates a full denoised image — only the score direction
    is needed, making this efficient for use inside an optimization loop.

    Args:
        rendered:        (H, W, 3) float32 RGB image from the renderer.
                         Resized to 512×512 before passing to the VAE.
        prompt:          Text prompt describing the desired appearance.
        pipe:            A loaded HuggingFace ``StableDiffusionPipeline``
                         (eval mode, UNet and VAE detached from gradients).
        negative_prompt: Optional negative conditioning text.
        guidance_scale:  Classifier-free guidance weight ω.
                         Higher → stronger push toward the prompt.
        t_range:         (t_min, t_max) as fractions of the diffusion
                         schedule to sample timesteps from.
                         Lower t_max → finer detail; higher → coarser shape.
        cfg_rescale:     Rescale coefficient from Lin et al. (2023) to reduce
                         over-saturation.  0.0 disables it.

    Returns:
        Scalar loss tensor, differentiable w.r.t. ``rendered``.

    Raises:
        ValueError: If ``pipe`` is None.

    Setup::

        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")   # or "mps" for Apple Silicon
        pipe.unet.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
    """
    if pipe is None:
        raise ValueError(
            "SDS loss requires a diffusion pipeline.  "
            "See the docstring for setup instructions."
        )

    device = rendered.device
    unet_dtype = next(pipe.unet.parameters()).dtype

    # ------------------------------------------------------------------ #
    # 1. Encode the rendered image into the VAE latent space              #
    # ------------------------------------------------------------------ #

    # (H, W, 3) → (1, 3, 512, 512)  in [-1, 1]
    img = rendered[..., :3].permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False)
    img = img.to(device=device, dtype=unet_dtype)
    img_sd = img * 2.0 - 1.0          # [0,1] → [-1, 1]

    with torch.no_grad():
        latents = pipe.vae.encode(img_sd).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor   # (1, 4, 64, 64)

    # Re-attach to autograd so the gradient reaches `rendered`
    latents = latents.requires_grad_(True)

    # ------------------------------------------------------------------ #
    # 2. Sample a random diffusion timestep                               #
    # ------------------------------------------------------------------ #

    n_train_steps = pipe.scheduler.config.num_train_timesteps
    t_lo = max(1, int(t_range[0] * n_train_steps))
    t_hi = min(n_train_steps - 1, int(t_range[1] * n_train_steps))
    t = torch.randint(t_lo, t_hi, (1,), device=device)

    # ------------------------------------------------------------------ #
    # 3. Add noise to the latents (forward diffusion)                     #
    # ------------------------------------------------------------------ #

    noise = torch.randn_like(latents)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    # ------------------------------------------------------------------ #
    # 4. Encode text prompt (with negative conditioning)                  #
    # ------------------------------------------------------------------ #

    with torch.no_grad():
        prompts = [prompt, negative_prompt]
        text_ids = pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        text_embeds = pipe.text_encoder(text_ids)[0]   # (2, 77, 768)

    # ------------------------------------------------------------------ #
    # 5. Predict noise with classifier-free guidance                      #
    # ------------------------------------------------------------------ #

    latents_in  = torch.cat([noisy_latents] * 2)       # (2, 4, 64, 64)
    t_in        = torch.cat([t] * 2)

    with torch.no_grad():
        noise_pred = pipe.unet(
            latents_in,
            t_in,
            encoder_hidden_states=text_embeds,
        ).sample   # (2, 4, 64, 64)

    noise_cond, noise_uncond = noise_pred.chunk(2)

    # Classifier-free guidance
    noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    # Optional CFG rescaling (Lin et al. 2023) to avoid over-saturation
    if cfg_rescale > 0.0:
        std_pos    = noise_cond.std(dim=list(range(1, noise_cond.ndim)), keepdim=True)
        std_guided = noise_guided.std(dim=list(range(1, noise_guided.ndim)), keepdim=True)
        factor     = std_pos / (std_guided + 1e-8)
        factor     = cfg_rescale * factor + (1.0 - cfg_rescale)
        noise_guided = noise_guided * factor

    # ------------------------------------------------------------------ #
    # 6. Compute the SDS gradient weight w(t)                             #
    # ------------------------------------------------------------------ #

    # w(t) = 1 - ᾱ_t  (signal-to-noise weight from the original paper)
    alpha_bar = pipe.scheduler.alphas_cumprod.to(device=device, dtype=unet_dtype)
    w = (1.0 - alpha_bar[t]).sqrt().view(-1, 1, 1, 1)  # (1, 1, 1, 1)

    # ------------------------------------------------------------------ #
    # 7. SDS pseudo-loss: L = w(t) · (ε_θ − ε) · stop_grad(latents)    #
    # The gradient of this w.r.t. latents is exactly the SDS update.     #
    # ------------------------------------------------------------------ #

    grad   = w * (noise_guided - noise)       # (1, 4, 64, 64)
    loss   = (latents * grad.detach()).sum()  # scalar, grad flows through latents

    return loss
