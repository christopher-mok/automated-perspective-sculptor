"""Optimization loop for the anamorphic sculpture.

SceneOptimizer wraps the render → loss → backward → step cycle.  It is a
plain Python object with no Qt dependency so it can be used from scripts,
notebooks, or the UI worker thread equally.

Typical use (from a script)
---------------------------
    optimizer = SceneOptimizer(patches, renderer, cam1, cam2, t1, t2, lr=1e-3)
    for step, metrics in optimizer.run(n_steps=500):
        print(step, metrics["loss"])

Typical use (from the UI worker)
---------------------------------
    See ui/worker.py — the worker wraps ``run()`` in a QThread and emits
    Qt signals for each step.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from core.loss import masked_rgb_loss, sds_loss, silhouette_loss
from core.renderer import DiffRenderer

if TYPE_CHECKING:
    from core.patch import Patch
    from scene.camera import Camera
    from scene.scene import Mesh


DEFAULT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.05, 0.05, 0.05),
    (0.95, 0.95, 0.95),
)


def parse_palette(text: str | Sequence[str] | Sequence[Sequence[float]] | None) -> torch.Tensor:
    """Parse user-selected colours into an (K, 3) float tensor in [0, 1].

    Accepted forms:
      - "#111111, #f4d35e, #2f6690"
      - ["#111111", "#f4d35e"]
      - [[0.1, 0.2, 0.3], [255, 128, 0]]
    """
    if text is None or text == "":
        return torch.tensor(DEFAULT_PALETTE, dtype=torch.float32)

    if isinstance(text, str):
        raw_items: Sequence[Any] = [p.strip() for p in text.replace(";", ",").split(",")]
    else:
        raw_items = text

    colors: list[list[float]] = []
    for item in raw_items:
        if item is None or item == "":
            continue

        if isinstance(item, str):
            value = item.strip()
            if value.startswith("#"):
                value = value[1:]
            if len(value) == 3:
                value = "".join(ch * 2 for ch in value)
            if len(value) != 6:
                raise ValueError(f"Palette colour {item!r} must be #RGB or #RRGGBB.")
            colors.append([
                int(value[0:2], 16) / 255.0,
                int(value[2:4], 16) / 255.0,
                int(value[4:6], 16) / 255.0,
            ])
            continue

        vals = [float(v) for v in item]
        if len(vals) != 3:
            raise ValueError("Palette RGB entries must contain exactly 3 values.")
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        colors.append(vals)

    if not colors:
        colors = [list(c) for c in DEFAULT_PALETTE]

    return torch.tensor(colors, dtype=torch.float32).clamp(0.0, 1.0)


def image_to_tensor(image: str | Path | np.ndarray | torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Load/convert an image to (H, W, 3) float32 in [0, 1]."""
    if isinstance(image, torch.Tensor):
        t = image.detach().to(device=device, dtype=torch.float32)
        if t.max() > 1.0:
            t = t / 255.0
        return t[..., :3].clamp(0.0, 1.0)

    if isinstance(image, (str, Path)):
        from PIL import Image

        arr = np.array(Image.open(image).convert("RGB"))
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = arr[..., :3]
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    if t.max() > 1.0:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def quantize_to_palette(image: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Map every pixel to the nearest user-selected colour."""
    img = image[..., :3]
    pal = palette.to(device=img.device, dtype=img.dtype).clamp(0.0, 1.0)
    flat = img.reshape(-1, 3)
    distances = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(dim=-1)
    nearest = distances.argmin(dim=1)
    return pal[nearest].reshape_as(img)


def foreground_mask_from_image(
    image: str | Path | np.ndarray | torch.Tensor,
    palette: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Estimate target foreground as pixels that differ from the corner background."""
    img = image_to_tensor(image, device)
    q = quantize_to_palette(img, palette)
    h, w = q.shape[:2]
    band = max(1, min(h, w) // 20)
    corners = torch.cat([
        img[:band, :band].reshape(-1, 3),
        img[:band, -band:].reshape(-1, 3),
        img[-band:, :band].reshape(-1, 3),
        img[-band:, -band:].reshape(-1, 3),
    ])
    bg_rgb = corners.median(dim=0).values
    pal = palette.to(device=img.device, dtype=img.dtype)
    bg_idx = ((pal - bg_rgb.unsqueeze(0)) ** 2).sum(dim=1).argmin()
    bg_color = pal[bg_idx]
    mask = (((q - bg_color) ** 2).sum(dim=-1, keepdim=True) > 1e-6).float()
    if float(mask.mean().detach().cpu()) < 1e-4:
        distances = ((img - bg_rgb) ** 2).sum(dim=-1, keepdim=True)
        mask = (distances > 0.02 ** 2).float()
    return mask


def snap_patches_to_palette(
    patches: Sequence["Patch"],
    palette: str | Sequence[str] | Sequence[Sequence[float]] | torch.Tensor | None,
) -> torch.Tensor:
    """Snap each patch albedo to the nearest palette colour."""
    pal = palette if isinstance(palette, torch.Tensor) else parse_palette(palette)
    with torch.no_grad():
        for patch in patches:
            patch_palette = pal.to(device=patch.albedo.device, dtype=patch.albedo.dtype)
            rgb = patch.albedo.detach().clamp(0.0, 1.0)
            idx = ((patch_palette - rgb.unsqueeze(0)) ** 2).sum(dim=1).argmin()
            patch.albedo.copy_(patch_palette[idx])
            patch.albedo.requires_grad_(False)
    return pal


def _parameter_groups(patches: Sequence["Patch"]) -> list[torch.nn.Parameter]:
    """Return learnable shape/orientation parameters, excluding albedo."""
    params: list[torch.nn.Parameter] = []
    for patch in patches:
        params.extend([patch.center, patch.theta])
        for cp in patch.control_points:
            params.extend([
                cp.x,
                cp.y,
                cp.z,
                cp.handle_scale,
                cp.handle_rotation,
            ])
    return params


class SceneOptimizer:
    """Render, compare to quantized target images, and Adam-step patches."""

    def __init__(
        self,
        patches: list["Patch"],
        camera1: "Camera",
        camera2: "Camera",
        target1: str | Path | np.ndarray | torch.Tensor,
        target2: str | Path | np.ndarray | torch.Tensor | None = None,
        *,
        palette: str | Sequence[str] | Sequence[Sequence[float]] | None = None,
        renderer: DiffRenderer | None = None,
        resolution: tuple[int, int] = (256, 256),
        lr: float = 1e-3,
        view2_loss: str = "mse",
        sds_prompt: str = "",
        sds_pipe: Any | None = None,
        device: str = "cpu",
        n_per_segment: int = 20,
        silhouette_weight: float = 2.0,
    ) -> None:
        if not patches:
            raise ValueError("SceneOptimizer requires at least one patch.")

        self.patches = patches
        self.camera1 = camera1
        self.camera2 = camera2
        self.device = device
        self.resolution = resolution
        self.view2_loss = view2_loss.lower()
        self.sds_prompt = sds_prompt
        self.sds_pipe = sds_pipe
        self.silhouette_weight = silhouette_weight

        self.palette = parse_palette(palette).to(device)
        self.target1 = quantize_to_palette(image_to_tensor(target1, device), self.palette)
        self.target1_mask = foreground_mask_from_image(target1, self.palette, device)
        self.target2 = (
            quantize_to_palette(image_to_tensor(target2, device), self.palette)
            if target2 is not None else None
        )
        self.target2_mask = (
            foreground_mask_from_image(target2, self.palette, device)
            if target2 is not None else None
        )

        self.renderer = renderer or DiffRenderer(device=device, n_per_segment=n_per_segment)
        self.optim = torch.optim.Adam(_parameter_groups(patches), lr=lr)
        snap_patches_to_palette(self.patches, self.palette)

    def step(self) -> dict[str, float]:
        self.optim.zero_grad(set_to_none=True)

        render1, render2 = self.renderer.render_both(
            self.patches,
            self.camera1,
            self.camera2,
            self.resolution,
        )

        loss1_rgb = masked_rgb_loss(render1, self.target1, self.target1_mask)
        loss1_silhouette = silhouette_loss(render1, self.target1_mask)
        loss1 = loss1_rgb + self.silhouette_weight * loss1_silhouette
        loss2 = torch.zeros((), device=loss1.device)
        loss2_silhouette = torch.zeros((), device=loss1.device)
        if self.view2_loss.startswith("sds"):
            if not self.sds_prompt:
                raise ValueError("SDS optimization requires a text prompt.")
            loss2 = sds_loss(render2[..., :3], self.sds_prompt, self.sds_pipe)
        elif self.target2 is not None:
            assert self.target2_mask is not None
            loss2_rgb = masked_rgb_loss(render2, self.target2, self.target2_mask)
            loss2_silhouette = silhouette_loss(render2, self.target2_mask)
            loss2 = loss2_rgb + self.silhouette_weight * loss2_silhouette

        loss = loss1 + loss2
        loss.backward()
        self.optim.step()
        self._post_step_constraints()

        return {
            "loss": float(loss.detach().cpu()),
            "view1_mse": float(loss1.detach().cpu()),
            "view2_loss": float(loss2.detach().cpu()),
            "view1_silhouette": float(loss1_silhouette.detach().cpu()),
            "view2_silhouette": float(loss2_silhouette.detach().cpu()),
        }

    def _post_step_constraints(self) -> None:
        with torch.no_grad():
            for patch in self.patches:
                patch.center.data = torch.nan_to_num(patch.center.data, nan=0.0)
                patch.theta.data = torch.nan_to_num(patch.theta.data, nan=0.0)
                for cp in patch.control_points:
                    cp.x.data = torch.nan_to_num(cp.x.data, nan=0.0)
                    cp.y.data = torch.nan_to_num(cp.y.data, nan=0.0)
                    cp.z.data = torch.nan_to_num(cp.z.data, nan=0.0).clamp(-0.25, 0.25)
                    cp.handle_scale.data = torch.nan_to_num(cp.handle_scale.data, nan=0.01).clamp(0.01, 2.0)
                    cp.handle_rotation.data = torch.nan_to_num(cp.handle_rotation.data, nan=0.0)

    def mesh_snapshot(self, n_per_segment: int = 20) -> list["Mesh"]:
        return [p.to_mesh(n_per_segment=n_per_segment) for p in self.patches]

    def run(self, n_steps: int = 500) -> Iterator[tuple[int, dict[str, float]]]:
        for step_idx in range(1, n_steps + 1):
            yield step_idx, self.step()
