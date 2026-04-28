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
import torch.nn.functional as F

from core.loss import masked_rgb_loss, sds_loss, silhouette_loss
from core.renderer import DiffRenderer

if TYPE_CHECKING:
    from core.patch import Patch
    from scene.camera import Camera
    from scene.scene import Mesh


DEFAULT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.95, 0.95),
    (0.05, 0.05, 0.05),
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


def fit_image_to_resolution(
    image: str | Path | np.ndarray | torch.Tensor,
    resolution: tuple[int, int],
    device: str = "cpu",
) -> torch.Tensor:
    """Scale an image into a fixed canvas without changing its aspect ratio."""
    img = image_to_tensor(image, device)
    target_h, target_w = resolution
    src_h, src_w = img.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return torch.zeros(target_h, target_w, 3, device=device)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    corners = torch.stack([
        img[0, 0],
        img[0, -1],
        img[-1, 0],
        img[-1, -1],
    ])
    background = corners.median(dim=0).values
    canvas = background.view(1, 1, 3).expand(target_h, target_w, 3).clone()

    resized = img.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        resized,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas.clamp(0.0, 1.0)


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
    resolution: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Estimate target foreground as pixels that differ from the corner background."""
    img = (
        fit_image_to_resolution(image, resolution, device)
        if resolution is not None else image_to_tensor(image, device)
    )
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
            cp.z.requires_grad_(False)
            cp.z.grad = None
            params.extend([
                cp.x,
                cp.y,
                cp.handle_scale,
                cp.handle_rotation,
            ])
    return params


def _patch_collision_radius(patch: "Patch") -> torch.Tensor:
    """Conservative world-space radius enclosing one flat patch."""
    max_radius = torch.zeros((), device=patch.center.device)
    for cp in patch.control_points:
        pos = cp.pos
        handle = cp.handle_out()
        cp_radius = torch.stack([
            torch.linalg.norm(pos),
            torch.linalg.norm(pos + handle),
            torch.linalg.norm(pos - handle),
        ]).max()
        max_radius = torch.maximum(max_radius, cp_radius)
    return max_radius + patch.DEFAULT_THICKNESS * 0.5


def patch_overlap_loss(
    patches: Sequence["Patch"],
    margin: float = 0.005,
) -> torch.Tensor:
    """Soft penalty for overlapping conservative patch collision radii."""
    if len(patches) < 2:
        device = patches[0].center.device if patches else "cpu"
        return torch.zeros((), device=device)

    losses: list[torch.Tensor] = []
    radii = [_patch_collision_radius(patch) for patch in patches]
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            delta = patches[j].center - patches[i].center
            dist = torch.linalg.norm(delta) + 1e-8
            allowed = radii[i] + radii[j] + margin
            overlap = torch.relu(allowed - dist)
            losses.append(overlap ** 2)

    if not losses:
        return torch.zeros((), device=patches[0].center.device)
    return torch.stack(losses).mean()


def _camera_mvp_tensor(camera: "Camera", device: str) -> torch.Tensor:
    view = torch.from_numpy(camera.view_matrix()).to(device=device, dtype=torch.float32)
    proj = torch.from_numpy(camera.projection_matrix()).to(device=device, dtype=torch.float32)
    return proj @ view


def _patch_projected_area(
    patch: "Patch",
    camera: "Camera",
    n_per_segment: int = 6,
) -> torch.Tensor:
    """Approximate one patch's screen-space area in normalized device coords."""
    pts = patch.sample_spline_world(n_per_segment)
    ones = torch.ones(len(pts), 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)

    mvp = _camera_mvp_tensor(camera, str(pts.device)).to(dtype=pts.dtype)
    clip = pts_h @ mvp.T
    w = clip[:, 3].clamp_min(1e-6)
    ndc = clip[:, :2] / w.unsqueeze(1)

    x = ndc[:, 0]
    y = ndc[:, 1]
    area = 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1))
    )

    front_fraction = (clip[:, 3] > 1e-5).to(dtype=pts.dtype).mean().detach()
    return area * front_fraction


def patch_visibility_loss(
    patches: Sequence["Patch"],
    cameras: Sequence["Camera"],
    min_projected_area: float = 0.0015,
) -> torch.Tensor:
    """Soft per-piece, per-camera penalty for pieces that are barely visible."""
    if not patches:
        return torch.zeros(())

    losses: list[torch.Tensor] = []
    for patch in patches:
        for camera in cameras:
            area = _patch_projected_area(patch, camera)
            shortfall = torch.relu(area.new_tensor(min_projected_area) - area)
            losses.append((shortfall / max(min_projected_area, 1e-8)) ** 2)

    if not losses:
        return torch.zeros((), device=patches[0].center.device)
    return torch.stack(losses).mean()


def _patch_camera_bounds_loss(
    patch: "Patch",
    camera: "Camera",
    n_per_segment: int = 6,
    xy_limit: float = 0.98,
) -> torch.Tensor:
    """Soft penalty for patch outline points outside one camera frustum."""
    pts = patch.sample_spline_world(n_per_segment)
    ones = torch.ones(len(pts), 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)

    mvp = _camera_mvp_tensor(camera, str(pts.device)).to(dtype=pts.dtype)
    clip = pts_h @ mvp.T
    w = clip[:, 3]
    w_safe = w.abs().clamp_min(1e-4)
    ndc = clip[:, :3] / w_safe.unsqueeze(1)

    xy_excess = torch.relu(torch.abs(ndc[:, :2]) - xy_limit)
    z_excess = torch.relu(torch.abs(ndc[:, 2]) - 1.0)
    behind = torch.relu(w.new_tensor(1e-4) - w)
    return xy_excess.square().mean() + z_excess.square().mean() + behind.square().mean()


def patch_camera_bounds_loss(
    patches: Sequence["Patch"],
    cameras: Sequence["Camera"],
    xy_limit: float = 0.98,
) -> torch.Tensor:
    """Soft penalty for pieces drifting outside either camera view."""
    if not patches:
        return torch.zeros(())

    losses: list[torch.Tensor] = []
    for patch in patches:
        for camera in cameras:
            losses.append(_patch_camera_bounds_loss(patch, camera, xy_limit=xy_limit))

    if not losses:
        return torch.zeros((), device=patches[0].center.device)
    return torch.stack(losses).mean()


def _camera_forward_tensor(camera: "Camera", device: str, dtype: torch.dtype) -> torch.Tensor:
    forward_np = camera.target - camera.position
    forward_np = forward_np / max(float(np.linalg.norm(forward_np)), 1e-8)
    return torch.from_numpy(forward_np).to(device=device, dtype=dtype)


def _patch_normal(patch: "Patch") -> torch.Tensor:
    normal = patch.rotation_matrix() @ patch.center.new_tensor([0.0, 0.0, 1.0])
    return normal / torch.linalg.norm(normal).clamp_min(1e-8)


def patch_edge_on_loss(
    patches: Sequence["Patch"],
    cameras: Sequence["Camera"],
    min_facing: float = 0.18,
) -> torch.Tensor:
    """Soft penalty for patches that become nearly edge-on to a camera."""
    if not patches:
        return torch.zeros(())

    losses: list[torch.Tensor] = []
    for patch in patches:
        normal = _patch_normal(patch)
        for camera in cameras:
            forward = _camera_forward_tensor(camera, str(normal.device), normal.dtype)
            facing = torch.abs(torch.dot(normal, forward))
            shortfall = torch.relu(facing.new_tensor(min_facing) - facing)
            losses.append((shortfall / max(min_facing, 1e-8)) ** 2)

    if not losses:
        return torch.zeros((), device=patches[0].center.device)
    return torch.stack(losses).mean()


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
        resolution: tuple[int, int] = (192, 256),
        lr: float = 1e-3,
        view2_loss: str = "mse",
        sds_prompt: str = "",
        sds_pipe: Any | None = None,
        device: str = "cpu",
        n_per_segment: int = 20,
        silhouette_weight: float = 2.0,
        #overlap_weight: float = 0.05,
        overlap_weight: float = 0.7,
        overlap_margin: float = 0.005,
        # visibility_weight: float = 0.05,
        # min_projected_area: float = 0.0015,
        visibility_weight: float = 2,
        min_projected_area: float = 0.01,
        edge_on_weight: float = 0.02,
        min_camera_facing: float = 0.18,
        camera_bounds_weight: float = 0.3,
        camera_bounds_xy_limit: float = 0.98,
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
        self.overlap_weight = overlap_weight
        self.overlap_margin = overlap_margin
        self.visibility_weight = visibility_weight
        self.min_projected_area = min_projected_area
        self.edge_on_weight = edge_on_weight
        self.min_camera_facing = min_camera_facing
        self.camera_bounds_weight = camera_bounds_weight
        self.camera_bounds_xy_limit = camera_bounds_xy_limit

        self.palette = parse_palette(palette).to(device)
        target1_fit = fit_image_to_resolution(target1, self.resolution, device)
        self.target1 = quantize_to_palette(target1_fit, self.palette)
        self.target1_mask = foreground_mask_from_image(
            target1_fit,
            self.palette,
            device,
        )
        target2_fit = (
            fit_image_to_resolution(target2, self.resolution, device)
            if target2 is not None else None
        )
        self.target2 = (
            quantize_to_palette(target2_fit, self.palette)
            if target2_fit is not None else None
        )
        self.target2_mask = (
            foreground_mask_from_image(target2_fit, self.palette, device)
            if target2_fit is not None else None
        )

        self.renderer = renderer or DiffRenderer(device=device, n_per_segment=n_per_segment)
        self.optim = torch.optim.Adam(_parameter_groups(patches), lr=lr)
        snap_patches_to_palette(self.patches, self.palette)
        self._post_step_constraints()

    def step(self, step_idx: int = 1, total_steps: int = 1) -> dict[str, float]:
        self.optim.zero_grad(set_to_none=True)

        render1, render2 = self.renderer.render_both(
            self.patches,
            self.camera1,
            self.camera2,
            self.resolution,
        )

        loss, components = self._loss_from_renders(render1, render2, self.patches)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        self._post_step_constraints()

        loss_value = float(loss.detach().cpu())
        return {
            "loss": loss_value,
            "view1_mse": float(components["loss1"].detach().cpu()),
            "view2_loss": float(components["loss2"].detach().cpu()),
            "view1_silhouette": float(components["loss1_silhouette"].detach().cpu()),
            "view2_silhouette": float(components["loss2_silhouette"].detach().cpu()),
            "overlap": float(components["overlap"].detach().cpu()),
            "visibility": float(components["visibility"].detach().cpu()),
            "edge_on": float(components["edge_on"].detach().cpu()),
            "camera_bounds": float(components["camera_bounds"].detach().cpu()),
        }

    def _loss_from_renders(
        self,
        render1: torch.Tensor,
        render2: torch.Tensor,
        patches: Sequence["Patch"],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        if patches:
            overlap = patch_overlap_loss(patches, self.overlap_margin)
            visibility = patch_visibility_loss(
                patches,
                (self.camera1, self.camera2),
                self.min_projected_area,
            )
            edge_on = patch_edge_on_loss(
                patches,
                (self.camera1, self.camera2),
                self.min_camera_facing,
            )
            camera_bounds = patch_camera_bounds_loss(
                patches,
                (self.camera1, self.camera2),
                self.camera_bounds_xy_limit,
            )
        else:
            overlap = torch.zeros((), device=loss1.device)
            visibility = torch.zeros((), device=loss1.device)
            edge_on = torch.zeros((), device=loss1.device)
            camera_bounds = torch.zeros((), device=loss1.device)
        loss = (
            loss1
            + loss2
            + self.overlap_weight * overlap
            + self.visibility_weight * visibility
            + self.edge_on_weight * edge_on
            + self.camera_bounds_weight * camera_bounds
        )
        return loss, {
            "loss1": loss1,
            "loss2": loss2,
            "loss1_silhouette": loss1_silhouette,
            "loss2_silhouette": loss2_silhouette,
            "overlap": overlap,
            "visibility": visibility,
            "edge_on": edge_on,
            "camera_bounds": camera_bounds,
        }

    def _post_step_constraints(self) -> None:
        with torch.no_grad():
            for patch in self.patches:
                patch.center.data = torch.nan_to_num(patch.center.data, nan=0.0)
                patch.theta.data = torch.nan_to_num(patch.theta.data, nan=0.0)
                for cp in patch.control_points:
                    cp.x.data = torch.nan_to_num(cp.x.data, nan=0.0)
                    cp.y.data = torch.nan_to_num(cp.y.data, nan=0.0)
                    cp.z.data.zero_()
                    cp.handle_scale.data = torch.nan_to_num(cp.handle_scale.data, nan=0.01).clamp(0.01, 2.0)
                    cp.handle_rotation.data = torch.nan_to_num(cp.handle_rotation.data, nan=0.0)

    def mesh_snapshot(self, n_per_segment: int = 20) -> list["Mesh"]:
        return [p.to_mesh(n_per_segment=n_per_segment) for p in self.patches]

    def run(self, n_steps: int = 500) -> Iterator[tuple[int, dict[str, float]]]:
        for step_idx in range(1, n_steps + 1):
            yield step_idx, self.step(step_idx, n_steps)
