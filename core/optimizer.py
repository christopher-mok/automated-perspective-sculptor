"""Optimization loop for the anamorphic sculpture.

SceneOptimizer wraps the render → loss → backward → step cycle.  It is a
plain Python object with no Qt dependency so it can be used from scripts,
notebooks, or the UI worker thread equally.

Typical use (from a script)
---------------------------
    optimizer = SceneOptimizer(patches, cam1, cam2, target1_mask, target2_mask, lr=1e-3)
    for step, metrics in optimizer.run(n_steps=500):
        print(step, metrics["loss"])

Typical use (from the UI worker)
---------------------------------
    See ui/worker.py — the worker wraps ``run()`` in a QThread and emits
    Qt signals for each step.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from core.loss import sds_loss, silhouette_loss
from core.renderer import DiffRenderer

if TYPE_CHECKING:
    from core.patch import Patch
    from scene.camera import Camera
    from scene.scene import Mesh


DEFAULT_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.95, 0.95),
    (0.05, 0.05, 0.05),
)
THETA_CAMERA_MARGIN: float = np.deg2rad(15.0)


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


def mask_to_tensor(mask: np.ndarray | torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Convert a binary mask to (H, W, 1) float32 in {0, 1}."""
    if isinstance(mask, torch.Tensor):
        t = mask.detach().to(device=device, dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(mask)).to(device=device, dtype=torch.float32)

    if t.ndim == 2:
        t = t.unsqueeze(-1)
    elif t.ndim == 3:
        # Support both HWC and CHW layouts.
        if t.shape[-1] in (1, 3, 4):
            t = t[..., :1]
        elif t.shape[0] in (1, 3, 4):
            t = t[:1].permute(1, 2, 0)
        else:
            raise ValueError(
                f"Unsupported 3D mask shape {tuple(t.shape)}. "
                "Expected HxWx1 or 1xHxW."
            )
    else:
        raise ValueError(f"Mask must have shape (H,W) or (H,W,1), got {tuple(t.shape)}.")

    if t.max() > 1.0:
        t = t / 255.0
    return (t >= 0.5).to(dtype=torch.float32)


def fit_mask_to_resolution(
    mask: np.ndarray | torch.Tensor,
    resolution: tuple[int, int],
    device: str = "cpu",
) -> torch.Tensor:
    """Center-fit a binary mask into target resolution using nearest resize."""
    src = mask_to_tensor(mask, device)
    target_h, target_w = resolution
    src_h, src_w = src.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return torch.zeros(target_h, target_w, 1, device=device)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = src.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        resized,
        size=(new_h, new_w),
        mode="nearest",
    ).squeeze(0).permute(1, 2, 0)

    canvas = torch.zeros(target_h, target_w, 1, device=device, dtype=torch.float32)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


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


def _wrap_theta_half_turn(theta: float) -> float:
    """Wrap a Y rotation to [-pi/2, pi/2), treating theta and theta+pi as equivalent."""
    return ((theta + np.pi * 0.5) % np.pi) - np.pi * 0.5


def _theta_distance(a: float, b: float) -> float:
    """Shortest angular distance when opposite patch normals are equivalent."""
    return abs(_wrap_theta_half_turn(a - b))


def _camera_yaw_angles(cameras: Sequence["Camera"]) -> list[float]:
    """Camera yaw angles in the same theta convention used by Patch."""
    angles: list[float] = []
    for camera in cameras:
        offset = camera.position - camera.target
        angles.append(_wrap_theta_half_turn(float(np.arctan2(offset[0], offset[2]))))
    return angles


def theta_allowed(
    theta: float,
    camera_angles: Sequence[float],
    margin: float = THETA_CAMERA_MARGIN,
) -> bool:
    """Return True when theta is at least margin radians away from every camera yaw."""
    return all(_theta_distance(theta, angle) >= margin for angle in camera_angles)


def constrain_theta_to_camera_band(
    theta: float,
    camera_angles: Sequence[float],
    margin: float = THETA_CAMERA_MARGIN,
) -> float:
    """Project theta to the nearest orientation outside the camera edge-on margin."""
    theta = _wrap_theta_half_turn(theta)
    if theta_allowed(theta, camera_angles, margin):
        return theta

    candidates: list[float] = []
    for angle in camera_angles:
        candidates.append(_wrap_theta_half_turn(angle - margin))
        candidates.append(_wrap_theta_half_turn(angle + margin))

    valid = [
        candidate for candidate in candidates
        if theta_allowed(candidate, camera_angles, margin * 0.999)
    ]
    if not valid:
        valid = candidates
    return min(valid, key=lambda candidate: _theta_distance(theta, candidate))


class SceneOptimizer:
    """Render, compare to binary silhouette masks, and Adam-step patches."""

    def __init__(
        self,
        patches: list["Patch"],
        camera1: "Camera",
        camera2: "Camera",
        target1_mask: np.ndarray | torch.Tensor,
        target2_mask: np.ndarray | torch.Tensor | None = None,
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
        theta_camera_margin: float = THETA_CAMERA_MARGIN,
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
        self.theta_camera_margin = theta_camera_margin
        self.theta_camera_angles = _camera_yaw_angles((camera1, camera2))
        self.camera_bounds_weight = camera_bounds_weight
        self.camera_bounds_xy_limit = camera_bounds_xy_limit

        self.palette = parse_palette(palette).to(device)
        self.target1_mask = fit_mask_to_resolution(target1_mask, self.resolution, device)
        self.target2_mask = (
            fit_mask_to_resolution(target2_mask, self.resolution, device)
            if target2_mask is not None else None
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
            "view1_mask": float(components["loss1"].detach().cpu()),
            "view2_target": float(components["loss2"].detach().cpu()),
            "view1_mse": float(components["loss1"].detach().cpu()),
            "view2_loss": float(components["loss2"].detach().cpu()),
            "view1_silhouette": float(components["loss1_silhouette"].detach().cpu()),
            "view2_silhouette": float(components["loss2_silhouette"].detach().cpu()),
            "overlap": float(components["overlap"].detach().cpu()),
            "visibility": float(components["visibility"].detach().cpu()),
            "camera_bounds": float(components["camera_bounds"].detach().cpu()),
            "overlap_weighted": float((self.overlap_weight * components["overlap"]).detach().cpu()),
            "visibility_weighted": float((self.visibility_weight * components["visibility"]).detach().cpu()),
            "camera_bounds_weighted": float((self.camera_bounds_weight * components["camera_bounds"]).detach().cpu()),
        }

    def _loss_from_renders(
        self,
        render1: torch.Tensor,
        render2: torch.Tensor,
        patches: Sequence["Patch"],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # silhouette_loss compares rendered alpha to a binary target mask.
        # This makes white input background (mask=0) equivalent to alpha=0 output.
        loss1_silhouette = silhouette_loss(render1, self.target1_mask)
        loss1 = self.silhouette_weight * loss1_silhouette
        loss2_term = torch.zeros((), device=loss1.device)
        loss2 = torch.zeros((), device=loss1.device)
        loss2_silhouette = torch.zeros((), device=loss1.device)
        if self.view2_loss.startswith("sds"):
            if not self.sds_prompt:
                raise ValueError("SDS optimization requires a text prompt.")
            loss2 = sds_loss(render2[..., :3], self.sds_prompt, self.sds_pipe)
            loss2_term = loss2
        elif self.target2_mask is not None:
            assert self.target2_mask is not None
            loss2_silhouette = silhouette_loss(render2, self.target2_mask)
            loss2_term = loss2_silhouette
            loss2 = self.silhouette_weight * loss2_silhouette

        if patches:
            overlap = patch_overlap_loss(patches, self.overlap_margin)
            visibility = patch_visibility_loss(
                patches,
                (self.camera1, self.camera2),
                self.min_projected_area,
            )
            camera_bounds = patch_camera_bounds_loss(
                patches,
                (self.camera1, self.camera2),
                self.camera_bounds_xy_limit,
            )
        else:
            overlap = torch.zeros((), device=loss1.device)
            visibility = torch.zeros((), device=loss1.device)
            camera_bounds = torch.zeros((), device=loss1.device)
        loss = (
            loss1
            + loss2
            + self.overlap_weight * overlap
            + self.visibility_weight * visibility
            + self.camera_bounds_weight * camera_bounds
        )
        return loss, {
            "loss1": loss1_silhouette,
            "loss2": loss2_term,
            "loss1_silhouette": loss1_silhouette,
            "loss2_silhouette": loss2_silhouette,
            "overlap": overlap,
            "visibility": visibility,
            "camera_bounds": camera_bounds,
        }

    def _post_step_constraints(self) -> None:
        with torch.no_grad():
            for patch in self.patches:
                patch.center.data = torch.nan_to_num(patch.center.data, nan=0.0)
                patch.theta.data = torch.nan_to_num(patch.theta.data, nan=0.0)
                constrained_theta = constrain_theta_to_camera_band(
                    float(patch.theta.detach().cpu()),
                    self.theta_camera_angles,
                    self.theta_camera_margin,
                )
                patch.theta.copy_(patch.theta.new_tensor(constrained_theta))
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
