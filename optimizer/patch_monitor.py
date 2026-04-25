"""Patch health monitoring and restart helpers."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from core.patch import Patch
    from core.renderer import DiffRenderer
    from scene.camera import Camera


LossEvaluator = Callable[[Sequence["Patch"]], float]
_RESTART_BOX_SIZE = 5.0


def _camera_forward_tensor(camera: "Camera", device: str, dtype: torch.dtype) -> torch.Tensor:
    forward_np = camera.target - camera.position
    forward_np = forward_np / max(float(np.linalg.norm(forward_np)), 1e-8)
    return torch.from_numpy(forward_np).to(device=device, dtype=dtype)


def _patch_normal(patch: "Patch") -> torch.Tensor:
    normal = patch.rotation_matrix() @ patch.center.new_tensor([0.0, 0.0, 1.0])
    return normal / torch.linalg.norm(normal).clamp_min(1e-8)


def _patch_parameters(patch: "Patch") -> list[torch.nn.Parameter]:
    params = [patch.center, patch.theta]
    for cp in patch.control_points:
        params.extend([
            cp.x,
            cp.y,
            cp.z,
            cp.handle_scale,
            cp.handle_rotation,
        ])
    return params


def _reset_patch_control_points(
    patch: "Patch",
    radius: float = 0.18,
) -> None:
    """Reset a patch outline to the standard rounded-pentagon initialization."""
    n = len(patch.control_points)
    handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / max(n, 1))
    device = patch.center.device
    dtype = patch.center.dtype

    for i, cp in enumerate(patch.control_points):
        angle = 2.0 * math.pi * i / max(n, 1) - math.pi / 2.0
        x_local = radius * math.cos(angle)
        y_local = radius * math.sin(angle)
        handle_rotation = angle + math.pi / 2.0

        cp.x.copy_(torch.tensor(x_local, device=device, dtype=dtype))
        cp.y.copy_(torch.tensor(y_local, device=device, dtype=dtype))
        cp.z.zero_()
        cp.handle_scale.copy_(torch.tensor(handle_scale, device=device, dtype=dtype))
        cp.handle_rotation.copy_(torch.tensor(handle_rotation, device=device, dtype=dtype))


def reset_optimizer_state(
    optimizer: torch.optim.Optimizer,
    patch: "Patch",
) -> None:
    """Clear Adam state for every parameter owned by one patch."""
    for param in _patch_parameters(patch):
        optimizer.state.pop(param, None)


def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) * 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, kernel_size, kernel_size)


def _blur_error_map(error: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    if error.ndim == 3:
        error = error.squeeze(-1)
    pad = kernel_size // 2
    sigma = max(kernel_size / 3.0, 1.0)
    kernel = _gaussian_kernel(kernel_size, sigma, error.device, error.dtype)
    return F.conv2d(error[None, None], kernel, padding=pad)[0, 0]


def _error_map(rendered: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
    if target is None:
        return torch.zeros(rendered.shape[:2], device=rendered.device, dtype=rendered.dtype)
    return (rendered[..., :3] - target[..., :3]).square().mean(dim=-1)


def _project_world_point(
    camera: "Camera",
    point: torch.Tensor,
    resolution: tuple[int, int],
) -> tuple[int, int] | None:
    height, width = resolution
    point_np = point.detach().cpu().numpy().astype(np.float32)
    point_h = np.array([point_np[0], point_np[1], point_np[2], 1.0], dtype=np.float32)
    clip = point_h @ (camera.projection_matrix() @ camera.view_matrix()).T
    if abs(float(clip[3])) < 1e-8:
        return None
    ndc = clip[:3] / clip[3]
    if not np.isfinite(ndc).all():
        return None
    col = int(round((float(ndc[0]) * 0.5 + 0.5) * (width - 1)))
    row = int(round((0.5 - float(ndc[1]) * 0.5) * (height - 1)))
    if row < 0 or row >= height or col < 0 or col >= width:
        return None
    return row, col


def _project_world_np(
    camera: "Camera",
    point: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[int, int] | None:
    height, width = resolution
    point_h = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    clip = point_h @ (camera.projection_matrix() @ camera.view_matrix()).T
    if abs(float(clip[3])) < 1e-8:
        return None
    ndc = clip[:3] / clip[3]
    if not np.isfinite(ndc).all():
        return None
    if abs(float(ndc[0])) > 1.0 or abs(float(ndc[1])) > 1.0 or abs(float(ndc[2])) > 1.0:
        return None
    col = int(round((float(ndc[0]) * 0.5 + 0.5) * (width - 1)))
    row = int(round((0.5 - float(ndc[1]) * 0.5) * (height - 1)))
    if row < 0 or row >= height or col < 0 or col >= width:
        return None
    return row, col


def _placement_from_error_in_box(
    cameras: Sequence["Camera"],
    error_maps: Sequence[torch.Tensor],
    resolution: tuple[int, int],
    placement_mask: torch.Tensor | None,
    box_size: float = _RESTART_BOX_SIZE,
    n_candidates: int = 4096,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Choose a random point inside a 5x5x5 box that projects to high error."""
    half = box_size * 0.5
    candidates = np.random.uniform(-half, half, size=(n_candidates, 3)).astype(np.float32)
    errors_cpu = [error.detach().cpu() for error in error_maps]
    mask_cpu = placement_mask.detach().cpu() if placement_mask is not None else None

    best_score = -math.inf
    best_point = candidates[0]
    best_pixel = (0, 0)
    fallback_score = -math.inf
    fallback_point = candidates[0]
    fallback_pixel = (0, 0)

    for point in candidates:
        projected = [
            (pixel, error)
            for camera, error in zip(cameras, errors_cpu, strict=False)
            if (pixel := _project_world_np(camera, point, resolution)) is not None
        ]
        if len(projected) < len(errors_cpu):
            continue

        scores: list[float] = []
        mask_scores: list[float] = []
        for (row, col), error in projected:
            scores.append(float(error[row, col]))
            if mask_cpu is not None:
                mask_scores.append(float(mask_cpu[row, col]))
        score = float(np.mean(scores))
        mask_score = float(np.mean(mask_scores)) if mask_scores else 1.0

        if score > fallback_score:
            fallback_score = score
            fallback_point = point
            fallback_pixel = projected[0][0]

        masked_score = score * mask_score
        if masked_score > best_score:
            best_score = masked_score
            best_point = point
            best_pixel = projected[0][0]

    if best_score <= 0.0 and fallback_score > -math.inf:
        return fallback_point, fallback_pixel
    return best_point, best_pixel


def _healthy_coverage(
    renderer: "DiffRenderer",
    patches: Sequence["Patch"],
    cameras: Sequence["Camera"],
    resolution: tuple[int, int],
) -> torch.Tensor | None:
    healthy = list(patches)
    if not healthy:
        return None
    with torch.no_grad():
        cov1, cov2 = renderer.render_both(healthy, cameras[0], cameras[1], resolution)
        return ((cov1[..., 3] + cov2[..., 3]) * 0.5).clamp(0.0, 1.0)


def restart_patch(
    patch: "Patch",
    patch_index: int,
    patches: Sequence["Patch"],
    renderer: "DiffRenderer",
    cameras: Sequence["Camera"],
    target1: torch.Tensor,
    target2: torch.Tensor | None,
    resolution: tuple[int, int],
    optimizer: torch.optim.Optimizer,
    rendered1: torch.Tensor,
    rendered2: torch.Tensor,
    reasons: Sequence[str],
    placement_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Move a degenerate patch to a high-error region and reset its shape."""
    with torch.no_grad():
        err1 = _blur_error_map(_error_map(rendered1, target1), kernel_size=15)
        err2 = _blur_error_map(_error_map(rendered2, target2), kernel_size=15)
        error_maps = [err1] if target2 is None else [err1, err2]

        healthy = [p for i, p in enumerate(patches) if i != patch_index]
        coverage = _healthy_coverage(renderer, healthy, cameras, resolution)
        score_mask = placement_mask
        if coverage is not None:
            uncovered = (1.0 - coverage).clamp(0.0, 1.0)
            score_mask = uncovered if score_mask is None else score_mask * uncovered

        height, width = err1.shape
        placement, (row, col) = _placement_from_error_in_box(
            cameras[:len(error_maps)],
            error_maps,
            resolution,
            score_mask,
        )
        if placement_mask is not None:
            yy, xx = torch.meshgrid(
                torch.arange(height, device=placement_mask.device),
                torch.arange(width, device=placement_mask.device),
                indexing="ij",
            )
            radius = max(8, min(height, width) // 12)
            used = (yy - row).square() + (xx - col).square() <= radius * radius
            placement_mask[used] = 0.0

        if not np.isfinite(placement).all():
            placement = np.zeros(3, dtype=np.float32)

        device = patch.center.device
        dtype = patch.center.dtype
        patch.center.copy_(torch.tensor(placement, device=device, dtype=dtype))
        patch.theta.zero_()
        _reset_patch_control_points(patch)

    reset_optimizer_state(optimizer, patch)
    reason_text = ", ".join(reasons)
    print(
        f"[Patch restart] index={patch_index}, reasons={reason_text}, "
        f"placed=({placement[0]:.3f}, {placement[1]:.3f}, {placement[2]:.3f})"
    )
    return placement


class PatchHealthMonitor:
    """Track degenerate patch behavior and restart unhealthy patches."""

    def __init__(
        self,
        patches: Sequence["Patch"],
        cameras: Sequence["Camera"],
        renderer: "DiffRenderer",
        target1: torch.Tensor,
        target2: torch.Tensor | None,
        resolution: tuple[int, int],
        optimizer: torch.optim.Optimizer,
        *,
        restart_interval: int = 100,
        edge_on_threshold: float = 0.1,
        plateau_window: int = 50,
        plateau_threshold: float = 1e-6,
        bottom_contribution_fraction: float = 0.1,
        max_restart_fraction: float = 0.2,
    ) -> None:
        self.patches = patches
        self.cameras = cameras
        self.renderer = renderer
        self.target1 = target1
        self.target2 = target2
        self.resolution = resolution
        self.optimizer = optimizer
        self.restart_interval = restart_interval
        self.edge_on_threshold = edge_on_threshold
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.bottom_contribution_fraction = bottom_contribution_fraction
        self.max_restart_fraction = max_restart_fraction
        self.loss_histories = [deque(maxlen=plateau_window) for _ in patches]
        self.last_contributions = [0.0 for _ in patches]
        self.restart_count = 0

    def update(
        self,
        step_idx: int,
        rendered1: torch.Tensor,
        rendered2: torch.Tensor,
    ) -> None:
        """Record per-patch image-error history used by plateau detection."""
        with torch.no_grad():
            errors = [
                _error_map(rendered1, self.target1),
                _error_map(rendered2, self.target2),
            ]
            for idx, patch in enumerate(self.patches):
                samples: list[float] = []
                for camera, error in zip(self.cameras, errors, strict=False):
                    pixel = _project_world_point(camera, patch.center, self.resolution)
                    if pixel is None:
                        continue
                    row, col = pixel
                    samples.append(float(error[row, col].detach().cpu()))
                if samples:
                    self.loss_histories[idx].append(float(np.mean(samples)))

    def _edge_on_reasons(self, patch: "Patch") -> list[str]:
        reasons: list[str] = []
        normal = _patch_normal(patch)
        for cam_idx, camera in enumerate(self.cameras, start=1):
            forward = _camera_forward_tensor(camera, str(normal.device), normal.dtype)
            facing = float(torch.abs(torch.dot(normal, forward)).detach().cpu())
            if facing < self.edge_on_threshold:
                reasons.append(f"edge-on view {cam_idx} ({facing:.4f})")
        return reasons

    def _plateaued(self, patch_index: int) -> bool:
        history = self.loss_histories[patch_index]
        if len(history) < self.plateau_window:
            return False
        return float(np.std(np.array(history, dtype=np.float64))) < self.plateau_threshold

    def _loss_contributions(
        self,
        current_loss: float,
        evaluate_loss: LossEvaluator,
    ) -> list[float]:
        contributions: list[float] = []
        for idx in range(len(self.patches)):
            subset = [patch for j, patch in enumerate(self.patches) if j != idx]
            removed_loss = evaluate_loss(subset)
            contributions.append(removed_loss - current_loss)
        self.last_contributions = contributions
        return contributions

    def check_and_restart(
        self,
        step_idx: int,
        current_loss: float,
        evaluate_loss: LossEvaluator,
        rendered1: torch.Tensor,
        rendered2: torch.Tensor,
    ) -> int:
        """Restart unhealthy patches when the configured interval is reached."""
        if self.restart_interval <= 0 or step_idx % self.restart_interval != 0:
            return 0

        contributions = self._loss_contributions(current_loss, evaluate_loss)
        if contributions:
            cutoff = float(np.quantile(np.array(contributions), self.bottom_contribution_fraction))
        else:
            cutoff = -math.inf

        flagged: list[tuple[int, list[str]]] = []
        for idx, patch in enumerate(self.patches):
            reasons = self._edge_on_reasons(patch)
            if contributions[idx] <= cutoff:
                reasons.append(f"low contribution ({contributions[idx]:.6f})")
            if self._plateaued(idx):
                reasons.append("plateaued loss")
            if reasons:
                flagged.append((idx, reasons))

        if not flagged:
            return 0

        max_restarts = max(1, int(math.ceil(len(self.patches) * self.max_restart_fraction)))
        flagged = flagged[:max_restarts]
        placement_mask = torch.ones(
            rendered1.shape[:2],
            device=rendered1.device,
            dtype=rendered1.dtype,
        )

        for idx, reasons in flagged:
            restart_patch(
                self.patches[idx],
                idx,
                self.patches,
                self.renderer,
                self.cameras,
                self.target1,
                self.target2,
                self.resolution,
                self.optimizer,
                rendered1,
                rendered2,
                reasons,
                placement_mask,
            )
            self.loss_histories[idx].clear()
            self.restart_count += 1

        return len(flagged)
