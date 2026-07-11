"""Approximate swept volume from two target-image silhouettes."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.patch import Patch
    from scene.camera import Camera
    from scene.scene import Mesh


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr > 10
    if arr.shape[-1] >= 4:
        return arr[..., 3] > 10

    rgb = arr[..., :3].astype(np.float32)
    h, w = rgb.shape[:2]
    band = max(1, min(h, w) // 20)
    corners = np.concatenate([
        rgb[:band, :band].reshape(-1, 3),
        rgb[:band, -band:].reshape(-1, 3),
        rgb[-band:, :band].reshape(-1, 3),
        rgb[-band:, -band:].reshape(-1, 3),
    ])
    background = np.median(corners, axis=0)
    return np.sum((rgb - background) ** 2, axis=-1) > (8.0 ** 2)


def _project_points(
    points: np.ndarray,
    camera: "Camera",
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    homog = np.c_[pts, np.ones(len(pts), dtype=np.float32)]
    mvp = camera.projection_matrix() @ camera.view_matrix()
    clip = homog @ mvp.T
    clip_w = clip[:, 3]
    valid = clip_w > 1e-6
    ndc = np.zeros((len(pts), 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / clip_w[valid, None]

    px = np.rint((ndc[:, 0] * 0.5 + 0.5) * (w - 1)).astype(np.int32)
    py = np.rint((1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)).astype(np.int32)
    valid &= (
        (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= -1.0)
        & (ndc[:, 2] <= 1.0)
        & (px >= 0)
        & (px < w)
        & (py >= 0)
        & (py < h)
    )
    return px, py, valid


@dataclass
class SweptVolume:
    """Voxel approximation of the intersection of two silhouette extrusions."""

    points: np.ndarray
    grid_step: float
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    masks: tuple[np.ndarray, np.ndarray]
    cell_scores: tuple[np.ndarray, np.ndarray]
    cameras: tuple["Camera", "Camera"]

    @classmethod
    def from_images(
        cls,
        image1: np.ndarray,
        image2: np.ndarray,
        cameras: list["Camera"],
        *,
        hanging_plane_size: float,
        resolution: int = 108,
        inflate_steps: int = 2,
        slice_batch_size: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "SweptVolume":
        if len(cameras) < 2:
            raise ValueError("Swept volume requires two cameras.")

        mask1 = _foreground_mask(image1)
        mask2 = _foreground_mask(image2)
        score1 = _silhouette_cell_scores(mask1)
        score2 = _silhouette_cell_scores(mask2)
        half = max(float(hanging_plane_size) * 0.5, 1e-4)
        bounds_min = np.array([-half, -half, -half], dtype=np.float32)
        bounds_max = np.array([half, half, half], dtype=np.float32)
        axis = np.linspace(-half, half, max(4, int(resolution)), dtype=np.float32)

        kept_batches: list[np.ndarray] = []
        total_slices = len(axis)
        batch_size = max(1, int(slice_batch_size))
        for start in range(0, total_slices, batch_size):
            end = min(start + batch_size, total_slices)
            xx, yy, zz = np.meshgrid(axis, axis, axis[start:end], indexing="xy")
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
            inside = cls._points_inside_masks(points, mask1, mask2, cameras[0], cameras[1])
            if np.any(inside):
                kept_batches.append(points[inside])
            if progress_callback is not None:
                progress_callback(end, total_slices)

        kept = (
            np.concatenate(kept_batches, axis=0)
            if kept_batches else np.empty((0, 3), dtype=np.float32)
        )
        if len(kept) == 0:
            raise ValueError("Swept volume is empty. Check that both target images overlap in 3D.")

        grid_step = float(axis[1] - axis[0]) if len(axis) > 1 else float(half * 2.0)
        kept = _inflate_points(
            kept,
            grid_step=grid_step,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            steps=inflate_steps,
        )
        return cls(
            points=kept.astype(np.float32),
            grid_step=grid_step,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            masks=(mask1, mask2),
            cell_scores=(score1, score2),
            cameras=(cameras[0], cameras[1]),
        )

    @staticmethod
    def _points_inside_masks(
        points: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
        camera1: "Camera",
        camera2: "Camera",
    ) -> np.ndarray:
        x1, y1, valid1 = _project_points(points, camera1, mask1.shape)
        x2, y2, valid2 = _project_points(points, camera2, mask2.shape)
        inside = valid1 & valid2
        idx = np.where(inside)[0]
        if len(idx) == 0:
            return inside
        inside_idx = idx[mask1[y1[idx], x1[idx]] & mask2[y2[idx], x2[idx]]]
        result = np.zeros(len(points), dtype=bool)
        result[inside_idx] = True
        return result

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        in_bounds = np.all((pts >= self.bounds_min) & (pts <= self.bounds_max), axis=1)
        result = np.zeros(len(pts), dtype=bool)
        if np.any(in_bounds):
            result[in_bounds] = self._points_inside_masks(
                pts[in_bounds],
                self.masks[0],
                self.masks[1],
                self.cameras[0],
                self.cameras[1],
            )
        return result

    def score_points(self, points: np.ndarray) -> np.ndarray:
        """Return the weakest per-view silhouette-cell score for each 3D point."""
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        scores = np.full(len(pts), -100.0, dtype=np.float32)
        x1, y1, valid1 = _project_points(pts, self.cameras[0], self.cell_scores[0].shape)
        x2, y2, valid2 = _project_points(pts, self.cameras[1], self.cell_scores[1].shape)
        valid = valid1 & valid2
        idx = np.where(valid)[0]
        if len(idx) > 0:
            view1_scores = self.cell_scores[0][y1[idx], x1[idx]]
            view2_scores = self.cell_scores[1][y2[idx], x2[idx]]
            scores[idx] = np.minimum(view1_scores, view2_scores)
        return scores

    def sample_point(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if len(self.points) == 0:
            raise ValueError("Cannot sample from an empty swept volume.")
        rng = rng or np.random.default_rng()
        return self.sample_point_at_index(int(rng.integers(0, len(self.points))), rng)

    def sample_point_at_index(
        self,
        index: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if len(self.points) == 0:
            raise ValueError("Cannot sample from an empty swept volume.")
        rng = rng or np.random.default_rng()
        base = self.points[int(index) % len(self.points)].copy()
        jitter = rng.uniform(-0.45, 0.45, size=3).astype(np.float32) * self.grid_step
        return np.clip(base + jitter, self.bounds_min, self.bounds_max).astype(np.float32)

    def patch_entirely_outside(self, patch: "Patch", n_per_segment: int = 16) -> bool:
        pts = patch.sample_spline_world(n_per_segment).detach().cpu().numpy()
        return not bool(np.any(self.contains_points(pts)))

    def to_mesh(self, max_voxels: int = 240) -> "Mesh":
        from scene.scene import Mesh

        if len(self.points) == 0:
            return Mesh(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32))

        if len(self.points) > max_voxels:
            idx = np.linspace(0, len(self.points) - 1, max_voxels).astype(np.int32)
            points = self.points[idx]
        else:
            points = self.points

        s = self.grid_step * 0.32
        offsets = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ], dtype=np.float32)
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)

        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        for point in points:
            start = len(vertices) * 8
            vertices.append(point[None, :] + offsets)
            faces.append(cube_faces + start)

        return Mesh(
            np.concatenate(vertices, axis=0).astype(np.float32),
            np.concatenate(faces, axis=0).astype(np.int32),
            color=(0.35, 0.2, 0.75),
            label="swept_volume",
        )


def _inflate_points(
    points: np.ndarray,
    *,
    grid_step: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    steps: int,
) -> np.ndarray:
    if steps <= 0 or len(points) == 0:
        return points

    offsets = np.array([
        [dx, dy, dz]
        for dx in range(-steps, steps + 1)
        for dy in range(-steps, steps + 1)
        for dz in range(-steps, steps + 1)
    ], dtype=np.float32) * float(grid_step)
    expanded = (points[:, None, :] + offsets[None, :, :]).reshape((-1, 3))
    expanded = np.clip(expanded, bounds_min, bounds_max)
    quantized = np.rint((expanded - bounds_min) / grid_step).astype(np.int32)
    unique = np.unique(quantized, axis=0).astype(np.float32)
    inflated = bounds_min + unique * float(grid_step)
    return np.clip(inflated, bounds_min, bounds_max).astype(np.float32)


def _silhouette_cell_scores(mask: np.ndarray) -> np.ndarray:
    """Classify silhouette cells and store an approximate distance-to-edge score."""
    base_mask = np.asarray(mask, dtype=bool)
    if base_mask.ndim != 2:
        base_mask = base_mask.squeeze()
    h, w = base_mask.shape
    if h == 0 or w == 0:
        return np.empty_like(base_mask, dtype=np.float32)

    samples = np.stack([
        _sample_mask_cells(base_mask, 0.5, 0.5),
        _sample_mask_cells(base_mask, 0.25, 0.35),
        _sample_mask_cells(base_mask, 0.75, 0.65),
    ])
    sample_count = samples.sum(axis=0)
    inside = sample_count == samples.shape[0]
    border = (sample_count > 0) & (sample_count < samples.shape[0])

    padded = np.pad(inside, 1, mode="constant", constant_values=False)
    neighbors_inside = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    border |= inside & ~neighbors_inside
    inside_distance_region = inside & ~border

    inf = np.float32(max(h + w + 1, 1))
    dist = np.where(border, 0.0, inf).astype(np.float32)
    for y in range(1, h):
        dist[y, :] = np.minimum(dist[y, :], dist[y - 1, :] + 1.0)
    for y in range(h - 2, -1, -1):
        dist[y, :] = np.minimum(dist[y, :], dist[y + 1, :] + 1.0)
    for x in range(1, w):
        dist[:, x] = np.minimum(dist[:, x], dist[:, x - 1] + 1.0)
    for x in range(w - 2, -1, -1):
        dist[:, x] = np.minimum(dist[:, x], dist[:, x + 1] + 1.0)

    scores = np.full((h, w), -100.0, dtype=np.float32)
    scores[inside_distance_region] = dist[inside_distance_region]
    scores[border] = 0.0
    return scores


def _sample_mask_cells(mask: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
    """Bilinearly sample a binary cell mask at one deterministic in-cell offset."""
    h, w = mask.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = np.clip(xx + float(x_offset) - 0.5, 0.0, max(float(w - 1), 0.0))
    y = np.clip(yy + float(y_offset) - 0.5, 0.0, max(float(h - 1), 0.0))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    tx = x - x0
    ty = y - y0

    values = mask.astype(np.float32)
    top = values[y0, x0] * (1.0 - tx) + values[y0, x1] * tx
    bottom = values[y1, x0] * (1.0 - tx) + values[y1, x1] * tx
    return (top * (1.0 - ty) + bottom * ty) >= 0.5
