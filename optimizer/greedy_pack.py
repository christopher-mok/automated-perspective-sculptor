"""Greedy swept-volume square-packing baseline.

An alternative to Stochastic Rewrite Descent (SRD) and gradient-based
continuous optimization: build a fixed candidate set of axis-facing square
panels seeded from the swept volume (the 3D region where a panel projects
inside both target silhouettes), then greedily select panels by weighted
marginal pixel-coverage gain minus a spill penalty, exactly the "greedy
set-cover" baseline described in the review comment this module answers.

Every candidate is two-view-valid by construction: it is a degenerate,
fixed-size, fixed-yaw square placed at a swept-volume position, so it is
guaranteed (subject to grid resolution) to fall inside both silhouettes at
its center. There is no gradient descent and no SRD rewrite grammar here --
panel geometry and position are frozen at candidate-construction time and the
only decision made is *which* candidates to keep.

This module is entirely self-contained: it is opt-in via ``run_greedy.py``
and ``GreedyPackConfig`` and is never imported by ``core/optimizer.py``,
``optimizer/srd.py`` or ``run_final.py``, so it cannot change the behaviour
of any existing arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.patch import ControlPoint, Patch
from core.swept_volume import SweptVolume

if TYPE_CHECKING:
    from scene.camera import Camera


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GreedyPackConfig:
    """Every knob the greedy baseline uses.

    Kept as one dataclass, constructed only by ``run_greedy.py``, so the
    method is reachable exclusively through an explicit opt-in rather than by
    touching any default used elsewhere.
    """

    # Candidate geometry: a square panel of this local half-width/height
    # (world units, same scale as Patch's spline radius), at fixed yaw.
    panel_half_size: float = 0.12
    # theta (Y-axis yaw, radians) every candidate is placed at. None picks
    # the bisector of the two camera yaw angles, which is the orientation
    # equally foreshortened -- and so equally "square-looking" -- in both
    # views for the standard 90-degree two-camera rig.
    theta: float | None = None
    # Candidate positions are the swept volume's own inflated point cloud,
    # optionally thinned to at most this many points (uniform stride) so the
    # candidate count stays tractable independent of --swept-resolution.
    max_candidates: int = 6000

    # Marginal-gain weights, one per view. The reviewer's "average weighting
    # between the two targets" is w1 = w2 = 0.5.
    view1_weight: float = 0.5
    view2_weight: float = 0.5
    # Cost per newly covered pixel that falls outside the target silhouette
    # in either view (lambda in the score below).
    spill_weight: float = 1.0

    # Stopping rules.
    min_gain: float = 1.0
    max_panels: int = 400

    # Degenerate-Bezier construction: the exported Patch objects use 5
    # control points (Patch requires exactly 5) tracing a square outline
    # with one edge midpoint added, and this near-zero handle length so the
    # spline reduces to straight edges -- "zero handles" without violating
    # the positive-handle-length invariant the rest of the codebase assumes.
    handle_scale: float = 1e-4


@dataclass
class GreedyPackResult:
    placements: list[dict]
    history: list[dict]
    coverage1: np.ndarray
    coverage2: np.ndarray
    metrics: dict[str, float]
    patches: list[Patch]
    n_candidates: int


# ---------------------------------------------------------------------------
# Candidate geometry
# ---------------------------------------------------------------------------

# Local corners of a unit square, in the consistent winding order used both
# for candidate rasterization and for the exported Patch outline.
_UNIT_CORNERS = np.array(
    [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]],
    dtype=np.float32,
)


def _bisector_theta(cameras: tuple["Camera", "Camera"]) -> float:
    """Yaw angle equidistant from both cameras' edge-on directions.

    Mirrors the yaw convention used elsewhere (theta is a Y-axis rotation;
    a camera's own "edge-on" yaw is ``atan2(offset_x, offset_z)`` of its
    position relative to its target). For the standard two-camera rig 90
    degrees apart this bisector is 45 degrees, the orientation that is
    equally foreshortened in both views.
    """
    angles = []
    for camera in cameras:
        offset = camera.position - camera.target
        angles.append(math.atan2(float(offset[0]), float(offset[2])))
    return 0.5 * (angles[0] + angles[1])


def _square_world_corners(
    center: np.ndarray,
    theta: float,
    half_size: float,
) -> np.ndarray:
    """World-space (4, 3) corners of a flat square panel.

    Local space has the panel in the XY plane (Patch convention: X
    horizontal, Y vertical, Z = 0); a Y-axis rotation by ``theta`` maps
    local (x, y, 0) to world (x*cos - 0, y, -x*sin) + center.
    """
    c, s = math.cos(theta), math.sin(theta)
    local = _UNIT_CORNERS * half_size
    x_local = local[:, 0]
    y_local = local[:, 1]
    world = np.stack(
        [c * x_local, y_local, -s * x_local],
        axis=1,
    ).astype(np.float32)
    return world + np.asarray(center, dtype=np.float32)[None, :]


def _project_points_float(
    points: np.ndarray,
    camera: "Camera",
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points to sub-pixel (px, py) float coordinates.

    Returns ``(pixels, valid)`` where ``pixels`` is (N, 2) float32 and
    ``valid`` marks points in front of the camera and inside the NDC cube --
    the same visibility test ``core.swept_volume._project_points`` uses, but
    without rounding to integer pixels, since candidate footprints need
    sub-pixel corners to rasterize a filled quad rather than sample a point.
    """
    h, w = resolution
    pts = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    homog = np.c_[pts, np.ones(len(pts), dtype=np.float32)]
    mvp = camera.projection_matrix() @ camera.view_matrix()
    clip = homog @ mvp.T
    clip_w = clip[:, 3]
    valid = clip_w > 1e-6
    ndc = np.zeros((len(pts), 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / clip_w[valid, None]

    px = (ndc[:, 0] * 0.5 + 0.5) * (w - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)
    valid &= (
        (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
        & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    )
    return np.stack([px, py], axis=1), valid


def _rasterize_convex_quad(
    corners_px: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray | None:
    """Flat pixel indices covered by a filled convex quad, or None if empty.

    Pixel centers are tested at integer + 0.5 offsets against the quad's four
    edges (consistent winding, so "inside" is one consistent cross-product
    sign, with an epsilon so boundary pixels are included rather than
    dropped by floating-point noise).
    """
    h, w = resolution
    x_min = max(0, int(math.floor(corners_px[:, 0].min())))
    x_max = min(w - 1, int(math.ceil(corners_px[:, 0].max())))
    y_min = max(0, int(math.floor(corners_px[:, 1].min())))
    y_max = min(h - 1, int(math.ceil(corners_px[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return None

    ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    px = xs.astype(np.float32) + 0.5
    py = ys.astype(np.float32) + 0.5

    n = len(corners_px)
    # Polygon winding from the corners themselves (shoelace sign), so the
    # inside test below does not depend on how pixels happen to be
    # distributed in the bounding box.
    shoelace = sum(
        corners_px[i, 0] * corners_px[(i + 1) % n, 1]
        - corners_px[(i + 1) % n, 0] * corners_px[i, 1]
        for i in range(n)
    )
    sign = 1.0 if shoelace >= 0 else -1.0

    inside = np.ones(px.shape, dtype=bool)
    eps = 1e-4
    for i in range(n):
        ax, ay = corners_px[i]
        bx, by = corners_px[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside &= cross * sign >= -eps

    if not np.any(inside):
        return None
    rows = ys[inside]
    cols = xs[inside]
    return (rows * w + cols).astype(np.int32)


# ---------------------------------------------------------------------------
# Candidate set construction
# ---------------------------------------------------------------------------


def build_candidates(
    swept_volume: SweptVolume,
    cameras: tuple["Camera", "Camera"],
    resolution: tuple[int, int],
    config: GreedyPackConfig,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float]:
    """Build the two-view-valid candidate footprints.

    Returns ``(centers, footprints1, footprints2, theta)``: parallel lists,
    one entry per surviving candidate (positions whose square footprint is
    non-empty and fully in-frame in both views), plus the fixed yaw used.
    """
    points = swept_volume.points
    if len(points) > config.max_candidates:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(points), size=config.max_candidates, replace=False)
        points = points[idx]

    theta = config.theta if config.theta is not None else _bisector_theta(cameras)

    centers: list[np.ndarray] = []
    footprints1: list[np.ndarray] = []
    footprints2: list[np.ndarray] = []

    for center in points:
        corners_world = _square_world_corners(center, theta, config.panel_half_size)
        px1, valid1 = _project_points_float(corners_world, cameras[0], resolution)
        px2, valid2 = _project_points_float(corners_world, cameras[1], resolution)
        if not (bool(np.all(valid1)) and bool(np.all(valid2))):
            # Not two-view-valid (a corner falls out of frame in some view) --
            # discard rather than clip, so every surviving candidate really is
            # guaranteed valid in both views by construction.
            continue
        footprint1 = _rasterize_convex_quad(px1, resolution)
        footprint2 = _rasterize_convex_quad(px2, resolution)
        if footprint1 is None or footprint2 is None:
            continue
        centers.append(np.asarray(center, dtype=np.float32))
        footprints1.append(footprint1)
        footprints2.append(footprint2)

    return centers, footprints1, footprints2, theta


# ---------------------------------------------------------------------------
# Greedy set cover
# ---------------------------------------------------------------------------


def _pad_footprints(footprints: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged index arrays into one (N, max_len) int64 array plus a
    parallel validity mask, so per-round scoring is one vectorized gather
    instead of a Python loop over candidates."""
    max_len = max((len(f) for f in footprints), default=0)
    n = len(footprints)
    padded = np.zeros((n, max_len), dtype=np.int32)
    valid = np.zeros((n, max_len), dtype=bool)
    for i, footprint in enumerate(footprints):
        padded[i, :len(footprint)] = footprint
        valid[i, :len(footprint)] = True
    return padded, valid


def greedy_select(
    footprints1: list[np.ndarray],
    footprints2: list[np.ndarray],
    target1_flat: np.ndarray,
    target2_flat: np.ndarray,
    config: GreedyPackConfig,
) -> tuple[list[int], list[dict], np.ndarray, np.ndarray]:
    """Run the greedy marginal-gain set-cover loop.

    ``score = w1 * new_view1_target_pixels + w2 * new_view2_target_pixels
             - lambda * new_spill_pixels``

    computed against the *currently uncovered* target masks each round, so a
    candidate's score only reflects pixels no already-placed panel has
    already claimed. Stops when the best remaining marginal score drops
    below ``config.min_gain`` or ``config.max_panels`` panels are placed.
    """
    n = len(footprints1)
    padded1, valid1 = _pad_footprints(footprints1)
    padded2, valid2 = _pad_footprints(footprints2)

    covered1 = np.zeros(target1_flat.shape, dtype=bool)
    covered2 = np.zeros(target2_flat.shape, dtype=bool)
    available = np.ones(n, dtype=bool)

    selected: list[int] = []
    history: list[dict] = []

    for round_index in range(config.max_panels):
        if not np.any(available):
            break

        is_target1 = target1_flat[padded1] & valid1
        new1 = is_target1 & ~covered1[padded1]
        spill1 = valid1 & ~is_target1 & ~covered1[padded1]
        gain1 = new1.sum(axis=1)
        cost1 = spill1.sum(axis=1)

        is_target2 = target2_flat[padded2] & valid2
        new2 = is_target2 & ~covered2[padded2]
        spill2 = valid2 & ~is_target2 & ~covered2[padded2]
        gain2 = new2.sum(axis=1)
        cost2 = spill2.sum(axis=1)

        scores = (
            config.view1_weight * gain1 + config.view2_weight * gain2
            - config.spill_weight * (cost1 + cost2)
        ).astype(np.float64)
        scores[~available] = -np.inf

        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if not np.isfinite(best_score) or best_score < config.min_gain:
            break

        covered1[padded1[best][valid1[best]]] = True
        covered2[padded2[best][valid2[best]]] = True
        available[best] = False
        selected.append(best)

        history.append({
            "round": round_index,
            "candidate": best,
            "score": best_score,
            "new_view1": int(gain1[best]),
            "new_view2": int(gain2[best]),
            "spill_view1": int(cost1[best]),
            "spill_view2": int(cost2[best]),
            "covered_view1": int(covered1.sum()),
            "covered_view2": int(covered2.sum()),
            "panels": len(selected),
        })

    return selected, history, covered1, covered2


# ---------------------------------------------------------------------------
# Metrics (matches core.optimizer.SceneOptimizer._silhouette_stats exactly,
# recomputed here from boolean coverage masks so the greedy arm's numbers are
# comparable to every gradient/SRD arm without importing torch).
# ---------------------------------------------------------------------------


def _silhouette_stats(render_binary: np.ndarray, target_binary: np.ndarray) -> dict[str, float]:
    intersection = float((render_binary & target_binary).sum())
    union = float((render_binary | target_binary).sum())
    render_area = float(render_binary.sum())
    target_area = float(target_binary.sum())
    false_positive = render_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "coverage": intersection / target_area if target_area else 0.0,
        "precision": intersection / render_area if render_area else 0.0,
        "spill": false_positive / target_area if target_area else 0.0,
    }


def _compute_metrics(
    coverage1: np.ndarray,
    coverage2: np.ndarray,
    target1: np.ndarray,
    target2: np.ndarray,
) -> dict[str, float]:
    stats1 = _silhouette_stats(coverage1, target1)
    stats2 = _silhouette_stats(coverage2, target2)
    metrics = {f"view1_{k}": v for k, v in stats1.items()}
    metrics.update({f"view2_{k}": v for k, v in stats2.items()})
    for key in ("iou", "coverage", "precision", "spill"):
        metrics[f"mean_{key}"] = 0.5 * (stats1[key] + stats2[key])
    return metrics


# ---------------------------------------------------------------------------
# Patch export
# ---------------------------------------------------------------------------


def _square_patch(
    center: np.ndarray,
    theta: float,
    half_size: float,
    handle_scale: float,
    device: str = "cpu",
    label: str = "",
    albedo: tuple[float, float, float] = (0.6, 0.7, 0.9),
) -> Patch:
    """A near-zero-handle degenerate Patch tracing a square outline.

    Patch requires exactly 5 control points, so the square's 4 corners are
    traced plus one extra point splitting the last edge at its midpoint;
    with ``handle_scale`` near zero every segment reduces to a straight
    line, so the rendered outline is indistinguishable from a true square.
    """
    s = half_size
    local_points = np.array(
        [[s, s], [-s, s], [-s, -s], [0.0, -s], [s, -s]],
        dtype=np.float32,
    )
    control_points: list[ControlPoint] = []
    n = len(local_points)
    for i in range(n):
        prev_pt = local_points[(i - 1) % n]
        next_pt = local_points[(i + 1) % n]
        tangent = next_pt - prev_pt
        angle = float(math.atan2(tangent[1], tangent[0]))
        control_points.append(ControlPoint(
            x=float(local_points[i, 0]),
            y=float(local_points[i, 1]),
            z=0.0,
            handle_scale=handle_scale,
            handle_rotation=angle,
            device=device,
        ))
    return Patch(
        control_points=control_points,
        center=center.tolist(),
        theta=theta,
        albedo=list(albedo),
        device=device,
        label=label,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_greedy_pack(
    swept_volume: SweptVolume,
    cameras: tuple["Camera", "Camera"],
    target1_mask: np.ndarray,
    target2_mask: np.ndarray,
    config: GreedyPackConfig,
    device: str = "cpu",
) -> GreedyPackResult:
    """Build candidates from the swept volume and greedily pack them.

    ``target1_mask``/``target2_mask`` are (H, W) boolean foreground masks at
    the resolution the candidates were rasterized at.
    """
    resolution = target1_mask.shape[:2]
    centers, footprints1, footprints2, theta = build_candidates(
        swept_volume, cameras, resolution, config,
    )

    target1_flat = target1_mask.reshape(-1)
    target2_flat = target2_mask.reshape(-1)

    selected, history, coverage1, coverage2 = greedy_select(
        footprints1, footprints2, target1_flat, target2_flat, config,
    )

    metrics = _compute_metrics(coverage1.reshape(resolution), coverage2.reshape(resolution),
                                target1_mask, target2_mask)
    metrics["patches"] = float(len(selected))
    metrics["candidates"] = float(len(centers))

    placements: list[dict] = []
    patches: list[Patch] = []
    for order, candidate_index in enumerate(selected):
        center = centers[candidate_index]
        placements.append({
            "order": order,
            "candidate_index": candidate_index,
            "center": center.tolist(),
        })
        patches.append(_square_patch(
            center, theta, config.panel_half_size, config.handle_scale,
            device=device, label=f"greedy_{order}",
        ))

    return GreedyPackResult(
        placements=placements,
        history=history,
        coverage1=coverage1.reshape(resolution),
        coverage2=coverage2.reshape(resolution),
        metrics=metrics,
        patches=patches,
        n_candidates=len(centers),
    )
