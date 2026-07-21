"""Patch-vs-patch overlap tests.

Two families live here:

``sphere``  The historical test: each patch is replaced by a world-space
            bounding sphere centred on ``patch.center``, and any two spheres
            that interpenetrate are penalised. Orientation is ignored
            entirely, so for the thin wide plates this project produces the
            sphere is a gross over-approximation -- two coplanar-ish panels
            standing side by side are "overlapping" long before their
            material could ever touch. The optimizer then shoves them apart,
            which costs packing density and therefore detail.

``planar``  Orientation-aware. Every patch is flat (local z is frozen to 0)
            and rotates only about Y, so a patch is a *vertical prism*: an
            oriented rectangle in XZ extruded along Y. That structure makes
            both an exact and a tight smooth test cheap:

            exact   Two patch planes intersect in a vertical line. Clip each
                    patch's sampled outline against that line, which yields a
                    set of Y intervals per patch; the patches interpenetrate
                    iff the two interval sets meet. Panel thickness turns the
                    line into a band, handled by clipping at three offsets.
                    Near-parallel planes (where the intersection line shoots
                    off to infinity) fall back to a slab-distance test.

            proxy   Separating-axis test between the two prisms. Because Y is
                    an axis of both, the 3D test splits exactly into a Y
                    interval test plus a 2D oriented-rectangle SAT in XZ, so
                    the minimum over five signed depths is the penetration
                    depth. Exact for rectangular outlines and tight for the
                    rest, and differentiable everywhere the min/abs are.

The exact test is the hard/repair criterion; the proxy carries the gradient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from core.patch import Patch

# Below this |sin(theta_i - theta_j)| the two supporting planes are treated as
# parallel: the intersection line is numerically meaningless and the thickness
# band width (thickness / |sin d|) would blow up.
_SIN_PARALLEL_EPS: float = 0.05

# Cap on the thickness band half-width so a glancing intersection cannot widen
# the clip band past the panel itself.
_MAX_BAND: float = 1.0

# Outline crossings kept per clip line. Patches are self-intersection-free and
# roughly blob-shaped, so 2 crossings is typical; 8 leaves room for concave
# outlines without making the pairwise interval comparison expensive.
_MAX_CROSSINGS: int = 8

_BIG: float = 1.0e6


# ---------------------------------------------------------------------------
# Shared geometry extraction
# ---------------------------------------------------------------------------


def _patch_geometry(
    patches: Sequence["Patch"],
    n_per_segment: int,
) -> dict[str, torch.Tensor]:
    """Stack per-patch pose and local outline samples.

    Returns local outline points rather than world ones because every planar
    test below works in a patch's own frame, where the outline is a plain 2D
    polygon and the shared world Y axis maps to local y untouched (the only
    rotation is about Y).
    """
    centers = torch.stack([patch.center for patch in patches], dim=0)      # (N, 3)
    thetas = torch.stack([patch.theta for patch in patches], dim=0)        # (N,)
    local = torch.stack(
        [patch.sample_spline_local(n_per_segment) for patch in patches],
        dim=0,
    )                                                                      # (N, M, 3)
    return {
        "centers": centers,
        "thetas": thetas,
        "x": local[..., 0],
        "y": local[..., 1],
        "cos": torch.cos(thetas),
        "sin": torch.sin(thetas),
    }


def _half_thickness(patches: Sequence["Patch"]) -> float:
    return float(patches[0].DEFAULT_THICKNESS) * 0.5


def _pair_indices(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.triu_indices(n, n, offset=1, device=device)
    return idx[0], idx[1]


# ---------------------------------------------------------------------------
# Legacy bounding-sphere test
# ---------------------------------------------------------------------------


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


def sphere_overlap_loss(
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


# ---------------------------------------------------------------------------
# Planar proxy: prism separating-axis test
# ---------------------------------------------------------------------------


def _prism_penetration(
    geom: dict[str, torch.Tensor],
    half_thickness: float,
    margin: float,
    ii: torch.Tensor,
    jj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed penetration depth per pair, and the per-axis depths behind it.

    Each patch is bounded by the oriented box that its outline's local
    axis-aligned extent sweeps: half extents (hx, hy, thickness/2) along
    (local x, world Y, plane normal). Y is an axis of both boxes, so the 15
    axis 3D SAT collapses to the Y interval plus four in-plane axes.

    Returns ``(penetration, depths)`` with shapes (P,) and (P, 5). Positive
    penetration means the boxes interpenetrate; its magnitude is the smallest
    translation that would separate them.
    """
    x, y = geom["x"], geom["y"]
    cos, sin = geom["cos"], geom["sin"]
    centers = geom["centers"]

    x_min, x_max = x.min(dim=1).values, x.max(dim=1).values
    y_min, y_max = y.min(dim=1).values, y.max(dim=1).values
    hx = (x_max - x_min) * 0.5
    hy = (y_max - y_min) * 0.5
    ox = (x_max + x_min) * 0.5
    oy = (y_max + y_min) * 0.5

    # Box centre in world space. rot_y maps local (ox, oy, 0) to
    # (ox*cos, oy, -ox*sin), so only the local-x offset moves the patch in XZ.
    box_x = centers[:, 0] + ox * cos
    box_y = centers[:, 1] + oy
    box_z = centers[:, 2] - ox * sin

    # In-plane axis and plane normal, both horizontal (no Y component).
    ax_x, ax_z = cos, -sin
    nm_x, nm_z = sin, cos

    dy = (hy[ii] + hy[jj] + margin) - (box_y[ii] - box_y[jj]).abs()
    dx = box_x[jj] - box_x[ii]
    dz = box_z[jj] - box_z[ii]

    def _axis_depth(a_x: torch.Tensor, a_z: torch.Tensor) -> torch.Tensor:
        radius_i = (
            hx[ii] * (a_x * ax_x[ii] + a_z * ax_z[ii]).abs()
            + half_thickness * (a_x * nm_x[ii] + a_z * nm_z[ii]).abs()
        )
        radius_j = (
            hx[jj] * (a_x * ax_x[jj] + a_z * ax_z[jj]).abs()
            + half_thickness * (a_x * nm_x[jj] + a_z * nm_z[jj]).abs()
        )
        return (radius_i + radius_j + margin) - (a_x * dx + a_z * dz).abs()

    depths = torch.stack([
        dy,
        _axis_depth(ax_x[ii], ax_z[ii]),
        _axis_depth(nm_x[ii], nm_z[ii]),
        _axis_depth(ax_x[jj], ax_z[jj]),
        _axis_depth(nm_x[jj], nm_z[jj]),
    ], dim=1)                                                              # (P, 5)

    return depths.min(dim=1).values, depths


def planar_overlap_loss(
    patches: Sequence["Patch"],
    margin: float = 0.005,
    n_per_segment: int = 20,
) -> torch.Tensor:
    """Differentiable penetration penalty between patches as vertical prisms.

    Squared penetration depth summed over pairs and divided by the patch
    count. The sphere test averages over *all* pairs, which works there only
    because its over-approximation leaves most pairs violating; under the
    planar test a handful of pairs genuinely touch and the rest score exactly
    zero, so the same mean would divide a real violation by ~n^2/2 and leave
    no usable gradient. Dividing by the patch count instead keeps the term
    intensive -- SRD compares losses across configurations with different
    patch counts, so a penalty that grew with n would bias it against
    additions -- while a genuine interpenetration still registers.

    Values are therefore not comparable across overlap modes; compare each
    mode against itself.
    """
    if len(patches) < 2:
        device = patches[0].center.device if patches else "cpu"
        return torch.zeros((), device=device)

    geom = _patch_geometry(patches, n_per_segment)
    ii, jj = _pair_indices(len(patches), geom["centers"].device)
    penetration, _ = _prism_penetration(
        geom, _half_thickness(patches), margin, ii, jj
    )
    return (torch.relu(penetration) ** 2).sum() / len(patches)


# ---------------------------------------------------------------------------
# Planar exact test: clip both outlines against the plane intersection line
# ---------------------------------------------------------------------------


def _clip_intervals(
    x: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip closed outlines against the vertical local line ``x = u``.

    ``x``/``y`` are (P, M) outline samples in one patch's local frame and ``u``
    is (P,). Returns ``(lo, hi, valid)`` of shape (P, _MAX_CROSSINGS // 2):
    the Y intervals where the patch's material crosses the line.
    """
    s0 = x - u.unsqueeze(1)
    s1 = torch.roll(s0, -1, dims=1)
    y0 = y
    y1 = torch.roll(y, -1, dims=1)

    # Half-open straddle rule: a vertex sitting exactly on the line is counted
    # by one of its two edges, never both, so the crossing count stays even.
    crosses = (s0 < 0) != (s1 < 0)
    denominator = s0 - s1
    t = s0 / torch.where(denominator.abs() < 1e-12, torch.full_like(denominator, 1e-12), denominator)
    y_cross = y0 + t.clamp(0.0, 1.0) * (y1 - y0)

    filled = torch.where(crosses, y_cross, torch.full_like(y_cross, _BIG))
    ordered, _ = torch.sort(filled, dim=1)
    count = crosses.sum(dim=1)

    n_pairs = _MAX_CROSSINGS // 2
    width = min(_MAX_CROSSINGS, ordered.shape[1])
    ordered = ordered[:, :width]
    if width < _MAX_CROSSINGS:
        pad = torch.full(
            (ordered.shape[0], _MAX_CROSSINGS - width),
            _BIG,
            device=ordered.device,
            dtype=ordered.dtype,
        )
        ordered = torch.cat([ordered, pad], dim=1)

    lo = ordered[:, 0::2][:, :n_pairs]
    hi = ordered[:, 1::2][:, :n_pairs]
    slot = torch.arange(n_pairs, device=x.device).unsqueeze(0)
    valid = (2 * slot + 1) < count.unsqueeze(1)
    return lo, hi, valid


def _intervals_for_band(
    x: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    band: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union of clip intervals across the near, centre and far band lines.

    The other patch's slab meets this patch's plane in a band of local-x
    half-width ``band``; sampling its two edges plus its centre captures the
    material inside it without a full polygon-vs-band clip.
    """
    parts = [_clip_intervals(x, y, u + offset * band) for offset in (-1.0, 0.0, 1.0)]
    lo = torch.cat([part[0] for part in parts], dim=1)
    hi = torch.cat([part[1] for part in parts], dim=1)
    valid = torch.cat([part[2] for part in parts], dim=1)
    return lo, hi, valid


def _interval_overlap_depth(
    lo_i: torch.Tensor, hi_i: torch.Tensor, valid_i: torch.Tensor,
    lo_j: torch.Tensor, hi_j: torch.Tensor, valid_j: torch.Tensor,
) -> torch.Tensor:
    """Deepest Y overlap between any interval of i and any interval of j."""
    depth = (
        torch.minimum(hi_i.unsqueeze(2), hi_j.unsqueeze(1))
        - torch.maximum(lo_i.unsqueeze(2), lo_j.unsqueeze(1))
    )
    both_valid = valid_i.unsqueeze(2) & valid_j.unsqueeze(1)
    depth = torch.where(both_valid, depth, torch.full_like(depth, -_BIG))
    return depth.flatten(1).max(dim=1).values


def exact_overlap_mask(
    patches: Sequence["Patch"],
    margin: float = 0.005,
    n_per_segment: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact interpenetration test for every patch pair.

    Returns ``(ii, jj, overlapping)``: the upper-triangular pair indices and a
    boolean mask. A pair counts as overlapping when their material comes within
    ``margin`` of touching.
    """
    n = len(patches)
    device = patches[0].center.device
    if n < 2:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, dtype=torch.bool, device=device)

    with torch.no_grad():
        geom = _patch_geometry(patches, n_per_segment)
        ii, jj = _pair_indices(n, device)
        half_thickness = _half_thickness(patches)

        centers, cos, sin = geom["centers"], geom["cos"], geom["sin"]
        x, y = geom["x"], geom["y"]

        sin_d = torch.sin(geom["thetas"][ii] - geom["thetas"][jj])
        crossing = sin_d.abs() > _SIN_PARALLEL_EPS
        safe_sin = torch.where(crossing, sin_d, torch.ones_like(sin_d))

        # --- Non-parallel planes: clip both outlines against their shared line.
        # Plane k is {p : (sin, cos) . (p.x, p.z) = d_k}; the 2x2 solve has
        # determinant sin(theta_i - theta_j), which is exactly `sin_d`.
        d = sin * centers[:, 0] + cos * centers[:, 2]
        q_x = (d[ii] * cos[jj] - cos[ii] * d[jj]) / safe_sin
        q_z = (sin[ii] * d[jj] - d[ii] * sin[jj]) / safe_sin

        band = (half_thickness / safe_sin.abs()).clamp(max=_MAX_BAND)

        def _local_u(k: torch.Tensor) -> torch.Tensor:
            return (
                (q_x - centers[k, 0]) * cos[k] - (q_z - centers[k, 2]) * sin[k]
            )

        # Clip in each patch's own frame but compare in world Y: rot_y leaves
        # the Y axis alone, so world y is just local y plus the patch centre.
        world_y = y + centers[:, 1].unsqueeze(1)
        lo_i, hi_i, valid_i = _intervals_for_band(x[ii], world_y[ii], _local_u(ii), band)
        lo_j, hi_j, valid_j = _intervals_for_band(x[jj], world_y[jj], _local_u(jj), band)
        line_depth = _interval_overlap_depth(
            lo_i, hi_i, valid_i, lo_j, hi_j, valid_j
        )
        crossing_overlap = crossing & (line_depth > -margin)

        # --- Near-parallel planes: the intersection line is meaningless, so
        # compare slab separation and then the in-plane extents. This is an
        # extent (not outline) test, but it still respects orientation, and it
        # only ever fires for pairs within half a degree of parallel.
        delta = centers[jj] - centers[ii]
        separation = (
            sin[ii] * delta[:, 0] + cos[ii] * delta[:, 2]
        ).abs()
        close = separation < (2.0 * half_thickness + margin)

        # Project patch j's outline into patch i's frame. Near-parallel means
        # local y is shared and local x differs only by the small angle.
        world_x = centers[jj, 0].unsqueeze(1) + x[jj] * cos[jj].unsqueeze(1)
        world_z = centers[jj, 2].unsqueeze(1) - x[jj] * sin[jj].unsqueeze(1)
        proj_x = (
            (world_x - centers[ii, 0].unsqueeze(1)) * cos[ii].unsqueeze(1)
            - (world_z - centers[ii, 2].unsqueeze(1)) * sin[ii].unsqueeze(1)
        )
        proj_y = centers[jj, 1].unsqueeze(1) + y[jj] - centers[ii, 1].unsqueeze(1)

        extent_overlap = (
            (proj_x.min(1).values - margin < x[ii].max(1).values)
            & (proj_x.max(1).values + margin > x[ii].min(1).values)
            & (proj_y.min(1).values - margin < y[ii].max(1).values)
            & (proj_y.max(1).values + margin > y[ii].min(1).values)
        )
        parallel_overlap = (~crossing) & close & extent_overlap

    return ii, jj, crossing_overlap | parallel_overlap


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def planar_overlap_repair(
    patches: Sequence["Patch"],
    margin: float = 0.005,
    n_per_segment: int = 20,
    max_shift: float = 0.02,
) -> tuple[int, float]:
    """Push genuinely interpenetrating patches apart along their shallowest axis.

    The exact test decides *whether* a pair interpenetrates; the SAT proxy
    supplies the cheapest direction to separate them along. Shifts for all
    pairs are accumulated and applied once, so the result does not depend on
    pair ordering, and each pair's contribution is capped at ``max_shift`` to
    keep a badly tangled configuration from teleporting patches across the
    scene in a single step.

    Returns ``(pairs_repaired, total_shift)``.
    """
    if len(patches) < 2:
        return 0, 0.0

    ii, jj, overlapping = exact_overlap_mask(patches, margin, n_per_segment)
    if not bool(overlapping.any()):
        return 0, 0.0

    with torch.no_grad():
        geom = _patch_geometry(patches, n_per_segment)
        ii_o, jj_o = ii[overlapping], jj[overlapping]
        _, depths = _prism_penetration(
            geom, _half_thickness(patches), margin, ii_o, jj_o
        )
        best_depth, best_axis = depths.min(dim=1)
        push = (best_depth.clamp(min=0.0) * 0.5 + margin).clamp(max=max_shift)

        cos, sin = geom["cos"], geom["sin"]
        # Axis 0 is world Y; axes 1..4 are the in-plane and normal directions
        # of patch i and patch j respectively.
        axis_x = torch.stack([
            torch.zeros_like(cos[ii_o]),
            cos[ii_o], sin[ii_o], cos[jj_o], sin[jj_o],
        ], dim=1)
        axis_y = torch.stack([
            torch.ones_like(cos[ii_o]),
            *[torch.zeros_like(cos[ii_o])] * 4,
        ], dim=1)
        axis_z = torch.stack([
            torch.zeros_like(cos[ii_o]),
            -sin[ii_o], cos[ii_o], -sin[jj_o], cos[jj_o],
        ], dim=1)

        rows = torch.arange(len(ii_o), device=best_axis.device)
        direction = torch.stack([
            axis_x[rows, best_axis],
            axis_y[rows, best_axis],
            axis_z[rows, best_axis],
        ], dim=1)

        # Point the axis from i toward j so the push always separates them.
        centers = geom["centers"]
        along = ((centers[jj_o] - centers[ii_o]) * direction).sum(dim=1)
        sign = torch.where(along >= 0, 1.0, -1.0)
        offset = direction * (push * sign).unsqueeze(1)

        shifts = torch.zeros_like(centers)
        shifts.index_add_(0, jj_o, offset * 0.5)
        shifts.index_add_(0, ii_o, -offset * 0.5)

        for index, patch in enumerate(patches):
            patch.center.data.add_(shifts[index])

        total_shift = float(shifts.norm(dim=1).sum().cpu())

    return int(overlapping.sum().cpu()), total_shift


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

OVERLAP_MODES = ("sphere", "planar")


def overlap_loss(
    patches: Sequence["Patch"],
    margin: float = 0.005,
    mode: str = "sphere",
    n_per_segment: int = 20,
) -> torch.Tensor:
    if mode == "planar":
        return planar_overlap_loss(patches, margin, n_per_segment)
    if mode == "sphere":
        return sphere_overlap_loss(patches, margin)
    raise ValueError(f"Unknown overlap mode {mode!r}; expected one of {OVERLAP_MODES}.")
