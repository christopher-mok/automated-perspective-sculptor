"""Patch initialization strategies.

Each function returns a list of Patch objects positioned in the XZ plane
(y=0) ready to be passed to SceneOptimizer.

Strategies
----------
init_grid   : Patches on a regular grid — fast, deterministic.
init_random : Patches at random positions — good for breaking symmetry.
init_sam    : Use Meta's Segment Anything Model to seed patch positions from
              a reference image (requires the ``segment-anything`` package).
"""

from __future__ import annotations

import math
import os

import numpy as np

from core.patch import ControlPoint, Patch


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_DEFAULT_BOUNDS: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
_DEFAULT_RADIUS: float = 0.18   # spline radius in local patch units
_DEFAULT_Y:      float = 0.0


# ---------------------------------------------------------------------------
# Spline-patch factory
# ---------------------------------------------------------------------------


def _make_patch(
    center: list[float],
    theta: float,
    radius: float = _DEFAULT_RADIUS,
    albedo: list[float] | None = None,
    device: str = "cpu",
    label: str = "",
) -> Patch:
    """Create a Patch whose spline outline approximates a regular pentagon.

    Control points are placed evenly around a circle of ``radius`` in the
    local XY plane.  Handles are set for a smooth circular approximation
    so the initial shape is a rounded closed curve.

    Args:
        center: [x, y, z] world-space centre of the patch.
        theta:  Y-axis rotation in radians.
        radius: Radius of the initial circle in local units.
        albedo: RGB colour in [0, 1].  Defaults to neutral grey.
        device: PyTorch device string.
        label:  Human-readable name for the patch.
    """
    if albedo is None:
        albedo = [0.5, 0.5, 0.5]

    n = Patch.N_CONTROL_POINTS
    handle_scale = radius * (4.0 / 3.0) * math.tan(math.pi / n)

    control_points: list[ControlPoint] = []
    for i in range(n):
        # Start at the top (−π/2) and go counter-clockwise
        angle = 2.0 * math.pi * i / n - math.pi / 2.0
        x_local = radius * math.cos(angle)
        y_local = radius * math.sin(angle)
        # Tangent direction is perpendicular to the radius (CCW)
        handle_rot = angle + math.pi / 2.0

        control_points.append(ControlPoint(
            x=x_local,
            y=y_local,
            z=0.0,
            handle_scale=handle_scale,
            handle_rotation=handle_rot,
            device=device,
        ))

    return Patch(
        control_points=control_points,
        center=center,
        theta=theta,
        albedo=albedo,
        device=device,
        label=label,
    )


# ---------------------------------------------------------------------------
# Grid initialization
# ---------------------------------------------------------------------------


def init_grid(
    n_patches: int,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    radius: float = _DEFAULT_RADIUS,
    y: float = _DEFAULT_Y,
    device: str = "cpu",
) -> list[Patch]:
    """Place patches on a uniform grid in the XZ plane.

    All patches start with theta = 0 (facing +Z) and a neutral grey spline
    approximating a circle of the given radius.

    Args:
        n_patches: Total number of patches to create.
        bounds:    (x_min, x_max, z_min, z_max) world-space extent.
        radius:    Initial spline radius in local units.
        y:         Y position of all patch centres.
        device:    PyTorch device string.
    """
    x_min, x_max, z_min, z_max = bounds
    cols = math.ceil(math.sqrt(n_patches))
    rows = math.ceil(n_patches / cols)

    xs = np.linspace(x_min, x_max, cols, dtype=np.float32)
    zs = np.linspace(z_min, z_max, rows, dtype=np.float32)

    patches: list[Patch] = []
    for row in range(rows):
        for col in range(cols):
            if len(patches) >= n_patches:
                break
            patches.append(_make_patch(
                center=[float(xs[col]), y, float(zs[row])],
                theta=0.0,
                radius=radius,
                device=device,
                label=f"patch_{len(patches):04d}",
            ))

    return patches


# ---------------------------------------------------------------------------
# Random initialization
# ---------------------------------------------------------------------------


def init_random(
    n_patches: int,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    radius: float = _DEFAULT_RADIUS,
    y: float = _DEFAULT_Y,
    theta_range: tuple[float, float] = (-math.pi / 4, math.pi / 4),
    seed: int | None = None,
    device: str = "cpu",
) -> list[Patch]:
    """Place patches at uniformly random positions within bounds.

    Args:
        n_patches:   Total number of patches to create.
        bounds:      (x_min, x_max, z_min, z_max) world-space extent.
        radius:      Initial spline radius in local units.
        y:           Y height of all patches.
        theta_range: (min, max) range for random Y-axis rotation in radians.
        seed:        Optional RNG seed for reproducibility.
        device:      PyTorch device string.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, z_min, z_max = bounds
    t_min, t_max = theta_range

    patches: list[Patch] = []
    for i in range(n_patches):
        x     = float(rng.uniform(x_min, x_max))
        z     = float(rng.uniform(z_min, z_max))
        theta = float(rng.uniform(t_min, t_max))
        patches.append(_make_patch(
            center=[x, y, z],
            theta=theta,
            radius=radius,
            device=device,
            label=f"patch_{i:04d}",
        ))

    return patches


# ---------------------------------------------------------------------------
# SAM-guided initialization
# ---------------------------------------------------------------------------

_SAM_VARIANTS: dict[str, tuple[str, str, str]] = {
    "MobileSAM (fast)":         ("mobile_sam",       "mobile_sam.pt",        "vit_t"),
    "SAM vit_b (balanced)":     ("segment_anything", "sam_vit_b_01ec64.pth", "vit_b"),
    "SAM vit_h (best quality)": ("segment_anything", "sam_vit_h_4b8939.pth", "vit_h"),
}

_SAM_INSTALL_HINTS: dict[str, str] = {
    "mobile_sam": (
        "pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
        "curl -L https://github.com/ChaoningZhang/MobileSAM/releases/download/v1.0/mobile_sam.pt"
        " -o mobile_sam.pt"
    ),
    "segment_anything": (
        "pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        " -o sam_vit_b_01ec64.pth"
    ),
}


def init_sam(
    image: np.ndarray,
    n_patches: int,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    radius: float = _DEFAULT_RADIUS,
    y: float = _DEFAULT_Y,
    sam_variant: str = "MobileSAM (fast)",
    device: str = "cpu",
) -> list[Patch]:
    """Initialize patches from SAM segmentation of a reference image.

    Each segment's centroid becomes a patch centre; the spline radius is
    scaled proportionally to the segment's area.  Albedo is seeded from
    the segment's mean colour.

    Args:
        image:       (H, W, 3) uint8 RGB numpy array.
        n_patches:   Desired total number of patches.
        bounds:      World-space XZ extent.
        radius:      Base spline radius (scaled per segment).
        y:           Y height of all patches.
        sam_variant: Key in _SAM_VARIANTS matching the UI label.
        device:      PyTorch device string.
    """
    if sam_variant not in _SAM_VARIANTS:
        raise ValueError(
            f"Unknown SAM variant {sam_variant!r}. "
            f"Choose from: {list(_SAM_VARIANTS)}"
        )

    package, checkpoint, model_type = _SAM_VARIANTS[sam_variant]

    try:
        if package == "mobile_sam":
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore[import]
        else:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            f"{sam_variant} requires the '{package}' package. Install with:\n"
            f"  {_SAM_INSTALL_HINTS[package]}"
        ) from exc

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint!r}\n"
            f"Download instructions:\n  {_SAM_INSTALL_HINTS[package]}"
        )

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=200,
    )
    masks = generator.generate(image)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:n_patches]

    x_min, x_max, z_min, z_max = bounds
    H, W = image.shape[:2]

    patches: list[Patch] = []
    for i, mask_data in enumerate(masks):
        bx, by, bw, bh = mask_data["bbox"]
        cx_n = (bx + bw / 2.0) / W
        cy_n = (by + bh / 2.0) / H

        wx = x_min + cx_n * (x_max - x_min)
        wz = z_min + cy_n * (z_max - z_min)

        # Scale radius proportionally to segment footprint
        footprint    = math.sqrt(mask_data["area"] / (H * W))
        patch_radius = float(np.clip(radius * footprint * 6.0, radius * 0.4, radius * 3.0))

        seg: np.ndarray = mask_data["segmentation"]
        mean_rgb = (image[seg].mean(axis=0) / 255.0).tolist() if seg.any() else [0.5, 0.5, 0.5]

        patches.append(_make_patch(
            center=[wx, y, wz],
            theta=0.0,
            radius=patch_radius,
            albedo=mean_rgb,
            device=device,
            label=f"patch_{i:04d}",
        ))

    # Pad with random patches if SAM found fewer than requested
    if len(patches) < n_patches:
        extra = init_random(
            n_patches - len(patches),
            bounds=bounds,
            radius=radius,
            y=y,
            device=device,
            seed=42,
        )
        for j, p in enumerate(extra):
            p.label = f"patch_{len(patches) + j:04d}"
        patches.extend(extra)

    return patches


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def initialize_patches(
    mode: str,
    n_patches: int,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
    radius: float = _DEFAULT_RADIUS,
    y: float = _DEFAULT_Y,
    device: str = "cpu",
    *,
    reference_image: np.ndarray | None = None,
    sam_variant: str = "MobileSAM (fast)",
    seed: int | None = None,
) -> list[Patch]:
    """Single entry-point called by the UI.

    Args:
        mode:            "Grid", "Random", or "SAM segmentation".
        n_patches:       Number of patches to create.
        bounds:          World-space XZ extent.
        radius:          Initial spline radius per patch.
        y:               Patch centre height.
        device:          PyTorch device string.
        reference_image: Required for SAM mode.
        sam_variant:     SAM model label.
        seed:            RNG seed for Random mode.
    """
    if mode == "Grid":
        return init_grid(n_patches, bounds, radius, y, device)

    if mode == "Random":
        return init_random(n_patches, bounds, radius, y, seed=seed, device=device)

    if mode == "SAM segmentation":
        if reference_image is None:
            raise ValueError(
                "SAM segmentation requires a reference image. "
                "Load a View 1 target image first."
            )
        return init_sam(reference_image, n_patches, bounds, radius, y,
                        sam_variant=sam_variant, device=device)

    raise ValueError(f"Unknown initialization mode: {mode!r}")
