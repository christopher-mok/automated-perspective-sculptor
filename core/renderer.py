"""Differentiable renderer using nvdiffrast.

Pipeline
--------
For each patch:
  1. Sample the closed Bezier spline → world-space points
  2. Extrude the spline outline into a thin triangle mesh
  3. Assemble all patches into one batched vertex/triangle buffer
  4. Transform vertices to clip space via the camera MVP
  5. Rasterize with nvdiffrast
  6. Interpolate per-vertex albedo → flat-shaded colour per pixel
  7. Antialias patch edges (differentiable)
  8. Return (H, W, 4) RGBA — fully differentiable w.r.t. all patch parameters

nvdiffrast installation
-----------------------
  pip install setuptools wheel ninja
  pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

Backend selection
-----------------
CUDA available → RasterizeCudaContext  (fastest)
No CUDA        → RasterizeGLContext    (still requires the nvdiffrast CUDA
                 extension to be built at install time)

Platform note
-------------
Nvdiffrast's PyTorch package currently builds a CUDA extension during install.
That means macOS/Apple Silicon cannot install it directly because there is no
CUDA toolkit / CUDA_HOME. Use a Linux or Windows machine with an NVIDIA GPU for
the differentiable optimization path, or add a separate non-nvdiffrast renderer.

Threading note
--------------
nvdiffrast's GL context must be created and used on the *same* thread as the
one that initialises it.  Run the renderer inside the OptimizationWorker
(a QThread) and never call it from the main Qt thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from core.patch import Patch
    from scene.camera import Camera


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _create_context():
    """Return the best available nvdiffrast rasterisation context."""
    try:
        import nvdiffrast.torch as dr  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "nvdiffrast is required for differentiable rendering.\n"
            "Install on Linux/Windows with an NVIDIA GPU and CUDA toolkit:\n"
            "  pip install setuptools wheel ninja\n"
            "  pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation\n"
            "macOS/Apple Silicon cannot build the required CUDA extension."
        ) from exc

    if torch.cuda.is_available():
        return dr.RasterizeCudaContext()
    return dr.RasterizeGLContext()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _camera_mvp(camera: "Camera", device: str) -> torch.Tensor:
    """Return the (4, 4) row-major MVP = P @ V as a torch tensor.

    Our Camera matrices are row-major (standard math convention).
    To transform a world-space row-vector v: clip_row = v @ mvp.T
    """
    V = torch.from_numpy(camera.view_matrix()).float().to(device)
    P = torch.from_numpy(camera.projection_matrix()).float().to(device)
    return P @ V   # (4, 4)


# ---------------------------------------------------------------------------
# Geometry assembly
# ---------------------------------------------------------------------------


def _build_geometry(
    patches: list["Patch"],
    n_per_segment: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble all patch meshes into flat GPU buffers for nvdiffrast.

    Returns
    -------
    verts_world : (1, V_total, 4)  world-space homogeneous positions (w=1).
    tris        : (F_total, 3)     int32 triangle indices.
    colors      : (1, V_total, 3)  per-vertex albedo (flat per patch).
    """
    all_verts:  list[torch.Tensor] = []
    all_colors: list[torch.Tensor] = []
    all_tris:   list[torch.Tensor] = []
    v_offset = 0

    for patch in patches:
        verts_xyz, faces = patch.extruded_mesh_world(n_per_segment)
        n_verts = len(verts_xyz)
        ones = torch.ones(n_verts, 1, dtype=torch.float32, device=device)
        verts_h = torch.cat([verts_xyz, ones], dim=1)

        # Flat per-patch colour broadcast to every vertex
        albedo = patch.albedo.clamp(0.0, 1.0)                    # (3,)
        colors = albedo.unsqueeze(0).expand(n_verts, 3).clone()
        tris = faces + v_offset

        all_verts.append(verts_h)
        all_colors.append(colors)
        all_tris.append(tris)
        v_offset += n_verts

    verts_world = torch.cat(all_verts,  dim=0).unsqueeze(0)   # (1, V, 4)
    colors_all  = torch.cat(all_colors, dim=0).unsqueeze(0)   # (1, V, 3)
    tris_all    = torch.cat(all_tris,   dim=0)                 # (F, 3)

    return verts_world, tris_all, colors_all


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class DiffRenderer:
    """Differentiable patch renderer backed by nvdiffrast.

    The context is created lazily on first use so the object can be
    constructed before an OpenGL / CUDA context is available.

    Parameters
    ----------
    device       : PyTorch device string ("cpu", "mps", "cuda").
    n_per_segment: Bezier samples per spline segment.  Higher = smoother
                   outlines and finer gradients, but slower per step.
    """

    def __init__(self, device: str = "cpu", n_per_segment: int = 20) -> None:
        self.device        = device
        self.n_per_segment = n_per_segment
        self._ctx          = None   # lazy — created on first render() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_context(self) -> None:
        if self._ctx is None:
            self._ctx = _create_context()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        patches: list["Patch"],
        camera: "Camera",
        resolution: tuple[int, int],
    ) -> torch.Tensor:
        """Render all patches from one camera viewpoint.

        Args:
            patches:    List of Patch objects whose parameters require grad.
            camera:     Scene Camera (position, FOV, etc.).
            resolution: (height, width) of the output image in pixels.

        Returns:
            (H, W, 4) float32 RGBA tensor on self.device.
            Fully differentiable w.r.t. all patch parameters.
        """
        import nvdiffrast.torch as dr  # type: ignore[import]

        self._ensure_context()

        H, W = resolution

        if not patches:
            return torch.zeros(H, W, 4, device=self.device)

        # ---- Geometry --------------------------------------------------
        verts_world, tris, colors = _build_geometry(
            patches, self.n_per_segment, self.device
        )

        # ---- Clip-space transform: clip_row = world_row @ mvp.T --------
        mvp        = _camera_mvp(camera, self.device)   # (4, 4)
        verts_clip = verts_world @ mvp.T                # (1, V, 4)

        # ---- Rasterise -------------------------------------------------
        rast, _ = dr.rasterize(self._ctx, verts_clip, tris, (H, W))
        # rast: (1, H, W, 4)  — (u, v, z/w, triangle_id)

        # ---- Interpolate albedo ----------------------------------------
        color_out, _ = dr.interpolate(colors, rast, tris)   # (1, H, W, 3)

        # ---- Antialias -------------------------------------------------
        # Propagates gradients through silhouette edges
        color_out = dr.antialias(color_out, rast, verts_clip, tris)  # (1, H, W, 3)

        # ---- Compose RGBA ----------------------------------------------
        alpha = (rast[..., 3:4] > 0).float()              # (1, H, W, 1) — coverage mask
        alpha = dr.antialias(alpha, rast, verts_clip, tris).clamp(0.0, 1.0)
        color_out = color_out * alpha
        rgba  = torch.cat([color_out, alpha], dim=-1)      # (1, H, W, 4)

        return rgba.squeeze(0)   # (H, W, 4)

    def render_both(
        self,
        patches: list["Patch"],
        camera1: "Camera",
        camera2: "Camera",
        resolution: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render from both cameras in one call.

        Returns:
            (render1, render2) — each (H, W, 4) RGBA.
        """
        r1 = self.render(patches, camera1, resolution)
        r2 = self.render(patches, camera2, resolution)
        return r1, r2
