"""Spline-based patch primitive — the basic unit of the anamorphic sculpture.

Each Patch is a flat laser-cut piece whose outline is a closed cubic Bezier
spline with exactly 5 control points.  The piece stands upright and is
oriented by a single Y-axis rotation (theta).

Coordinate convention
---------------------
Local space  : control points live in the XY plane (z = 0).
               X = horizontal extent of the piece.
               Y = vertical extent (up).
After rot_y(theta) + translation to center → world space.

Bezier handles
--------------
Each control point has symmetric smooth handles (G1 continuity):
    handle_out = handle_scale * [cos(handle_rotation), sin(handle_rotation), 0]
    handle_in  = -handle_out

This matches Illustrator-style "smooth" nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from scene.scene import Mesh


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------


def rot_y(theta: torch.Tensor) -> torch.Tensor:
    """3×3 Y-axis rotation matrix. Fully differentiable."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack([
        torch.stack([ c, z, s]),
        torch.stack([ z, o, z]),
        torch.stack([-s, z, c]),
    ])  # (3, 3)


# ---------------------------------------------------------------------------
# Control point
# ---------------------------------------------------------------------------


class ControlPoint(nn.Module):
    """One node of the patch's closed Bezier spline.

    Parameters
    ----------
    x, y, z         : Position in local patch space.  z is initialised to 0
                       (flat piece) but is left learnable.
    handle_scale     : Length of both tangent handles (enforced positive).
    handle_rotation  : Direction angle of the outgoing handle in the XY plane.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        handle_scale: float = 0.15,
        handle_rotation: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        # Linked by Patch after construction. These are plain references, not
        # registered submodules, so the ModuleList below remains the owner.
        object.__setattr__(self, "next_control_point", None)
        object.__setattr__(self, "prev_control_point", None)
        self.x               = nn.Parameter(torch.tensor(x,               dtype=torch.float32, device=device))
        self.y               = nn.Parameter(torch.tensor(y,               dtype=torch.float32, device=device))
        self.z               = nn.Parameter(torch.tensor(z,               dtype=torch.float32, device=device))
        self.handle_scale    = nn.Parameter(torch.tensor(handle_scale,    dtype=torch.float32, device=device))
        self.handle_rotation = nn.Parameter(torch.tensor(handle_rotation, dtype=torch.float32, device=device))

    @property
    def pos(self) -> torch.Tensor:
        """(3,) position tensor."""
        return torch.stack([self.x, self.y, self.z])

    def handle_out(self) -> torch.Tensor:
        """(3,) outgoing tangent handle in local patch XY plane."""
        hs = self.handle_scale.abs()   # positive length
        return hs * torch.stack([
            torch.cos(self.handle_rotation),
            torch.sin(self.handle_rotation),
            torch.zeros_like(hs),
        ])

    def handle_in(self) -> torch.Tensor:
        """(3,) incoming tangent handle — symmetric opposite of handle_out."""
        return -self.handle_out()


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


class Patch(nn.Module):
    """A laser-cut piece defined by a closed cubic Bezier spline.

    The spline has N_CONTROL_POINTS = 5 nodes in local XY space.
    The whole piece is oriented by theta (Y-axis rotation) and placed at center.

    Learnable parameters
    --------------------
    center          : (3,) world-space position.
    theta           : scalar Y-axis rotation in radians.
    albedo          : (3,) RGB surface colour in [0, 1].
    control_points  : 5 × ControlPoint  (each has x, y, z, handle_scale, handle_rotation)
    """

    N_CONTROL_POINTS: int = 5
    DEFAULT_THICKNESS: float = 0.035

    def __init__(
        self,
        control_points: list[ControlPoint],
        center: list | np.ndarray,
        theta: float,
        albedo: list | tuple | np.ndarray = (0.6, 0.7, 0.9),
        device: str = "cpu",
        label: str = "",
    ) -> None:
        super().__init__()
        if len(control_points) != self.N_CONTROL_POINTS:
            raise ValueError(f"Patch requires exactly {self.N_CONTROL_POINTS} control points, got {len(control_points)}")

        self.control_points = nn.ModuleList(control_points)
        self._link_control_points()
        self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32, device=device))
        self.theta  = nn.Parameter(torch.tensor(theta,  dtype=torch.float32, device=device))
        self.albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float32, device=device))
        self.label  = label

    def _link_control_points(self) -> None:
        """Attach next/previous links for the closed control-point chain."""
        n = len(self.control_points)
        for i, cp in enumerate(self.control_points):
            object.__setattr__(cp, "prev_control_point", self.control_points[(i - 1) % n])
            object.__setattr__(cp, "next_control_point", self.control_points[(i + 1) % n])

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def rotation_matrix(self) -> torch.Tensor:
        """(3, 3) Y-axis rotation matrix."""
        return rot_y(self.theta)

    def local_to_world(self, local_pts: torch.Tensor) -> torch.Tensor:
        """Map (N, 3) local-space points to world space."""
        R = self.rotation_matrix()
        return (R @ local_pts.T).T + self.center.unsqueeze(0)

    # ------------------------------------------------------------------
    # Spline sampling (differentiable)
    # ------------------------------------------------------------------

    def sample_spline_local(self, n_per_segment: int = 20) -> torch.Tensor:
        """Sample the closed Bezier spline in local space.

        Returns (N_CONTROL_POINTS * n_per_segment, 3).
        The last sample of each segment is dropped to avoid duplicate
        points at segment joins (the closed loop reconnects automatically).
        """
        device = self.center.device
        t = torch.linspace(0.0, 1.0, n_per_segment + 1, device=device)[:-1]  # (T,)

        segments: list[torch.Tensor] = []
        n = self.N_CONTROL_POINTS
        for i in range(n):
            cp0 = cast(ControlPoint, self.control_points[i])
            cp1 = cast(ControlPoint, self.control_points[(i + 1) % n])

            P0 = cp0.pos
            P1 = P0 + cp0.handle_out()
            P2 = cp1.pos + cp1.handle_in()
            P3 = cp1.pos

            # Cubic Bezier: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
            t_ = t.unsqueeze(1)          # (T, 1)
            pts = (
                (1 - t_) ** 3           * P0 +
                3 * (1 - t_) ** 2 * t_  * P1 +
                3 * (1 - t_)      * t_**2 * P2 +
                t_ ** 3                 * P3
            )  # (T, 3)
            segments.append(pts)

        return torch.cat(segments, dim=0)   # (N*T, 3)

    def sample_spline_world(self, n_per_segment: int = 20) -> torch.Tensor:
        """Spline points in world space. Returns (N*T, 3)."""
        return self.local_to_world(self.sample_spline_local(n_per_segment))

    # ------------------------------------------------------------------
    # Mesh for homogeneous coordinate assembly (renderer)
    # ------------------------------------------------------------------

    def world_vertices_homogeneous(self, n_per_segment: int = 20) -> torch.Tensor:
        """(V, 4) world-space spline points with homogeneous w=1, for nvdiffrast."""
        pts  = self.sample_spline_world(n_per_segment)  # (V, 3)
        ones = torch.ones(len(pts), 1, dtype=pts.dtype, device=pts.device)
        return torch.cat([pts, ones], dim=1)             # (V, 4)

    def extruded_mesh_world(
        self,
        n_per_segment: int = 20,
        thickness: float = DEFAULT_THICKNESS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a fixed-thickness extrusion of the spline as world vertices/faces."""
        local = self.sample_spline_local(n_per_segment)
        N = len(local)
        half = thickness * 0.5
        offset = torch.tensor([0.0, 0.0, half], dtype=local.dtype, device=local.device)

        front = self.local_to_world(local + offset)
        back = self.local_to_world(local - offset)
        front_centroid = front.mean(dim=0, keepdim=True)
        back_centroid = back.mean(dim=0, keepdim=True)
        verts = torch.cat([front_centroid, front, back_centroid, back], dim=0)

        back_center = N + 1
        back_start = N + 2
        faces: list[list[int]] = []
        for i in range(N):
            j = (i + 1) % N
            fi = i + 1
            fj = j + 1
            bi = back_start + i
            bj = back_start + j
            faces.append([0, fi, fj])
            faces.append([back_center, bj, bi])
            faces.append([fi, bi, bj])
            faces.append([fi, bj, fj])

        return verts, torch.tensor(faces, dtype=torch.int32, device=self.center.device)

    def triangle_faces(self, n_per_segment: int = 20) -> torch.Tensor:
        """(F, 3) int32 triangle indices for a fan mesh from the centroid.

        Vertex layout expected by the renderer:
            index 0      : centroid  (prepend before calling)
            indices 1..V : spline sample points
        """
        V = self.N_CONTROL_POINTS * n_per_segment
        faces = torch.tensor(
            [[0, i + 1, (i + 1) % V + 1] for i in range(V)],
            dtype=torch.int32,
            device=self.center.device,
        )
        return faces  # (V, 3)

    # ------------------------------------------------------------------
    # Viewport bridge — detached numpy, safe across threads
    # ------------------------------------------------------------------

    def to_mesh(self, n_per_segment: int = 20) -> "Mesh":
        """Extruded scene.Mesh for the 3D viewport."""
        from scene.scene import Mesh

        with torch.no_grad():
            verts, faces = self.extruded_mesh_world(n_per_segment)
            vertices = verts.cpu().numpy()
            faces_np = faces.cpu().numpy()

        rgb   = self.albedo.detach().cpu().clamp(0.0, 1.0).numpy()
        color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        return Mesh(vertices, faces_np, color=color, label=self.label)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def clamp_albedo(self) -> None:
        with torch.no_grad():
            self.albedo.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _f(t: torch.Tensor) -> float:
            return float(t.detach().cpu())
        def _l(t: torch.Tensor) -> list:
            return t.detach().cpu().numpy().tolist()

        return {
            "label":  self.label,
            "center": _l(self.center),
            "theta":  _f(self.theta),
            "albedo": _l(self.albedo),
            "control_points": [
                {
                    "x":               _f(cast(ControlPoint, cp).x),
                    "y":               _f(cast(ControlPoint, cp).y),
                    "z":               _f(cast(ControlPoint, cp).z),
                    "handle_scale":    _f(cast(ControlPoint, cp).handle_scale),
                    "handle_rotation": _f(cast(ControlPoint, cp).handle_rotation),
                }
                for cp in self.control_points
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, device: str = "cpu") -> "Patch":
        cps = [
            ControlPoint(
                x=cp["x"], y=cp["y"], z=cp["z"],
                handle_scale=cp["handle_scale"],
                handle_rotation=cp["handle_rotation"],
                device=device,
            )
            for cp in d["control_points"]
        ]
        return cls(
            control_points=cps,
            center=d["center"],
            theta=float(d["theta"]),
            albedo=d["albedo"],
            device=device,
            label=d.get("label", ""),
        )

    def __repr__(self) -> str:
        c = self.center.detach().cpu().numpy().round(3).tolist()
        return f"Patch(label={self.label!r}, center={c}, theta={float(self.theta.detach()):.3f} rad)"
