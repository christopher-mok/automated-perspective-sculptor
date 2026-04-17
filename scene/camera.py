"""Camera class representing a view camera in the scene (not the viewport orbit camera)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Camera:
    """A perspective camera positioned in the scene.

    Used to represent the two view cameras whose frustums are drawn in the
    3D viewport. These are distinct from the viewport's own orbit camera.
    """

    def __init__(
        self,
        position: np.ndarray | list,
        target: np.ndarray | list,
        up: np.ndarray | list | None = None,
        fov: float = 50.0,
        aspect: float = 4.0 / 3.0,
        near: float = 0.4,
        far: float = 7.0,
        color: Tuple[float, float, float] = (1.0, 0.85, 0.0),
        label: str = "",
    ) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.up = np.asarray(
            up if up is not None else [0.0, 1.0, 0.0], dtype=np.float32
        )
        self.fov = fov        # vertical FOV in degrees
        self.aspect = aspect
        self.near = near
        self.far = far
        self.color = color
        self.label = label

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        """Return the 4×4 row-major view (look-at) matrix."""
        f = self.target - self.position
        length = np.linalg.norm(f)
        if length < 1e-8:
            return np.eye(4, dtype=np.float32)
        f = f / length

        world_up = self.up.copy()
        if abs(np.dot(f, world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        r = np.cross(f, world_up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)

        return np.array(
            [
                [r[0],  r[1],  r[2],  -np.dot(r, self.position)],
                [u[0],  u[1],  u[2],  -np.dot(u, self.position)],
                [-f[0], -f[1], -f[2],  np.dot(f, self.position)],
                [0.0,   0.0,   0.0,   1.0],
            ],
            dtype=np.float32,
        )

    def projection_matrix(self) -> np.ndarray:
        """Return the 4×4 row-major perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        n, fa = self.near, self.far
        return np.array(
            [
                [f / self.aspect, 0.0, 0.0,                       0.0],
                [0.0,             f,   0.0,                       0.0],
                [0.0,             0.0, (fa + n) / (n - fa),  2*fa*n / (n - fa)],
                [0.0,             0.0, -1.0,                      0.0],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Geometry for frustum wireframe
    # ------------------------------------------------------------------

    def frustum_line_vertices(self) -> np.ndarray:
        """Return an (N, 3) float32 array of endpoint pairs for GL_LINES.

        Draws: near rectangle, far rectangle, 4 side edges, and 4 apex lines
        from the camera position to the near-plane corners.
        """
        f = self.target - self.position
        f = f / np.linalg.norm(f)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(f, world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        r = np.cross(f, world_up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)

        def rect_corners(dist: float) -> list[np.ndarray]:
            h = np.tan(np.radians(self.fov) / 2.0) * dist
            w = h * self.aspect
            c = self.position + f * dist
            return [
                c + u * h + r * w,  # top-right
                c + u * h - r * w,  # top-left
                c - u * h - r * w,  # bottom-left
                c - u * h + r * w,  # bottom-right
            ]

        near_c = rect_corners(self.near)
        far_c = rect_corners(self.far)

        pairs: list[np.ndarray] = []

        # Near rectangle
        for i in range(4):
            pairs += [near_c[i], near_c[(i + 1) % 4]]
        # Far rectangle
        for i in range(4):
            pairs += [far_c[i], far_c[(i + 1) % 4]]
        # Side edges connecting near ↔ far
        for i in range(4):
            pairs += [near_c[i], far_c[i]]
        # Apex lines: camera position → near corners
        for nc in near_c:
            pairs += [self.position.copy(), nc]

        return np.array(pairs, dtype=np.float32)
