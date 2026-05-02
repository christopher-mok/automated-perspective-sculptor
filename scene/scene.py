"""Scene graph managing meshes and cameras.

The Scene is purely a data container — it has no OpenGL dependency.
The viewport reads from it and handles all GPU-side rendering.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from scene.camera import Camera


class Mesh:
    """A triangle mesh to be rendered in the 3D viewport.

    Attributes:
        vertices:  (V, 3) float32 array of XYZ positions.
        faces:     (F, 3) int32 array of triangle indices into vertices.
        color:     (R, G, B) diffuse color in [0, 1].
        transform: 4×4 float32 model transform matrix (row-major).
        visible:   Whether to draw this mesh.
        label:     Human-readable name, useful for UI and debugging.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        transform: np.ndarray | None = None,
        label: str = "",
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.color = color
        self.transform: np.ndarray = (
            transform if transform is not None else np.eye(4, dtype=np.float32)
        )
        self.visible = True
        self.label = label

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) axis-aligned bounding box."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


class Scene:
    """Top-level scene container.

    Usage
    -----
    scene = Scene()
    scene.add_camera(cam1)
    scene.add_camera(cam2)
    mesh = Mesh(vertices, faces, color=(1, 0, 0))
    scene.add_mesh(mesh)
    # later…
    scene.remove_mesh(mesh)
    """

    def __init__(self) -> None:
        self.cameras: list[Camera] = []
        self.meshes: list[Mesh] = []

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def add_camera(self, camera: Camera) -> None:
        self.cameras.append(camera)

    def remove_camera(self, camera: Camera) -> None:
        self.cameras.remove(camera)

    # ------------------------------------------------------------------
    # Mesh management
    # ------------------------------------------------------------------

    def add_mesh(self, mesh: Mesh) -> None:
        """Add a mesh to the scene. The viewport will upload it to the GPU
        on the next paint cycle."""
        self.meshes.append(mesh)

    def remove_mesh(self, mesh: Mesh) -> None:
        """Remove a mesh. The viewport will free GPU resources automatically."""
        self.meshes.remove(mesh)

    def clear_meshes(self) -> None:
        """Remove all meshes from the scene."""
        self.meshes.clear()

    def set_meshes(self, meshes: Sequence[Mesh]) -> None:
        """Replace all meshes at once (e.g. after re-initializing patches)."""
        self.meshes = list(meshes)
