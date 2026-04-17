"""Interactive 3D OpenGL viewport with orbit camera controls.

Uses OpenGL 3.3 Core Profile so it works on macOS and modern Linux/Windows.
All geometry is rendered via VAO/VBO pairs with a simple line shader.
Mesh faces are rendered with a flat-shaded triangle shader.

Orbit controls
--------------
- Left drag    → orbit (azimuth / elevation)
- Middle drag  → pan (translate target point)
- Scroll wheel → zoom (change distance)
- Momentum applied when left button released
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_NO_ERROR,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetError,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glViewport,
    GL_COMPILE_STATUS,
    GL_LINK_STATUS,
    glAttachShader,
    glGetProgramiv,
    glGetProgramInfoLog,
    glVertexAttribPointer,
)

if TYPE_CHECKING:
    from scene.scene import Mesh, Scene

# ---------------------------------------------------------------------------
# GLSL shaders
# ---------------------------------------------------------------------------

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(uColor, 1.0);
}
"""

# Flat-shaded triangle shader (for future mesh rendering)
_VERT_MESH_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
out vec3 vWorldPos;
void main() {
    vWorldPos = aPos;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

_FRAG_MESH_SRC = """
#version 330 core
in vec3 vWorldPos;
uniform vec3 uColor;
out vec4 fragColor;
void main() {
    // Simple flat shade with a fixed directional light
    vec3 light = normalize(vec3(0.5, 1.0, 0.8));
    vec3 dx = dFdx(vWorldPos);
    vec3 dy = dFdy(vWorldPos);
    vec3 normal = normalize(cross(dx, dy));
    float diff = max(dot(normal, light), 0.0) * 0.7 + 0.3;
    fragColor = vec4(uColor * diff, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Low-level GPU helpers
# ---------------------------------------------------------------------------


def _compile_program(vert_src: str, frag_src: str) -> int:
    def _compile_shader(src: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not ok:
            log = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compile error:\n{log}")
        return shader

    vert = _compile_shader(vert_src, GL_VERTEX_SHADER)
    frag = _compile_shader(frag_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vert)
    glAttachShader(program, frag)
    glLinkProgram(program)
    ok = glGetProgramiv(program, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader link error:\n{log}")
    return program


class _GPULines:
    """A VAO/VBO pair for rendering a static set of GL_LINES."""

    def __init__(self) -> None:
        self.vao: int = 0
        self.vbo: int = 0
        self.count: int = 0

    def upload(self, vertices: np.ndarray) -> None:
        """Upload (N, 3) float32 vertex pairs. Call after GL context is current."""
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        if self.vao == 0:
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
        self.count = len(data)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def draw(self) -> None:
        if self.vao and self.count:
            glBindVertexArray(self.vao)
            glDrawArrays(GL_LINES, 0, self.count)
            glBindVertexArray(0)

    def cleanup(self) -> None:
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            self.vao = self.vbo = 0


class _GPUMesh:
    """A VAO/VBO pair for rendering a triangle mesh (flat-shaded)."""

    def __init__(self) -> None:
        self.vao: int = 0
        self.vbo: int = 0
        self.count: int = 0
        self._scene_mesh: Mesh | None = None

    def upload(self, scene_mesh: "Mesh") -> None:
        data = scene_mesh.vertices[scene_mesh.faces.flatten()]
        data = np.ascontiguousarray(data, dtype=np.float32)
        if self.vao == 0:
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
        self.count = len(data)
        self._scene_mesh = scene_mesh
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindVertexArray(0)

    def draw(self) -> None:
        if self.vao and self.count:
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.count)
            glBindVertexArray(0)

    def cleanup(self) -> None:
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            glDeleteBuffers(1, [self.vbo])
            self.vao = self.vbo = 0
            self._scene_mesh = None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Row-major 4×4 view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array(
        [
            [r[0],  r[1],  r[2],  -np.dot(r, eye)],
            [u[0],  u[1],  u[2],  -np.dot(u, eye)],
            [-f[0], -f[1], -f[2],  np.dot(f, eye)],
            [0.0,   0.0,   0.0,   1.0],
        ],
        dtype=np.float32,
    )


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Row-major 4×4 perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    n, fa = near, far
    return np.array(
        [
            [f / aspect, 0.0, 0.0,                  0.0],
            [0.0,        f,   0.0,                  0.0],
            [0.0,        0.0, (fa + n) / (n - fa),  2*fa*n / (n - fa)],
            [0.0,        0.0, -1.0,                  0.0],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Orbit camera state
# ---------------------------------------------------------------------------


class OrbitController:
    """Tracks the orbit camera state (azimuth, elevation, distance, target).

    All angles are in degrees for readability; converted to radians when
    computing the camera position.
    """

    SENSITIVITY_ORBIT = 0.35   # degrees per screen pixel
    SENSITIVITY_PAN   = 0.005  # world units per screen pixel
    SENSITIVITY_ZOOM  = 0.12   # fraction of distance per scroll step

    def __init__(self) -> None:
        self.azimuth: float   = 35.0    # degrees, horizontal rotation
        self.elevation: float = 28.0    # degrees, above XZ plane
        self.distance: float  = 14.0
        self.target: np.ndarray = np.zeros(3, dtype=np.float32)
        self._up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        x = self.target[0] + self.distance * np.cos(el) * np.sin(az)
        y = self.target[1] + self.distance * np.sin(el)
        z = self.target[2] + self.distance * np.cos(el) * np.cos(az)
        return np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth  += dx * self.SENSITIVITY_ORBIT
        self.elevation -= dy * self.SENSITIVITY_ORBIT   # screen Y is down
        self.elevation  = float(np.clip(self.elevation, -89.0, 89.0))

    def pan(self, dx: float, dy: float) -> None:
        """Translate the target point in the camera's local right/up plane."""
        az = np.radians(self.azimuth)
        right = np.array([np.cos(az), 0.0, -np.sin(az)], dtype=np.float32)
        # Screen up is camera up projected to XZ + Y
        el = np.radians(self.elevation)
        cam_up = np.array(
            [-np.sin(az) * np.sin(el), np.cos(el), -np.cos(az) * np.sin(el)],
            dtype=np.float32,
        )
        scale = self.distance * self.SENSITIVITY_PAN
        self.target -= right * dx * scale
        self.target += cam_up * dy * scale

    def zoom(self, steps: float) -> None:
        self.distance *= (1.0 - steps * self.SENSITIVITY_ZOOM)
        self.distance = float(np.clip(self.distance, 0.5, 200.0))

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.target, self._up)


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------


def _make_grid(extent: float = 10.0, minor_step: float = 1.0) -> np.ndarray:
    """Return (N, 3) vertices for grid lines on the XZ plane."""
    n = int(round(extent / minor_step))
    pairs: list[list[float]] = []
    for i in range(-n, n + 1):
        x = i * minor_step
        pairs += [[-extent, 0.0, x], [extent, 0.0, x]]
        pairs += [[x, 0.0, -extent], [x, 0.0, extent]]
    return np.array(pairs, dtype=np.float32)


def _make_major_grid(extent: float = 10.0, major_step: float = 5.0) -> np.ndarray:
    n = int(round(extent / major_step))
    pairs: list[list[float]] = []
    for i in range(-n, n + 1):
        x = i * major_step
        pairs += [[-extent, 0.0, x], [extent, 0.0, x]]
        pairs += [[x, 0.0, -extent], [x, 0.0, extent]]
    return np.array(pairs, dtype=np.float32)


def _make_axis_lines(length: float = 2.5) -> dict[str, np.ndarray]:
    return {
        "x": np.array([[0, 0, 0], [length, 0, 0]], dtype=np.float32),
        "y": np.array([[0, 0, 0], [0, length, 0]], dtype=np.float32),
        "z": np.array([[0, 0, 0], [0, 0, length]], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Viewport widget
# ---------------------------------------------------------------------------


class Viewport(QOpenGLWidget):
    """OpenGL 3.3 Core viewport with orbit camera and scene rendering."""

    def __init__(self, scene: "Scene", parent=None) -> None:
        super().__init__(parent)
        self._scene = scene

        # Orbit camera
        self._orbit = OrbitController()

        # Mouse tracking state
        self._last_pos = QPoint()
        self._left_down = False
        self._middle_down = False

        # Momentum (left-drag orbit only)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._momentum_timer = QTimer(self)
        self._momentum_timer.setInterval(16)   # ~60 fps
        self._momentum_timer.timeout.connect(self._tick_momentum)

        # GPU resources (allocated in initializeGL)
        self._line_prog: int = 0
        self._mesh_prog: int = 0
        self._grid_minor = _GPULines()
        self._grid_major = _GPULines()
        self._axis_x = _GPULines()
        self._axis_y = _GPULines()
        self._axis_z = _GPULines()
        self._frustum_gpu: list[_GPULines] = []
        self._mesh_gpu: dict[int, _GPUMesh] = {}   # id(Mesh) → _GPUMesh

        # Projection
        self._fov = 45.0
        self._near = 0.1
        self._far = 500.0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 400)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scene(self) -> "Scene":
        return self._scene

    def frame_scene(self) -> None:
        """Reset orbit camera to a sensible default framing."""
        self._orbit.azimuth   = 35.0
        self._orbit.elevation = 28.0
        self._orbit.distance  = 14.0
        self._orbit.target[:] = 0.0
        self.update()

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        # macOS / Qt leaves pending errors in the context during setup.
        # Flush them before our first PyOpenGL call, otherwise PyOpenGL's
        # automatic glGetError() check raises on GL_DEPTH_TEST itself.
        while glGetError() != GL_NO_ERROR:
            pass

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.12, 0.12, 0.13, 1.0)

        self._line_prog = _compile_program(_VERT_SRC, _FRAG_SRC)
        self._mesh_prog = _compile_program(_VERT_MESH_SRC, _FRAG_MESH_SRC)

        # Grid
        self._grid_minor.upload(_make_grid(10.0, 1.0))
        self._grid_major.upload(_make_major_grid(10.0, 5.0))

        # Axes
        axes = _make_axis_lines(2.5)
        self._axis_x.upload(axes["x"])
        self._axis_y.upload(axes["y"])
        self._axis_z.upload(axes["z"])

        # Scene camera frustums
        self._upload_frustums()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w, h = self.width(), self.height()
        aspect = w / h if h else 1.0
        proj = _perspective(self._fov, aspect, self._near, self._far)
        view = self._orbit.view_matrix()
        vp = proj @ view   # row-major: P × V × model

        self._sync_meshes()

        # --- Line program ------------------------------------------------
        glUseProgram(self._line_prog)
        mvp_loc   = glGetUniformLocation(self._line_prog, "uMVP")
        color_loc = glGetUniformLocation(self._line_prog, "uColor")

        def draw_lines(gpu: _GPULines, color: tuple, mvp: np.ndarray) -> None:
            glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)
            glUniform3f(color_loc, *color)
            gpu.draw()

        # Grid (minor)
        draw_lines(self._grid_minor, (0.22, 0.22, 0.24), vp)
        # Grid (major) — slightly brighter, same width (Core Profile locks glLineWidth to 1.0)
        draw_lines(self._grid_major, (0.32, 0.32, 0.35), vp)
        # Axes
        draw_lines(self._axis_x, (0.80, 0.20, 0.20), vp)
        draw_lines(self._axis_y, (0.20, 0.75, 0.20), vp)
        draw_lines(self._axis_z, (0.20, 0.40, 0.90), vp)

        # Camera frustums
        for i, gpu in enumerate(self._frustum_gpu):
            if i < len(self._scene.cameras):
                cam = self._scene.cameras[i]
                draw_lines(gpu, cam.color, vp)

        # --- Mesh program -----------------------------------------------
        glUseProgram(self._mesh_prog)
        mesh_mvp_loc   = glGetUniformLocation(self._mesh_prog, "uMVP")
        mesh_color_loc = glGetUniformLocation(self._mesh_prog, "uColor")

        for scene_mesh, gpu_mesh in self._mesh_gpu.items():
            # Retrieve matching Mesh object from id
            mesh_obj = next(
                (m for m in self._scene.meshes if id(m) == scene_mesh), None
            )
            if mesh_obj is None or not mesh_obj.visible:
                continue
            mvp = vp @ mesh_obj.transform
            glUniformMatrix4fv(mesh_mvp_loc, 1, GL_TRUE, mvp)
            glUniform3f(mesh_color_loc, *mesh_obj.color)
            gpu_mesh.draw()

    # ------------------------------------------------------------------
    # Scene sync helpers
    # ------------------------------------------------------------------

    def _upload_frustums(self) -> None:
        """(Re)create GPU frustum wireframes from scene cameras."""
        for g in self._frustum_gpu:
            g.cleanup()
        self._frustum_gpu.clear()
        for cam in self._scene.cameras:
            g = _GPULines()
            g.upload(cam.frustum_line_vertices())
            self._frustum_gpu.append(g)

    def _sync_meshes(self) -> None:
        """Upload any new scene meshes; free GPU resources for removed ones."""
        current_ids = {id(m) for m in self._scene.meshes}

        # Remove stale GPU meshes
        stale = [k for k in self._mesh_gpu if k not in current_ids]
        for k in stale:
            self._mesh_gpu[k].cleanup()
            del self._mesh_gpu[k]

        # Upload new meshes
        for mesh in self._scene.meshes:
            if id(mesh) not in self._mesh_gpu:
                gpu = _GPUMesh()
                gpu.upload(mesh)
                self._mesh_gpu[id(mesh)] = gpu

    # ------------------------------------------------------------------
    # Mouse & keyboard input
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_down = True
            self._momentum_timer.stop()
            self._velocity[:] = 0.0
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._middle_down = True

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_down = False
            if np.linalg.norm(self._velocity) > 0.5:
                self._momentum_timer.start()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._middle_down = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        if self._left_down:
            self._velocity = np.array([-dx, -dy], dtype=np.float64) * 0.6
            self._orbit.orbit(-dx, -dy)
            self.update()
        elif self._middle_down:
            self._orbit.pan(dx, dy)
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        steps = delta / 120.0   # one notch = 120 units
        self._orbit.zoom(steps)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F:
            self.frame_scene()

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _tick_momentum(self) -> None:
        self._velocity *= 0.88
        if np.linalg.norm(self._velocity) < 0.08:
            self._momentum_timer.stop()
            self._velocity[:] = 0.0
            return
        self._orbit.orbit(self._velocity[0], self._velocity[1])
        self.update()
