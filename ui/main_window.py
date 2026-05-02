"""Main application window: three-panel layout."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from scene.camera import Camera
from scene.scene import Mesh, Scene
from ui.controls_panel import ControlsPanel
from ui.image_panel import ImagePanel
from ui.viewport import Viewport

_TARGET_TRANSPARENT_BORDER_FRACTION = 0.10
_ORIGINAL_BOUNDS_SIZE = 5.0
_HANGING_PLANE_Y = (_ORIGINAL_BOUNDS_SIZE * 0.5) + 1.0
_HANGING_PLANE_FRAME_THICKNESS = 0.035
_SCENE_CAMERA_FOV_DEG = 30.22  # 50mm equivalent on a 36x27mm 4:3 sensor.


def _make_scene_cameras() -> list[Camera]:
    """Create two scene cameras 90° apart, each aimed straight along its axis.

    Camera 1 sits on the +Z axis looking toward the origin (along −Z).
    Camera 2 sits on the +X axis looking toward the origin (along −X).
    """
    radius = 8.0
    origin = np.zeros(3, dtype=np.float32)

    cam1 = Camera(
        position=np.array([0.0, 0.0, radius], dtype=np.float32),
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
        color=(1.0, 0.85, 0.0),   # gold — View 1 along Z
        label="View 1",
    )
    cam2 = Camera(
        position=np.array([radius, 0.0, 0.0], dtype=np.float32),
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
        color=(0.2, 0.8, 1.0),   # cyan — View 2 along X
        label="View 2",
    )
    return [cam1, cam2]


def _load_target_image_with_border(path: str) -> np.ndarray:
    """Load an image as RGBA and add a transparent border around it."""
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGBA")
    border = max(1, int(round(max(image.size) * _TARGET_TRANSPARENT_BORDER_FRACTION)))
    return np.array(ImageOps.expand(image, border=border, fill=(0, 0, 0, 0)))


def _make_hanging_plane_mesh(size: float) -> Mesh:
    """Create a thin square frame marking the hanging plane footprint."""
    half = max(size * 0.5, _HANGING_PLANE_FRAME_THICKNESS)
    t = min(_HANGING_PLANE_FRAME_THICKNESS, half)
    y = _HANGING_PLANE_Y

    bars = [
        (-half, half, -half, -half + t),
        (-half, half, half - t, half),
        (-half, -half + t, -half, half),
        (half - t, half, -half, half),
    ]
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for x0, x1, z0, z1 in bars:
        start = len(vertices)
        vertices.extend([
            [x0, y, z0],
            [x1, y, z0],
            [x1, y, z1],
            [x0, y, z1],
        ])
        faces.extend([
            [start, start + 1, start + 2],
            [start, start + 2, start + 3],
        ])

    return Mesh(
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        color=(0.55, 0.62, 0.68),
        label="hanging_plane",
    )


class MainWindow(QMainWindow):
    """Top-level application window.

    Layout
    ------
    [ImagePanel | Viewport | ControlsPanel]
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Perspective Sculptor")
        self.resize(1400, 820)
        self.setMinimumSize(900, 600)

        # Dark palette for the whole window
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1a1a1b;
                color: #ddd;
            }
            QSplitter::handle {
                background: #333;
                width: 3px;
            }
            QScrollBar:vertical {
                background: #1a1a1b;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 4px;
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a3a3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                background: #4a9eff;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #2a6496;
                border-radius: 2px;
            }
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ddd;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                selection-background-color: #2a6496;
                color: #ddd;
                border: 1px solid #444;
            }
            """
        )

        # Build scene
        self._scene = Scene()
        for cam in _make_scene_cameras():
            self._scene.add_camera(cam)

        # Patch state
        self._patches: list = []
        self._target1_img: np.ndarray | None = None  # uint8 (H,W,4), padded RGBA
        self._target2_img: np.ndarray | None = None
        self._worker = None
        self._optimization_run_until_convergence = False
        self._reset_after_worker_stops = False

        # Build panels
        self._image_panel   = ImagePanel(self)
        self._viewport      = Viewport(self._scene, self)
        self._controls      = ControlsPanel(self)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._image_panel)
        splitter.addWidget(self._viewport)
        splitter.addWidget(self._controls)
        splitter.setStretchFactor(0, 0)   # image panel: fixed
        splitter.setStretchFactor(1, 1)   # viewport: expands
        splitter.setStretchFactor(2, 0)   # controls: fixed
        splitter.setSizes([270, 860, 270])
        splitter.setChildrenCollapsible(False)

        self.setCentralWidget(splitter)

        # Wire up signals
        self._image_panel.view1_loaded.connect(self._on_view1_loaded)
        self._image_panel.view2_loaded.connect(self._on_view2_loaded)
        self._controls.patches.initialize_requested.connect(self._on_initialize)
        self._controls.optimization.run_requested.connect(self._on_run_optimization)
        self._controls.optimization.pause_toggled.connect(self._on_pause_optimization)
        self._controls.optimization.palette_changed.connect(self._on_palette_changed)
        self._controls.optimization.reset_requested.connect(self._on_reset)
        self._controls.export.export_requested.connect(self._on_export_json)
        self._controls.patches.hanging_plane_size_changed.connect(
            self._on_hanging_plane_size_changed
        )
        self._update_hanging_plane_mesh()
        self._sync_export_enabled()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_view1_loaded(self, path: str) -> None:
        self._target1_img = _load_target_image_with_border(path)
        print(f"[View 1 target] loaded: {path}")

    def _on_view2_loaded(self, path: str) -> None:
        self._target2_img = _load_target_image_with_border(path)
        print(f"[View 2 target] loaded: {path}")

    def _on_initialize(self, n_patches: int, mode: str) -> None:
        from core.initialization import initialize_patches

        device = self._controls.patches.device

        try:
            self._patches = initialize_patches(
                mode=mode,
                n_patches=n_patches,
                reference_image=self._target1_img,
                sam_variant=self._controls.patches.sam_model,
                cameras=self._scene.cameras,
                device=device,
            )
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, self._controls.optimization.palette)
            self._constrain_patches_to_hanging_plane()
        except (ValueError, FileNotFoundError, ImportError) as exc:
            QMessageBox.warning(self, "Initialize patches", str(exc))
            return
        except RuntimeError as exc:
            # Catches e.g. MPS/CUDA not available on this machine
            QMessageBox.warning(self, "Device error", str(exc))
            return

        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()
        self._controls.srd.set_stats({"patches": len(self._patches)})
        self._sync_export_enabled()
        print(f"[Initialize patches] {len(self._patches)} patches ({mode}, {device})")

    def _on_run_optimization(self) -> None:
        if not self._patches:
            QMessageBox.warning(self, "Run optimization", "Initialize patches first.")
            return
        if self._target1_img is None:
            QMessageBox.warning(self, "Run optimization", "Load a View 1 target image first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        from ui.worker import OptimizationWorker

        opt = self._controls.optimization
        view2_loss = "sds" if "SDS" in opt.loss_type else "mse"
        if view2_loss == "sds" and not opt.sds_prompt.strip():
            QMessageBox.warning(self, "Run optimization", "Enter an SDS prompt first.")
            return

        try:
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, opt.palette)
        except ValueError as exc:
            QMessageBox.warning(self, "Run optimization", str(exc))
            return
        self._constrain_patches_to_hanging_plane()
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()

        self._optimization_run_until_convergence = opt.run_until_convergence
        self._worker = OptimizationWorker(
            patches=self._patches,
            cameras=self._scene.cameras,
            target1=self._target1_img,
            target2=self._target2_img,
            palette=opt.palette,
            lr=opt.learning_rate,
            n_steps=opt.n_steps,
            run_until_convergence=opt.run_until_convergence,
            convergence_threshold=opt.convergence_threshold,
            view2_loss=view2_loss,
            sds_prompt=opt.sds_prompt,
            device=self._controls.patches.device,
            hanging_plane_size=self._controls.patches.hanging_plane_size,
            srd_config=self._controls.srd.config,
            parent=self,
        )
        self._worker.step_completed.connect(self._on_optimization_step)
        self._worker.failed.connect(self._on_optimization_failed)
        self._worker.optimization_finished.connect(self._on_optimization_finished)

        self._controls.optimization.set_running(True)
        self._controls.patches.set_running(True)
        self._controls.srd.set_running(True)
        self._controls.optimization.reset_progress()
        self._sync_export_enabled()
        self._worker.start()
        if opt.run_until_convergence:
            print(
                f"[Optimization] started: until loss <= {opt.convergence_threshold:.3e}, "
                f"lr={opt.learning_rate:.3e}, palette={opt.palette!r}, targets=mask"
            )
        else:
            print(
                f"[Optimization] started: steps={opt.n_steps}, lr={opt.learning_rate:.3e}, "
                f"palette={opt.palette!r}, targets=mask"
            )

    def _on_optimization_step(self, step_idx: int, metrics: object, meshes: object) -> None:
        self._viewport.set_meshes(meshes)
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)
        if isinstance(metrics, dict):
            self._controls.srd.set_stats(metrics)
        if not self._optimization_run_until_convergence:
            self._controls.optimization.set_progress(step_idx)
        loss = metrics.get("loss", 0.0) if isinstance(metrics, dict) else 0.0
        if isinstance(metrics, dict):
            print(
                f"[Optimization] step={step_idx}, loss={loss:.6f}, "
                f"terms(avg): view1={metrics.get('view1_mse', 0.0):.6f}, "
                f"view2={metrics.get('view2_loss', 0.0):.6f}, "
                f"view1_sil={metrics.get('view1_silhouette', 0.0):.6f}, "
                f"view2_sil={metrics.get('view2_silhouette', 0.0):.6f}, "
                f"view1_neg={metrics.get('view1_negative_space', 0.0):.6f}, "
                f"view2_neg={metrics.get('view2_negative_space', 0.0):.6f}, "
                f"overlap={metrics.get('overlap', 0.0):.6f}, "
                f"camera_bounds={metrics.get('camera_bounds', 0.0):.6f}; "
                f"smallest_area={metrics.get('smallest_patch_area', 0.0):.6f}, "
                f"tiny_deleted={metrics.get('tiny_patches_deleted', 0.0):.0f}; "
                f"weighted: negative_space={metrics.get('negative_space_weighted', 0.0):.6f}, "
                f"overlap={metrics.get('overlap_weighted', 0.0):.6f}, "
                f"camera_bounds={metrics.get('camera_bounds_weighted', 0.0):.6f}; "
                f"srd: patches={metrics.get('srd_active_patches', metrics.get('patches', 0.0)):.0f}, "
                f"accepted={metrics.get('srd_accepted', 0.0):.0f}"
            )
        else:
            print(f"[Optimization] step={step_idx}, loss={loss:.6f}")

    def _on_pause_optimization(self, paused: bool) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.set_paused(paused)
        print("[Optimization] paused" if paused else "[Optimization] resumed")

    def _on_palette_changed(self) -> None:
        if not self._patches:
            return
        try:
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, self._controls.optimization.palette)
        except ValueError as exc:
            QMessageBox.warning(self, "Palette", str(exc))
            return
        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()

    def _on_hanging_plane_size_changed(self, size: float) -> None:
        self._update_hanging_plane_mesh()
        self._constrain_patches_to_hanging_plane()
        if self._patches:
            self._viewport.set_patches(self._patches)
            self._update_camera_previews_from_patches()
        print(f"[Hanging plane] size={size:.1f}, y={_HANGING_PLANE_Y:.1f}")

    def _on_reset(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._reset_after_worker_stops = True
            self._worker.request_stop()
            print("[Reset] stopping optimization before clearing state")
            return

        self._reset_state()

    def _on_export_json(self) -> None:
        if not self._patches:
            QMessageBox.warning(self, "Export", "No patches to export.")
            return

        try:
            from core.export import export_patches_to_json

            output_path = export_patches_to_json(
                self._patches,
                hanging_plane_size=self._controls.patches.hanging_plane_size,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            print(f"[Export] failed: {exc}")
            return

        QMessageBox.information(
            self,
            "Export complete",
            f"Saved piece data to {output_path}",
        )
        print(f"[Export] wrote JSON to {output_path}")

    def _reset_state(self) -> None:
        self._worker = None
        self._optimization_run_until_convergence = False
        self._reset_after_worker_stops = False
        self._patches = []
        self._target1_img = None
        self._target2_img = None
        self._image_panel.reset()
        self._viewport.reset()
        self._update_hanging_plane_mesh()
        self._controls.optimization.reset_controls()
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._controls.srd.set_stats({})
        self._sync_export_enabled()
        print("[Reset] cleared targets, patches, optimization state, and viewport")

    def _update_camera_previews_from_patches(self) -> None:
        meshes = [patch.to_mesh() for patch in self._patches]
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)

    def _update_hanging_plane_mesh(self) -> None:
        self._viewport.set_static_meshes([
            _make_hanging_plane_mesh(self._controls.patches.hanging_plane_size)
        ])

    def _constrain_patches_to_hanging_plane(self) -> None:
        if not self._patches:
            return
        import torch
        from core.optimizer import constrain_patch_to_square_xz_bounds

        half = max(self._controls.patches.hanging_plane_size * 0.5, 1e-4)
        with torch.no_grad():
            for patch in self._patches:
                constrain_patch_to_square_xz_bounds(patch, half)

    def _on_optimization_failed(self, message: str) -> None:
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._sync_export_enabled()
        QMessageBox.warning(self, "Optimization failed", message)
        print(f"[Optimization] failed: {message}")

    def _on_optimization_finished(self, metrics: object) -> None:
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        self._controls.patches.set_running(False)
        self._controls.srd.set_running(False)
        self._sync_export_enabled()
        loss = metrics.get("loss", None) if isinstance(metrics, dict) else None
        if loss is None:
            print("[Optimization] finished")
        else:
            print(f"[Optimization] finished: loss={loss:.6f}")

    def _sync_export_enabled(self) -> None:
        self._controls.export.set_enabled(bool(self._patches))

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(1500)
        super().closeEvent(event)
