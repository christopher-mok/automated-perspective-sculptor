"""Main application window: three-panel layout."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from scene.camera import Camera
from scene.scene import Scene
from ui.controls_panel import ControlsPanel
from ui.image_panel import ImagePanel
from ui.viewport import Viewport


def _make_scene_cameras() -> list[Camera]:
    """Create two scene cameras 90° apart, each aimed straight along its axis.

    Camera 1 sits on the +Z axis looking toward the origin (along −Z).
    Camera 2 sits on the +X axis looking toward the origin (along −X).
    """
    radius = 5.5
    origin = np.zeros(3, dtype=np.float32)

    cam1 = Camera(
        position=np.array([0.0, 0.0, radius], dtype=np.float32),
        target=origin,
        fov=50.0,
        aspect=4.0 / 3.0,
        near=0.35,
        far=6.0,
        color=(1.0, 0.85, 0.0),   # gold — View 1 along Z
        label="View 1",
    )
    cam2 = Camera(
        position=np.array([radius, 0.0, 0.0], dtype=np.float32),
        target=origin,
        fov=50.0,
        aspect=4.0 / 3.0,
        near=0.35,
        far=6.0,
        color=(0.2, 0.8, 1.0),   # cyan — View 2 along X
        label="View 2",
    )
    return [cam1, cam2]


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
        self._target1_img: np.ndarray | None = None  # uint8 (H,W,3) for SAM
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

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_view1_loaded(self, path: str) -> None:
        from PIL import Image
        self._target1_img = np.array(Image.open(path).convert("RGB"))
        print(f"[View 1 target] loaded: {path}")

    def _on_view2_loaded(self, path: str) -> None:
        from PIL import Image
        self._target2_img = np.array(Image.open(path).convert("RGB"))
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
                device=device,
            )
            from core.optimizer import snap_patches_to_palette

            snap_patches_to_palette(self._patches, self._controls.optimization.palette)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            QMessageBox.warning(self, "Initialize patches", str(exc))
            return
        except RuntimeError as exc:
            # Catches e.g. MPS/CUDA not available on this machine
            QMessageBox.warning(self, "Device error", str(exc))
            return

        self._viewport.set_patches(self._patches)
        self._update_camera_previews_from_patches()
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
            parent=self,
        )
        self._worker.step_completed.connect(self._on_optimization_step)
        self._worker.failed.connect(self._on_optimization_failed)
        self._worker.optimization_finished.connect(self._on_optimization_finished)

        self._controls.optimization.set_running(True)
        self._controls.optimization.reset_progress()
        self._controls.export.set_enabled(False)
        self._worker.start()
        if opt.run_until_convergence:
            print(
                f"[Optimization] started: until loss <= {opt.convergence_threshold:.3e}, "
                f"lr={opt.learning_rate:.3e}, palette={opt.palette!r}"
            )
        else:
            print(
                f"[Optimization] started: steps={opt.n_steps}, lr={opt.learning_rate:.3e}, "
                f"palette={opt.palette!r}"
            )

    def _on_optimization_step(self, step_idx: int, metrics: object, meshes: object) -> None:
        self._viewport.set_meshes(meshes)
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)
        if not self._optimization_run_until_convergence:
            self._controls.optimization.set_progress(step_idx)
        if step_idx == 1 or step_idx % 10 == 0:
            loss = metrics.get("loss", 0.0) if isinstance(metrics, dict) else 0.0
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

    def _on_reset(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._reset_after_worker_stops = True
            self._worker.request_stop()
            print("[Reset] stopping optimization before clearing state")
            return

        self._reset_state()

    def _reset_state(self) -> None:
        self._worker = None
        self._optimization_run_until_convergence = False
        self._reset_after_worker_stops = False
        self._patches = []
        self._target1_img = None
        self._target2_img = None
        self._image_panel.reset()
        self._viewport.reset()
        self._controls.optimization.reset_controls()
        self._controls.export.set_enabled(False)
        print("[Reset] cleared targets, patches, optimization state, and viewport")

    def _update_camera_previews_from_patches(self) -> None:
        meshes = [patch.to_mesh() for patch in self._patches]
        self._image_panel.set_camera_previews(meshes, self._scene.cameras)

    def _on_optimization_failed(self, message: str) -> None:
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        QMessageBox.warning(self, "Optimization failed", message)
        print(f"[Optimization] failed: {message}")

    def _on_optimization_finished(self, metrics: object) -> None:
        if self._reset_after_worker_stops:
            self._reset_state()
            return
        self._controls.optimization.set_running(False)
        self._controls.export.set_enabled(True)
        loss = metrics.get("loss", None) if isinstance(metrics, dict) else None
        if loss is None:
            print("[Optimization] finished")
        else:
            print(f"[Optimization] finished: loss={loss:.6f}")

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(1500)
        super().closeEvent(event)
