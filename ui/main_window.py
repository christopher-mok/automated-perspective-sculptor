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

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_view1_loaded(self, path: str) -> None:
        from PIL import Image
        self._target1_img = np.array(Image.open(path).convert("RGB"))
        print(f"[View 1 target] loaded: {path}")

    def _on_view2_loaded(self, path: str) -> None:
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
        except (ValueError, FileNotFoundError, ImportError) as exc:
            QMessageBox.warning(self, "Initialize patches", str(exc))
            return
        except RuntimeError as exc:
            # Catches e.g. MPS/CUDA not available on this machine
            QMessageBox.warning(self, "Device error", str(exc))
            return

        self._viewport.set_patches(self._patches)
        print(f"[Initialize patches] {len(self._patches)} patches ({mode}, {device})")
