"""Left panel: two image upload slots for View 1 and View 2 silhouette targets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPainter, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from scene.camera import Camera
    from scene.scene import Mesh

_ACCEPTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_THUMB_SIZE = 200   # px, max dimension for thumbnail
_PREVIEW_SIZE = (220, 165)


class DropZone(QLabel):
    """A drag-and-drop area that also accepts clicks to open a file dialog."""

    image_loaded = pyqtSignal(str)   # emits file path on successful load

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 170)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._set_empty_style()
        self.setText("Drop B/W image here\nblack=foreground")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    def _set_empty_style(self) -> None:
        self.setStyleSheet(
            """
            DropZone {
                border: 2px dashed #555;
                border-radius: 6px;
                color: #888;
                font-size: 12px;
                background: #1e1e1e;
            }
            DropZone:hover {
                border-color: #888;
                background: #252525;
            }
            """
        )

    def _set_filled_style(self) -> None:
        self.setStyleSheet(
            """
            DropZone {
                border: 2px solid #4a9eff;
                border-radius: 6px;
                background: #1e1e1e;
            }
            """
        )

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if any(Path(p).suffix.lower() in _ACCEPTED_EXTS for p in paths):
                event.acceptProposedAction()
                self.setStyleSheet(
                    self.styleSheet().replace("#555", "#4a9eff")
                )
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._set_empty_style()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._set_empty_style()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _ACCEPTED_EXTS:
                self._load(path)
                break

    # ------------------------------------------------------------------
    # Click to browse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open black/white target image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)",
        )
        if path:
            self._load(path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            return
        scaled = pix.scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setText("")
        self._set_filled_style()
        self.image_loaded.emit(path)

    def reset(self) -> None:
        super().clear()
        self.setText("Drop B/W image here\nblack=foreground")
        self._set_empty_style()


class ImageSlot(QGroupBox):
    """A labeled group containing a DropZone and a resolution label."""

    image_loaded = pyqtSignal(str)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._path: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._drop_zone = DropZone(self)
        self._drop_zone.image_loaded.connect(self._on_image)
        layout.addWidget(self._drop_zone)

        self._res_label = QLabel("No image loaded", self)
        self._res_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._res_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._res_label)

        self.setStyleSheet(
            """
            QGroupBox {
                color: #ccc;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
            """
        )

    @property
    def image_path(self) -> str | None:
        return self._path

    def reset(self) -> None:
        self._path = None
        self._drop_zone.reset()
        self._res_label.setText("No image loaded")

    def _on_image(self, path: str) -> None:
        self._path = path
        # Read actual resolution via QPixmap (already decoded by DropZone)
        pix = QPixmap(path)
        self._res_label.setText(f"{pix.width()} × {pix.height()} px")
        self.image_loaded.emit(path)


def _render_camera_preview(
    meshes: list["Mesh"],
    camera: "Camera",
    width: int,
    height: int,
) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(24, 24, 25))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    mvp = camera.projection_matrix() @ camera.view_matrix()
    draw_items: list[tuple[float, QPolygonF, tuple[float, float, float]]] = []

    for mesh in meshes:
        if not mesh.visible or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        verts = np.c_[mesh.vertices, np.ones(len(mesh.vertices), dtype=np.float32)]
        world = verts @ mesh.transform.T
        clip = world @ mvp.T
        w = clip[:, 3]
        valid = w > 1e-5
        ndc = np.zeros((len(clip), 3), dtype=np.float32)
        ndc[valid] = clip[valid, :3] / w[valid, None]

        screen = np.empty((len(clip), 2), dtype=np.float32)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)

        for face in mesh.faces:
            if not np.all(valid[face]):
                continue
            tri_ndc = ndc[face]
            if np.all((tri_ndc[:, 0] < -1.1) | (tri_ndc[:, 0] > 1.1)):
                continue
            if np.all((tri_ndc[:, 1] < -1.1) | (tri_ndc[:, 1] > 1.1)):
                continue
            if np.all((tri_ndc[:, 2] < -1.0) | (tri_ndc[:, 2] > 1.0)):
                continue
            pts = [screen[i] for i in face]
            polygon = QPolygonF()
            for x, y in pts:
                polygon.append(QPointF(float(x), float(y)))
            avg_z = float(tri_ndc[:, 2].mean())
            draw_items.append((avg_z, polygon, mesh.color))

    painter.setPen(Qt.PenStyle.NoPen)
    for _avg_z, polygon, color in sorted(draw_items, key=lambda item: item[0], reverse=True):
        r, g, b = [max(0, min(255, int(c * 255))) for c in color]
        painter.setBrush(QColor(r, g, b))
        painter.drawPolygon(polygon)

    painter.end()
    return image


class CameraPreviewSlot(QGroupBox):
    """A read-only thumbnail of what a scene camera sees."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(*_PREVIEW_SIZE)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._preview.setStyleSheet(
            "border: 1px solid #3a3a3a; border-radius: 4px; color: #777; background: #181819;"
        )
        self._preview.setText("No patches")
        layout.addWidget(self._preview)

        self.setStyleSheet(
            """
            QGroupBox {
                color: #ccc;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
            """
        )

    def set_preview(self, image: QImage) -> None:
        pix = QPixmap.fromImage(image).scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pix)
        self._preview.setText("")

    def reset(self) -> None:
        self._preview.clear()
        self._preview.setText("No patches")


class ImagePanel(QWidget):
    """Left panel with two stacked image upload slots."""

    view1_loaded = pyqtSignal(str)
    view2_loaded = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(250)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        header = QLabel("Target Masks", self)
        header.setStyleSheet("color: #ddd; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        self._slot1 = ImageSlot("View 1 target mask", self)
        self._slot1.image_loaded.connect(self.view1_loaded)
        layout.addWidget(self._slot1)

        self._slot2 = ImageSlot("View 2 target mask", self)
        self._slot2.image_loaded.connect(self.view2_loaded)
        layout.addWidget(self._slot2)

        preview_header = QLabel("Camera Views", self)
        preview_header.setStyleSheet("color: #ddd; font-size: 14px; font-weight: bold;")
        layout.addWidget(preview_header)

        self._preview1 = CameraPreviewSlot("View 1 camera", self)
        layout.addWidget(self._preview1)

        self._preview2 = CameraPreviewSlot("View 2 camera", self)
        layout.addWidget(self._preview2)

        layout.addStretch()

    @property
    def view1_path(self) -> str | None:
        return self._slot1.image_path

    @property
    def view2_path(self) -> str | None:
        return self._slot2.image_path

    def reset(self) -> None:
        self._slot1.reset()
        self._slot2.reset()
        self._preview1.reset()
        self._preview2.reset()

    def set_camera_previews(self, meshes: list["Mesh"], cameras: list["Camera"]) -> None:
        if len(cameras) < 2 or not meshes:
            self._preview1.reset()
            self._preview2.reset()
            return

        width, height = _PREVIEW_SIZE
        self._preview1.set_preview(_render_camera_preview(meshes, cameras[0], width, height))
        self._preview2.set_preview(_render_camera_preview(meshes, cameras[1], width, height))
