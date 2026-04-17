"""Left panel: two image upload slots for View 1 and View 2 target images."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_ACCEPTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_THUMB_SIZE = 200   # px, max dimension for thumbnail


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
        self.setText("Drop image here\nor click to browse")
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
            "Open target image",
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

    def _on_image(self, path: str) -> None:
        self._path = path
        # Read actual resolution via QPixmap (already decoded by DropZone)
        pix = QPixmap(path)
        self._res_label.setText(f"{pix.width()} × {pix.height()} px")
        self.image_loaded.emit(path)


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

        header = QLabel("Target Images", self)
        header.setStyleSheet("color: #ddd; font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        self._slot1 = ImageSlot("View 1 target", self)
        self._slot1.image_loaded.connect(self.view1_loaded)
        layout.addWidget(self._slot1)

        self._slot2 = ImageSlot("View 2 target", self)
        self._slot2.image_loaded.connect(self.view2_loaded)
        layout.addWidget(self._slot2)

        layout.addStretch()

    @property
    def view1_path(self) -> str | None:
        return self._slot1.image_path

    @property
    def view2_path(self) -> str | None:
        return self._slot2.image_path
