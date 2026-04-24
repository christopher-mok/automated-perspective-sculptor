"""Right panel: Patches, Optimization, and Export controls."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTION_STYLE = """
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

_LABEL_STYLE = "color: #aaa; font-size: 12px;"
_VALUE_STYLE = "color: #ddd; font-size: 12px; min-width: 52px;"


def _row(label_text: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setStyleSheet(_LABEL_STYLE)
    row.addWidget(lbl)
    row.addWidget(widget)
    return row


def _labeled_slider(
    minimum: int,
    maximum: int,
    default: int,
    fmt: str = "{}",
) -> tuple[QSlider, QLabel]:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)

    value_lbl = QLabel(fmt.format(default))
    value_lbl.setStyleSheet(_VALUE_STYLE)
    value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    return slider, value_lbl


# ---------------------------------------------------------------------------
# Patches section
# ---------------------------------------------------------------------------


class PatchesSection(QGroupBox):
    initialize_requested = pyqtSignal(int, str)   # n_patches, init_mode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Patches", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        # Number of patches slider
        self._n_slider, self._n_lbl = _labeled_slider(4, 200, 20, "{}")
        self._n_slider.valueChanged.connect(
            lambda v: self._n_lbl.setText(str(v))
        )

        n_row = QHBoxLayout()
        n_lbl = QLabel("Number of patches")
        n_lbl.setStyleSheet(_LABEL_STYLE)
        n_row.addWidget(n_lbl)
        n_row.addStretch()
        n_row.addWidget(self._n_lbl)
        layout.addLayout(n_row)
        layout.addWidget(self._n_slider)

        # Initialization dropdown
        self._init_combo = QComboBox()
        self._init_combo.addItems(["SAM segmentation", "Grid", "Random"])
        self._init_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        self._init_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addLayout(_row("Initialization", self._init_combo))

        # SAM model selector (only visible in SAM mode)
        self._sam_model_lbl = QLabel("SAM model")
        self._sam_model_lbl.setStyleSheet(_LABEL_STYLE)
        self._sam_model_combo = QComboBox()
        self._sam_model_combo.addItems([
            "MobileSAM (fast)",
            "SAM vit_b (balanced)",
            "SAM vit_h (best quality)",
        ])
        self._sam_model_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        self._sam_model_row = QHBoxLayout()
        self._sam_model_row.addWidget(self._sam_model_lbl)
        self._sam_model_row.addWidget(self._sam_model_combo)
        layout.addLayout(self._sam_model_row)

        # Device toggle
        self._device_combo = QComboBox()
        self._device_combo.addItems(["Mac (CPU)", "Mac (MPS)", "CUDA (NVIDIA)"])
        self._device_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        layout.addLayout(_row("Device", self._device_combo))

        # Initialize button
        self._init_btn = QPushButton("Initialize patches")
        self._init_btn.setStyleSheet(
            "QPushButton { background: #2a6496; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #3276b1; }"
            "QPushButton:pressed { background: #204d74; }"
        )
        self._init_btn.clicked.connect(self._on_initialize)
        layout.addWidget(self._init_btn)

        # Set initial SAM row visibility
        self._on_mode_changed(self._init_combo.currentText())

    def _on_mode_changed(self, text: str) -> None:
        sam_only = text == "SAM segmentation"
        self._sam_model_lbl.setVisible(sam_only)
        self._sam_model_combo.setVisible(sam_only)

    def _on_initialize(self) -> None:
        n = self._n_slider.value()
        mode = self._init_combo.currentText()
        print(f"[Initialize patches] n={n}, mode={mode!r}")
        self.initialize_requested.emit(n, mode)

        # Trigger visibility update in case the widget was just shown
        self._on_mode_changed(mode)

    @property
    def n_patches(self) -> int:
        return self._n_slider.value()

    @property
    def init_mode(self) -> str:
        return self._init_combo.currentText()

    @property
    def sam_model(self) -> str:
        """Returns the selected SAM variant label."""
        return self._sam_model_combo.currentText()

    @property
    def device(self) -> str:
        """Returns the torch device string for the selected option."""
        _map = {
            "Mac (CPU)":      "cpu",
            "Mac (MPS)":      "mps",
            "CUDA (NVIDIA)":  "cuda",
        }
        return _map.get(self._device_combo.currentText(), "cpu")


# ---------------------------------------------------------------------------
# Optimization section
# ---------------------------------------------------------------------------

_LR_MIN_LOG = -4.0   # 1e-4
_LR_MAX_LOG = -1.0   # 1e-1
_LR_STEPS = 1000


def _slider_to_lr(pos: int) -> float:
    t = pos / _LR_STEPS
    return 10.0 ** (_LR_MIN_LOG + (_LR_MAX_LOG - _LR_MIN_LOG) * t)


def _lr_to_slider(lr: float) -> int:
    log = math.log10(max(lr, 1e-8))
    t = (log - _LR_MIN_LOG) / (_LR_MAX_LOG - _LR_MIN_LOG)
    return int(round(t * _LR_STEPS))


class OptimizationSection(QGroupBox):
    run_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Optimization", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        # View 2 loss dropdown
        self._loss_combo = QComboBox()
        self._loss_combo.addItems(["MSE (target image)", "SDS (text prompt)"])
        self._loss_combo.setStyleSheet("color: #ddd; background: #2a2a2a;")
        self._loss_combo.currentTextChanged.connect(self._on_loss_changed)
        layout.addLayout(_row("View 2 loss", self._loss_combo))

        # SDS prompt input
        sds_lbl = QLabel("SDS prompt")
        sds_lbl.setStyleSheet(_LABEL_STYLE)
        self._sds_input = QLineEdit()
        self._sds_input.setPlaceholderText("Describe the target appearance…")
        self._sds_input.setStyleSheet(
            "color: #ddd; background: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 4px;"
        )
        self._sds_input.setEnabled(False)
        layout.addWidget(sds_lbl)
        layout.addWidget(self._sds_input)

        # Learning rate slider (log scale)
        default_lr = 1e-3
        self._lr_slider = QSlider(Qt.Orientation.Horizontal)
        self._lr_slider.setRange(0, _LR_STEPS)
        self._lr_slider.setValue(_lr_to_slider(default_lr))

        self._lr_lbl = QLabel(f"{default_lr:.2e}")
        self._lr_lbl.setStyleSheet(_VALUE_STYLE)
        self._lr_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._lr_slider.valueChanged.connect(
            lambda v: self._lr_lbl.setText(f"{_slider_to_lr(v):.2e}")
        )

        lr_row = QHBoxLayout()
        lr_row_lbl = QLabel("Learning rate")
        lr_row_lbl.setStyleSheet(_LABEL_STYLE)
        lr_row.addWidget(lr_row_lbl)
        lr_row.addStretch()
        lr_row.addWidget(self._lr_lbl)
        layout.addLayout(lr_row)
        layout.addWidget(self._lr_slider)

        # Step count
        self._steps_slider, self._steps_lbl = _labeled_slider(1, 1000, 200, "{}")
        self._steps_slider.valueChanged.connect(
            lambda v: self._steps_lbl.setText(str(v))
        )
        steps_row = QHBoxLayout()
        steps_lbl = QLabel("Steps")
        steps_lbl.setStyleSheet(_LABEL_STYLE)
        steps_row.addWidget(steps_lbl)
        steps_row.addStretch()
        steps_row.addWidget(self._steps_lbl)
        layout.addLayout(steps_row)
        layout.addWidget(self._steps_slider)

        # Discrete colour palette
        palette_lbl = QLabel("Palette")
        palette_lbl.setStyleSheet(_LABEL_STYLE)
        self._palette_input = QLineEdit()
        self._palette_input.setText("#111111, #f5f5f5")
        self._palette_input.setPlaceholderText("#111111, #f4d35e, #2f6690")
        self._palette_input.setStyleSheet(
            "color: #ddd; background: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 4px;"
        )
        layout.addWidget(palette_lbl)
        layout.addWidget(self._palette_input)

        # Run button
        self._run_btn = QPushButton("Run optimization")
        self._run_btn.setStyleSheet(
            "QPushButton { background: #3d6b2e; color: #fff; border-radius: 4px; padding: 6px; }"
            "QPushButton:hover { background: #4e8a3a; }"
            "QPushButton:pressed { background: #2c5020; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

    def _on_loss_changed(self, text: str) -> None:
        self._sds_input.setEnabled("SDS" in text)

    def _on_run(self) -> None:
        lr = _slider_to_lr(self._lr_slider.value())
        loss = self._loss_combo.currentText()
        prompt = self._sds_input.text()
        print(
            f"[Run optimization] loss={loss!r}, lr={lr:.4e}, sds_prompt={prompt!r}"
        )
        self.run_requested.emit()

    @property
    def learning_rate(self) -> float:
        return _slider_to_lr(self._lr_slider.value())

    @property
    def n_steps(self) -> int:
        return self._steps_slider.value()

    @property
    def palette(self) -> str:
        return self._palette_input.text()

    @property
    def loss_type(self) -> str:
        return self._loss_combo.currentText()

    @property
    def sds_prompt(self) -> str:
        return self._sds_input.text()

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Optimizing..." if running else "Run optimization")


# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------


class ExportSection(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Export", parent)
        self.setStyleSheet(_SECTION_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        self._export_btn = QPushButton("Export SVG for laser cutter")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #777; border-radius: 4px; padding: 6px; }"
            "QPushButton:enabled { background: #5a3e7a; color: #fff; }"
            "QPushButton:enabled:hover { background: #6f4e96; }"
        )
        layout.addWidget(self._export_btn)

        note = QLabel("Run optimization first to enable export.")
        note.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        note.setWordWrap(True)
        layout.addWidget(note)

    def set_enabled(self, enabled: bool) -> None:
        self._export_btn.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Main controls panel
# ---------------------------------------------------------------------------


class ControlsPanel(QWidget):
    """Right-side panel containing all controls in a scroll area."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Controls", self)
        header.setStyleSheet(
            "color: #ddd; font-size: 14px; font-weight: bold; padding: 10px 10px 4px 10px;"
        )
        outer.addWidget(header)

        # Scroll area so controls don't get clipped on small windows
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 6, 10, 10)
        layout.setSpacing(12)

        self.patches = PatchesSection(container)
        layout.addWidget(self.patches)

        self.optimization = OptimizationSection(container)
        layout.addWidget(self.optimization)

        self.export = ExportSection(container)
        layout.addWidget(self.export)

        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)
