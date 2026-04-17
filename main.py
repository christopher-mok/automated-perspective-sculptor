"""Entry point for the Perspective Sculptor application."""

import sys

from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def _configure_opengl() -> None:
    """Set up OpenGL 3.3 Core Profile before the QApplication is created."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)   # 4× MSAA
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> None:
    _configure_opengl()

    app = QApplication(sys.argv)
    app.setApplicationName("Perspective Sculptor")
    app.setOrganizationName("APS")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
