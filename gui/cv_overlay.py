from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class CVOverlayWindow(QWidget):
    """Visual debug inspector displaying the live cropped board, 8x8 grid lines, and square confidences."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CV Debug Overlay & Grid Inspector")
        self.resize(520, 540)
        self.setStyleSheet("background-color: #11111B; color: #CDD6F4;")

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.lbl_image: QLabel = QLabel("Awaiting Live Screen Capture...")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_image.setStyleSheet("border: 1px solid #313244; border-radius: 6px;")
        layout.addWidget(self.lbl_image)

        self.lbl_info: QLabel = QLabel("Status: Idle | Green = Square Boundaries | Cyan = Inset ROI")
        self.lbl_info.setStyleSheet("color: #BAC2DE; font-size: 11px; padding: 4px;")
        layout.addWidget(self.lbl_info)

    def update_debug_frame(self, qimage: QImage) -> None:
        """Scales and renders the live annotated OpenCV frame."""
        if not self.isVisible():
            return

        pixmap: QPixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.lbl_image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled_pixmap)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hides the overlay window on user close rather than destroying it."""
        self.hide()
        event.ignore()
