"""Custom widgets for bird recognition GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .styles import RESULT_CARD_STYLE, TOP1_CARD_STYLE

try:
    from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, Qt, Signal
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
    from PySide6.QtWidgets import (
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import Property, QEasingCurve, QPropertyAnimation, Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont
    from PyQt5.QtWidgets import (
        QFrame,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QVBoxLayout,
        QWidget,
    )

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure


class DropZone(QFrame):
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(110)
        layout = QVBoxLayout(self)
        self.icon = QLabel("🎵")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 36px; background: transparent;")
        self.hint = QLabel("拖拽音频文件到此处\n或点击下方按钮选择 / 录音")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color: #a5d6a7; font-size: 14px; background: transparent;")
        layout.addWidget(self.icon)
        layout.addWidget(self.hint)
        self._drag_active = False

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(
                    (".wav", ".ogg", ".mp3", ".flac", ".m4a", ".aac")
                ):
                    event.acceptProposedAction()
                    self.setProperty("dragActive", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    self._drag_active = True
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self._drag_active = False

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".wav", ".ogg", ".mp3", ".flac", ".m4a", ".aac")):
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=2.2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_alpha(0.0)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()

    def _style_ax(self):
        self.ax.set_facecolor((0, 0, 0, 0))
        self.ax.tick_params(colors="#9fd4b0", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#4caf50")
            spine.set_alpha(0.35)

    def plot_waveform(self, y: np.ndarray, sr: int):
        self.ax.clear()
        self._style_ax()
        if len(y) == 0:
            self.draw()
            return
        t = np.arange(len(y)) / sr
        self.ax.plot(t, y, color="#69f0ae", linewidth=0.9, alpha=0.95)
        self.ax.fill_between(t, y, 0, color="#26a69a", alpha=0.25)
        self.ax.set_xlabel("时间 (秒)", color="#9fd4b0", fontsize=9)
        self.ax.set_ylabel("振幅", color="#9fd4b0", fontsize=9)
        self.ax.set_title("波形", color="#c8e6c9", fontsize=10, pad=6)
        self.fig.tight_layout(pad=0.8)
        self.draw()

    def plot_mel(self, mel: np.ndarray, sr: int, hop: int = 512):
        self.ax.clear()
        self._style_ax()
        if mel.size == 0:
            self.draw()
            return
        extent = [0, mel.shape[1] * hop / sr, 0, mel.shape[0]]
        im = self.ax.imshow(
            mel, origin="lower", aspect="auto", cmap="magma",
            extent=extent, interpolation="bilinear",
        )
        self.ax.set_xlabel("时间 (秒)", color="#9fd4b0", fontsize=9)
        self.ax.set_ylabel("Mel 频带", color="#9fd4b0", fontsize=9)
        self.ax.set_title("Mel 频谱图", color="#c8e6c9", fontsize=10, pad=6)
        if not hasattr(self, "_cbar") or self._cbar is None:
            self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.025, pad=0.02)
            self._cbar.ax.yaxis.set_tick_params(color="#9fd4b0", labelcolor="#9fd4b0")
        else:
            self._cbar.update_normal(im)
        self.fig.tight_layout(pad=0.8)
        self.draw()


class AnimatedBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 1000)
        self.setTextVisible(False)
        self._target = 0
        self._anim = QPropertyAnimation(self, b"animatedValue")
        self._anim.setDuration(900)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def getAnimatedValue(self) -> int:
        return self.value()

    def setAnimatedValue(self, v: int):
        self.setValue(v)

    animatedValue = Property(int, getAnimatedValue, setAnimatedValue)

    def animate_to(self, prob: float):
        self._target = int(prob * 1000)
        self._anim.stop()
        self._anim.setStartValue(self.value())
        self._anim.setEndValue(self._target)
        self._anim.start()


class ResultCard(QFrame):
    def __init__(self, rank: int, parent=None):
        super().__init__(parent)
        self.setObjectName("Top1Card" if rank == 1 else "ResultCard")
        if rank == 1:
            self.setStyleSheet(TOP1_CARD_STYLE)
        else:
            alpha = 0.05 + max(0, (6 - rank) * 0.015)
            self.setStyleSheet(
                RESULT_CARD_STYLE.format(alpha=alpha, border_alpha=0.12)
            )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24 if rank == 1 else 14)
        shadow.setOffset(0, 4)
        shadow.setColor(Qt.black if rank == 1 else Qt.darkGray)
        self.setGraphicsEffect(shadow)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)

        self.rank_label = QLabel(f"#{rank}")
        self.rank_label.setFixedWidth(36)
        self.rank_label.setAlignment(Qt.AlignCenter)
        rank_color = "#ffd54f" if rank == 1 else "#80cbc4"
        self.rank_label.setStyleSheet(
            f"font-size: {'22' if rank == 1 else '16'}px; font-weight: 700; color: {rank_color};"
        )

        text_col = QVBoxLayout()
        self.name_label = QLabel("—")
        self.name_label.setStyleSheet(
            f"font-size: {'18' if rank == 1 else '14'}px; font-weight: 600; color: #f1f8e9;"
        )
        self.sci_label = QLabel("")
        self.sci_label.setStyleSheet("font-size: 11px; color: #a5d6a7; font-style: italic;")
        self.code_label = QLabel("")
        self.code_label.setStyleSheet("font-size: 10px; color: #78909c;")
        text_col.addWidget(self.name_label)
        text_col.addWidget(self.sci_label)
        text_col.addWidget(self.code_label)

        right = QVBoxLayout()
        self.pct_label = QLabel("0%")
        self.pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.pct_label.setStyleSheet(
            f"font-size: {'26' if rank == 1 else '16'}px; font-weight: 700; color: #69f0ae;"
        )
        self.bar = AnimatedBar()
        self.bar.setFixedHeight(12 if rank == 1 else 8)
        right.addWidget(self.pct_label)
        right.addWidget(self.bar)

        layout.addWidget(self.rank_label)
        layout.addLayout(text_col, stretch=1)
        layout.addLayout(right, stretch=1)

    def set_result(self, common_name: str, scientific: str, code: str, prob: float):
        self.name_label.setText(common_name)
        self.sci_label.setText(scientific or " ")
        self.code_label.setText(f"ID: {code}")
        self.pct_label.setText(f"{prob * 100:.1f}%")
        self.bar.animate_to(prob)


def glass_panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("GlassPanel")
    shadow = QGraphicsDropShadowEffect(frame)
    shadow.setBlurRadius(20)
    shadow.setOffset(0, 3)
    frame.setGraphicsEffect(shadow)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    header = QLabel(title)
    header.setFont(QFont("Segoe UI", 12, QFont.Bold))
    header.setStyleSheet("color: #c5e1a5; margin-bottom: 4px;")
    layout.addWidget(header)
    return frame, layout
