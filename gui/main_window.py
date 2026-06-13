#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""主窗口 — 鸟类声音识别 Stacking 集成 GUI。"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

try:
    from PySide6.QtCore import QThread, QTimer, Qt, Signal
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

from .icon_utils import load_app_icon
from .styles import APP_STYLE
from .widgets import DropZone, MplCanvas, ResultCard, glass_panel


class PredictWorker(QThread):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, predictor, audio_path: str):
        super().__init__()
        self.predictor = predictor
        self.audio_path = audio_path

    def run(self):
        try:
            result = self.predictor.predict(self.audio_path, top_k=5)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, predictor, project_root: Path):
        super().__init__()
        self.predictor = predictor
        self.project_root = project_root
        self._worker: PredictWorker | None = None
        self._record_path: Path | None = None
        self._recording = False

        self.setWindowTitle("BirdCLEF · 鸟类声音智能识别系统")
        icon = load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.setMinimumSize(1280, 820)
        self.resize(1400, 900)

        root = QWidget()
        root.setObjectName("RootWidget")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(22, 18, 22, 12)
        main.setSpacing(14)

        main.addLayout(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addLayout(self._build_left_panel(), stretch=3)
        body.addLayout(self._build_right_panel(), stretch=2)
        main.addLayout(body, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪 — 请选择或拖入音频文件")

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        col = QVBoxLayout()
        title = QLabel("🦜 BirdCLEF 鸟类声音识别")
        title.setObjectName("TitleLabel")
        sub = QLabel("Stacking 深度学习 + 机器学习融合 · 实时 Mel 频谱分析 · Top-5 物种预测")
        sub.setObjectName("SubtitleLabel")
        col.addWidget(title)
        col.addWidget(sub)
        row.addLayout(col, stretch=1)

        metrics = self.predictor.metrics
        label = getattr(self.predictor, "model_label", "模型")
        badge_text = (
            f"🏆 {label}  |  "
            f"Acc {metrics.get('accuracy', 0)*100:.1f}%  |  "
            f"F1 {metrics.get('macro_f1', 0)*100:.1f}%  |  "
            f"Top-5 {metrics.get('top_5_acc', 0)*100:.1f}%"
        )
        badge = QLabel(badge_text)
        badge.setObjectName("BadgeLabel")
        badge.setAlignment(Qt.AlignCenter)
        row.addWidget(badge, alignment=Qt.AlignRight | Qt.AlignVCenter)
        return row

    def _build_left_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        input_panel, input_layout = glass_panel("📥 音频输入")
        self.drop_zone = DropZone()
        self.drop_zone.fileDropped.connect(self._on_audio_selected)
        input_layout.addWidget(self.drop_zone)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("📂  选择音频文件")
        self.btn_open.clicked.connect(self._browse_file)
        self.btn_demo = QPushButton("🎬  加载演示样本")
        self.btn_demo.setObjectName("AccentButton")
        self.btn_demo.clicked.connect(self._load_demo_sample)
        self.btn_predict = QPushButton("🚀  开始识别")
        self.btn_predict.setObjectName("AccentButton")
        self.btn_predict.clicked.connect(self._run_predict)
        self.btn_record = QPushButton("🎙  录音 5 秒")
        self.btn_record.setObjectName("RecordButton")
        self.btn_record.clicked.connect(self._toggle_record)
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_demo)
        btn_row.addWidget(self.btn_record)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_predict)
        input_layout.addLayout(btn_row)

        self.file_label = QLabel("当前文件: 未选择")
        self.file_label.setStyleSheet("color: #90a4ae; font-size: 12px;")
        input_layout.addWidget(self.file_label)
        col.addWidget(input_panel)

        wave_panel, wave_layout = glass_panel("📊 波形可视化")
        self.wave_canvas = MplCanvas(width=6, height=2.0)
        wave_layout.addWidget(self.wave_canvas)
        col.addWidget(wave_panel)

        mel_panel, mel_layout = glass_panel("🌈 Mel 频谱图")
        self.mel_canvas = MplCanvas(width=6, height=2.4)
        mel_layout.addWidget(self.mel_canvas)
        col.addWidget(mel_panel, stretch=1)

        self._current_audio: str | None = None
        return col

    def _build_right_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        result_panel, result_layout = glass_panel("🏅 识别结果 Top-5")

        self.result_cards = [ResultCard(i + 1) for i in range(5)]
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(10)
        for card in self.result_cards:
            inner_layout.addWidget(card)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        result_layout.addWidget(scroll, stretch=1)

        self.summary_label = QLabel("等待识别…")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "font-size: 13px; color: #b2dfdb; padding: 8px; "
            "background: rgba(0,0,0,0.2); border-radius: 10px;"
        )
        result_layout.addWidget(self.summary_label)
        col.addWidget(result_panel, stretch=1)

        fusion_panel, fusion_layout = glass_panel("🔬 模型融合详情")
        self.fusion_label = QLabel(
            f"DL 组件: {self.predictor.manifest.get('dl_component', 'cnn').upper()}  |  "
            f"ML 组件: {self.predictor.manifest.get('ml_component', 'svm').upper()}\n"
            "Stacking 元学习器将 CNN 与 SVM 概率向量拼接后，"
            "由 Logistic Regression 学习最优融合权重。"
        )
        self.fusion_label.setWordWrap(True)
        self.fusion_label.setStyleSheet("color: #a5d6a7; font-size: 12px; line-height: 1.5;")
        fusion_layout.addWidget(self.fusion_label)
        col.addWidget(fusion_panel)
        return col

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            str(self.project_root),
            "音频 (*.wav *.ogg *.mp3 *.flac *.m4a);;全部 (*.*)",
        )
        if path:
            self._on_audio_selected(path)

    def _find_demo_audio(self) -> str | None:
        audio_root = self.project_root / "train_audio"
        if not audio_root.exists():
            return None
        for ext in ("*.ogg", "*.wav", "*.mp3", "*.flac"):
            files = sorted(audio_root.rglob(ext))
            if files:
                return str(files[0])
        return None

    def _load_demo_sample(self):
        demo = self._find_demo_audio()
        if demo:
            self._on_audio_selected(demo)
            self.status.showMessage(f"已加载演示样本: {Path(demo).name}")
        else:
            QMessageBox.information(
                self, "提示",
                "未在 train_audio/ 下找到演示音频，请手动选择文件。",
            )

    def _on_audio_selected(self, path: str):
        self._current_audio = path
        self.file_label.setText(f"当前文件: {Path(path).name}")
        self.drop_zone.hint.setText(f"已选择: {Path(path).name}\n点击「开始识别」运行 Stacking 模型")
        self.status.showMessage(f"已选择: {path}")

    def _toggle_record(self):
        if self._recording:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            QMessageBox.warning(
                self, "录音不可用",
                "请安装: pip install sounddevice soundfile",
            )
            return

        sr = self.predictor.sample_rate
        duration = self.predictor.duration
        self._recording = True
        self.btn_record.setText("⏺  录音中…")
        self.btn_record.setProperty("recording", True)
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)
        self.status.showMessage(f"正在录音 {duration:.0f} 秒…")
        QApplication.processEvents()

        try:
            audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            tmp = Path(tempfile.gettempdir()) / "bird_gui_record.wav"
            sf.write(tmp, audio, sr)
            self._on_audio_selected(str(tmp))
            self.status.showMessage("录音完成，可点击开始识别")
        except Exception as e:
            QMessageBox.critical(self, "录音失败", str(e))
        finally:
            self._recording = False
            self.btn_record.setText("🎙  录音 5 秒")
            self.btn_record.setProperty("recording", False)
            self.btn_record.style().unpolish(self.btn_record)
            self.btn_record.style().polish(self.btn_record)

    def _run_predict(self):
        if not self._current_audio:
            QMessageBox.warning(self, "提示", "请先选择或录制音频。")
            return
        if self._worker and self._worker.isRunning():
            return

        self.btn_predict.setEnabled(False)
        self.btn_predict.setText("⏳  识别中…")
        self.status.showMessage("Stacking 模型推理中…")
        self._worker = PredictWorker(self.predictor, self._current_audio)
        self._worker.finished.connect(self._on_predict_done)
        self._worker.failed.connect(self._on_predict_failed)
        self._worker.start()

    def _on_predict_done(self, result):
        self.btn_predict.setEnabled(True)
        self.btn_predict.setText("🚀  开始识别")

        self.wave_canvas.plot_waveform(result.waveform, result.sample_rate)
        self.mel_canvas.plot_mel(result.mel, result.sample_rate)

        for i, card in enumerate(self.result_cards):
            if i < len(result.items):
                item = result.items[i]
                card.set_result(
                    item.species.common_name,
                    item.species.scientific_name,
                    item.species.code,
                    item.probability,
                )
                card.show()
            else:
                card.hide()

        top = result.items[0]
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setText(
            f"✨ 最可能物种: <b>{top.species.common_name}</b> "
            f"({top.species.scientific_name})<br>"
            f"置信度 <b>{top.probability*100:.1f}%</b>  |  "
            f"支持 {len(self.predictor.species)} 类鸟种分类"
        )
        self.status.showMessage(
            f"识别完成 — {top.species.common_name} ({top.probability*100:.1f}%)"
        )

    def _on_predict_failed(self, tb: str):
        self.btn_predict.setEnabled(True)
        self.btn_predict.setText("🚀  开始识别")
        self.status.showMessage("识别失败")
        QMessageBox.critical(self, "识别错误", tb)


def run_app(predictor, project_root: Path):
    app = QApplication.instance() or QApplication([])
    icon = load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    win = MainWindow(predictor, project_root)
    win.show()
    return app.exec()
