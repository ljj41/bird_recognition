"""BirdCLEF GUI dark nature theme."""

APP_STYLE = """
QMainWindow, QWidget#RootWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #0a1628, stop:0.45 #0f2b2a, stop:1 #1a1a2e);
    color: #e8f5e9;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
}

QLabel { color: #dceee0; background: transparent; }

QLabel#TitleLabel {
    font-size: 28px;
    font-weight: 700;
    color: #f4ffd8;
}

QLabel#SubtitleLabel {
    font-size: 13px;
    color: #9fd4b0;
}

QLabel#BadgeLabel {
    background: rgba(255, 213, 79, 0.18);
    border: 1px solid rgba(255, 213, 79, 0.55);
    border-radius: 14px;
    padding: 6px 14px;
    color: #ffe082;
    font-weight: 600;
}

QFrame#GlassPanel {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
}

QFrame#DropZone {
    background: rgba(76, 175, 80, 0.08);
    border: 2px dashed rgba(129, 199, 132, 0.55);
    border-radius: 16px;
}

QFrame#DropZone[dragActive="true"] {
    background: rgba(76, 175, 80, 0.22);
    border: 2px solid #81c784;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2e7d32, stop:1 #43a047);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #388e3c, stop:1 #66bb6a);
}

QPushButton:pressed { background: #1b5e20; }

QPushButton#AccentButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #f9a825, stop:1 #ffb300);
    color: #1a1a2e;
}

QPushButton#RecordButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #c62828, stop:1 #e53935);
    min-width: 120px;
}

QPushButton#RecordButton[recording="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff6f00, stop:1 #ff9100);
}

QProgressBar {
    background: rgba(255,255,255,0.08);
    border: none;
    border-radius: 8px;
    height: 14px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #26a69a, stop:1 #69f0ae);
}

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

QStatusBar {
    background: rgba(0,0,0,0.35);
    color: #a5d6a7;
    border-top: 1px solid rgba(255,255,255,0.08);
}
"""

RESULT_CARD_STYLE = """
QFrame#ResultCard {{
    background: rgba(255,255,255,{alpha});
    border: 1px solid rgba(255,255,255,{border_alpha});
    border-radius: 14px;
}}
"""

TOP1_CARD_STYLE = """
QFrame#Top1Card {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(255, 213, 79, 0.22), stop:1 rgba(76, 175, 80, 0.18));
    border: 2px solid rgba(255, 213, 79, 0.65);
    border-radius: 18px;
}
"""
