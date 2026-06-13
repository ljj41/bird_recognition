"""应用图标加载工具。"""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtGui import QIcon
except ImportError:
    from PyQt5.QtGui import QIcon

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def app_icon_path() -> Path | None:
    for name in ("app_icon.ico", "app_icon.png"):
        path = ASSETS_DIR / name
        if path.exists():
            return path
    return None


def load_app_icon() -> QIcon | None:
    path = app_icon_path()
    if path is None:
        return None
    icon = QIcon(str(path))
    return icon if not icon.isNull() else None
