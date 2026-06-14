"""Matplotlib 中文字体配置（GUI 波形 / Mel 谱图）。"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import matplotlib
from matplotlib import font_manager


def _register_font_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        font_manager.fontManager.addfont(str(path))
        prop = font_manager.FontProperties(fname=str(path))
        return prop.get_name()
    except Exception:
        return None


def _discover_chinese_font_family() -> str | None:
    system = platform.system()

    if system == "Windows":
        win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for filename, fallback in (
            ("msyh.ttc", "Microsoft YaHei"),
            ("msyhbd.ttc", "Microsoft YaHei"),
            ("simhei.ttf", "SimHei"),
            ("simsun.ttc", "SimSun"),
            ("msyhl.ttc", "Microsoft YaHei"),
        ):
            family = _register_font_file(win_fonts / filename)
            if family:
                return family
            if (win_fonts / filename).is_file():
                return fallback

    # macOS / Linux：尝试系统已安装字体名
    preferred_names = (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_names:
        if name in available:
            return name

    return None


def setup_chinese_matplotlib() -> str:
    """配置 matplotlib 使用中文字体，返回实际字体族名。"""
    if getattr(setup_chinese_matplotlib, "_configured", False):
        return setup_chinese_matplotlib._family  # type: ignore[attr-defined]

    family = _discover_chinese_font_family()
    if family:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans", "Arial"]
    else:
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "DejaVu Sans",
        ]

    matplotlib.rcParams["axes.unicode_minus"] = False

    setup_chinese_matplotlib._configured = True
    setup_chinese_matplotlib._family = family or "Microsoft YaHei"  # type: ignore[attr-defined]
    return setup_chinese_matplotlib._family  # type: ignore[attr-defined]
