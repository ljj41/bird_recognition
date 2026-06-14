#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动鸟类声音识别 GUI（PySide6 / PyQt5）。

首次使用前请导出 Stacking 部署包:
  python bird_recognition\\scripts\\export_stacking_bundle.py

启动界面:
  python bird_recognition\\scripts\\run_gui.py

依赖:
  pip install PySide6 matplotlib sounddevice soundfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import setup_console  # noqa: F401

from gui.mpl_fonts import setup_chinese_matplotlib

setup_chinese_matplotlib()


def parse_args():
    p = argparse.ArgumentParser(description="启动鸟类声音识别 GUI")
    p.add_argument(
        "--bundle",
        default="stacking_top37",
        help="部署包目录名，位于 outputs/models/deploy/ 下 "
             "(stacking_top37=37类Stacking, stitch_top37=37类Stitch, stacking_top10=10类)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
        except ImportError:
            print("请先安装 GUI 依赖: pip install PySide6 matplotlib")
            sys.exit(1)

    from gui.main_window import run_app

    bundle = PROJECT_ROOT / "outputs" / "models" / "deploy" / args.bundle
    if not bundle.exists():
        app = QApplication(sys.argv)
        if args.bundle.startswith("stacking"):
            hint = (
                "未找到 Stacking 部署包，请先在 Anaconda Prompt 中运行:\n\n"
                "python bird_recognition\\scripts\\export_stacking_bundle.py "
                "--config bird_recognition\\configs\\compare.yaml"
            )
        elif args.bundle == "stitch_top37":
            hint = (
                "未找到 Stitch 部署包，请先在 Anaconda Prompt 中运行:\n\n"
                "python bird_recognition\\scripts\\export_stitch_bundle.py"
            )
        else:
            hint = (
                f"未找到部署包 {args.bundle}，请先导出，例如:\n\n"
                "python bird_recognition\\scripts\\export_stacking_bundle.py\n"
                "或\n"
                "python bird_recognition\\scripts\\export_stitch_bundle.py"
            )
        QMessageBox.critical(None, "缺少模型包", hint)
        sys.exit(1)

    manifest_path = bundle / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        import json
        manifest = json.load(f)

    if manifest.get("model_type") == "stacking":
        from src.inference.stacking_predictor import StackingBirdPredictor
        print("正在加载 Stacking 模型…")
        predictor = StackingBirdPredictor(bundle)
    else:
        from src.inference.dl_predictor import DLBirdPredictor
        print(f"正在加载 {manifest.get('model_name', 'dl').upper()} 模型…")
        predictor = DLBirdPredictor(bundle)

    print(f"  {predictor.model_label}")
    print(f"  支持 {len(predictor.species)} 种鸟类")
    sys.exit(run_app(predictor, PROJECT_ROOT))


if __name__ == "__main__":
    main()
