#!/usr/bin/env python3
"""
PL Spectrum Stitcher / Converter — entry point.
Run:  python stitchcraft.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from qt_material import apply_stylesheet
from app import StitchApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PL Stitcher")
    apply_stylesheet(app, theme="dark_teal.xml")
    window = StitchApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
