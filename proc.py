#!/usr/bin/env python3
"""
PL Spectrum Stitcher / Converter — entry point.
Run:  python proc.py
"""
import sys
from PyQt5.QtWidgets import QApplication
from app import StitchApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PL Stitcher")
    app.setStyle("Fusion")
    window = StitchApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
