import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QAbstractItemView, QFrame,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from io_utils import _read_h5_spectrum
from plot_utils import build_spectrum_figure, png_path_for


class PlotTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Plot state ────────────────────────────────────────────
        self._plot_files: list = []
        self._plot_data:  list = []   # {label, path, energy, counts, pump_fluence, power_mW}
        self._plot_canvas = None       # currently embedded FigureCanvasQTAgg, or None

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        g_files = QGroupBox("H5 files")
        fl = QVBoxLayout(g_files)
        self._plot_file_list = QListWidget()
        self._plot_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._plot_file_list.currentRowChanged.connect(self._plot_on_select)
        fl.addWidget(self._plot_file_list)
        fb = QHBoxLayout()
        btn_add    = QPushButton("Add…")
        btn_remove = QPushButton("Remove")
        btn_clear  = QPushButton("Clear")
        btn_add.clicked.connect(self._plot_on_add)
        btn_remove.clicked.connect(self._plot_on_remove)
        btn_clear.clicked.connect(self._plot_on_clear)
        for b in (btn_add, btn_remove, btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        sl.addWidget(g_files, stretch=1)

        self._plot_info_lbl = QLabel("Load one or more .h5 files.")
        self._plot_info_lbl.setWordWrap(True)
        sl.addWidget(self._plot_info_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._plot_btn_save = QPushButton("Save all as PNG")
        self._plot_btn_save.setEnabled(False)
        self._plot_btn_save.setMinimumHeight(32)
        bold = QFont()
        bold.setBold(True)
        self._plot_btn_save.setFont(bold)
        self._plot_btn_save.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._plot_btn_save.clicked.connect(self._plot_on_save_all)
        sl.addWidget(self._plot_btn_save)

        layout.addWidget(sidebar)

        # ── Right: matplotlib preview ───────────────────────────────
        right = QWidget()
        self._plot_right_layout = QVBoxLayout(right)
        self._plot_right_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_right_layout.setSpacing(4)
        self._plot_placeholder = QLabel("Select a file to preview its plot.")
        self._plot_placeholder.setAlignment(Qt.AlignCenter)
        self._plot_right_layout.addWidget(self._plot_placeholder, stretch=1)
        layout.addWidget(right, stretch=1)

    # ── File management ──────────────────────────────────────────

    def _plot_on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDF5 files", "", "HDF5 files (*.h5);;All files (*.*)"
        )
        added = 0
        errors = []
        for p in paths:
            if p in self._plot_files:
                continue
            try:
                data = _read_h5_spectrum(p)
            except Exception as exc:
                errors.append(f"{os.path.basename(p)}: {exc}")
                continue
            self._plot_files.append(p)
            self._plot_data.append(data)
            item = QListWidgetItem(data["label"])
            item.setToolTip(p)
            self._plot_file_list.addItem(item)
            added += 1
        if errors:
            QMessageBox.warning(self, "Some files failed to load", "\n".join(errors))
        if added:
            self._plot_btn_save.setEnabled(True)
            if self._plot_file_list.currentRow() < 0:
                self._plot_file_list.setCurrentRow(0)

    def _plot_on_remove(self):
        rows = sorted(
            [self._plot_file_list.row(i) for i in self._plot_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._plot_file_list.takeItem(row)
            self._plot_files.pop(row)
            self._plot_data.pop(row)
        if not self._plot_data:
            self._plot_btn_save.setEnabled(False)
            self._plot_clear_preview()
            self._plot_info_lbl.setText("Load one or more .h5 files.")
        elif self._plot_file_list.currentRow() < 0:
            self._plot_file_list.setCurrentRow(0)

    def _plot_on_clear(self):
        self._plot_files.clear()
        self._plot_data.clear()
        self._plot_file_list.clear()
        self._plot_btn_save.setEnabled(False)
        self._plot_clear_preview()
        self._plot_info_lbl.setText("Load one or more .h5 files.")

    # ── Preview ──────────────────────────────────────────────────

    def _plot_on_select(self, row):
        if row < 0 or row >= len(self._plot_data):
            return
        data = self._plot_data[row]
        fig, missing = build_spectrum_figure(data)
        self._plot_set_canvas(FigureCanvasQTAgg(fig))
        n_powers = data["counts"].shape[1]
        if missing:
            self._plot_info_lbl.setText(
                f"{data['label']}: {n_powers} power step(s). "
                "No Pump_fluence in file — legend shows power (mW) instead."
            )
        else:
            self._plot_info_lbl.setText(f"{data['label']}: {n_powers} power step(s).")

    def _plot_set_canvas(self, canvas):
        if self._plot_canvas is not None:
            self._plot_right_layout.removeWidget(self._plot_canvas)
            self._plot_canvas.setParent(None)
            self._plot_canvas.deleteLater()
        else:
            self._plot_right_layout.removeWidget(self._plot_placeholder)
            self._plot_placeholder.setParent(None)
        self._plot_canvas = canvas
        self._plot_right_layout.addWidget(canvas, stretch=1)

    def _plot_clear_preview(self):
        if self._plot_canvas is not None:
            self._plot_right_layout.removeWidget(self._plot_canvas)
            self._plot_canvas.setParent(None)
            self._plot_canvas.deleteLater()
            self._plot_canvas = None
            self._plot_right_layout.addWidget(self._plot_placeholder, stretch=1)

    # ── Save ─────────────────────────────────────────────────────

    def _plot_on_save_all(self):
        if not self._plot_data:
            return
        saved, failed, missing_fluence = [], [], []
        for data in self._plot_data:
            try:
                fig, missing = build_spectrum_figure(data)
                out_path = png_path_for(data["path"])
                fig.savefig(out_path, bbox_inches="tight")
                saved.append(out_path)
                if missing:
                    missing_fluence.append(data["label"])
            except Exception as exc:
                failed.append(f"{data['label']}: {exc}")

        lines = [f"Saved {len(saved)} PNG file(s):"] + [f"  {p}" for p in saved]
        if missing_fluence:
            lines += ["", "No Pump_fluence in (legend used power instead): "
                      + ", ".join(missing_fluence)]
        if failed:
            lines += ["", "Failed:"] + [f"  {m}" for m in failed]
        QMessageBox.information(self, "Save as PNG", "\n".join(lines))

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(f"Saved {len(saved)} PNG file(s).")
