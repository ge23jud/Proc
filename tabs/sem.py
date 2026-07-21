import csv
import os
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QCheckBox, QComboBox, QLineEdit,
    QListWidget, QListWidgetItem, QAbstractItemView, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QScrollArea, QStyle,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import PGCanvas, make_pg_toolbar
from sem_utils import (
    read_sem_image, detect_wires, crop_around, detect_best_wire_in_crop,
    measure_two_points,
)

pg.setConfigOptions(imageAxisOrder='row-major')

_MEASURE_TYPES = ["Length", "Bottom diameter", "Top diameter", "Droplet diameter", "Distance"]
_DIST_TYPES = {"Manual", "Semi-auto (manual)"}   # row types whose "Value 3" is a distance, not an aspect ratio


class SemTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── SEM state ─────────────────────────────────────────────
        self._sem_images = []        # list of per-image dicts, see _sem_load_files
        self._sem_rows = []          # results-table backing rows, see _sem_refresh_table
        self._sem_current = -1       # index into _sem_images
        self._sem_mode = "auto"      # "auto" | "semiauto" | "manual"
        self._sem_manual_pts = []    # 0-2 pending (x, y) pixel points
        self._sem_manual_last = None  # last computed measure_two_points() result
        self._sem_pending_fallback = None  # index into current image's semi_clicks awaiting manual correction

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scrollbar_w = scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        scroll.setFixedWidth(300 + scrollbar_w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        sl.addWidget(self._sem_build_files_group())
        sl.addWidget(self._sem_build_mode_group())
        self._sem_g_auto = self._sem_build_auto_group()
        self._sem_g_semi = self._sem_build_semi_group()
        self._sem_g_manual = self._sem_build_manual_group()
        sl.addWidget(self._sem_g_auto)
        sl.addWidget(self._sem_g_semi)
        sl.addWidget(self._sem_g_manual)
        sl.addWidget(self._sem_build_results_group())
        sl.addStretch(1)

        scroll.setWidget(sidebar)
        layout.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._sem_canvas = PGCanvas(right, welcome_msg="Load a SEM image (.tif) to begin.")
        self._sem_canvas.sigDataClicked.connect(self._sem_on_canvas_click)
        self._sem_toolbar = make_pg_toolbar(self._sem_canvas, right)
        rl.addWidget(self._sem_toolbar)
        rl.addWidget(self._sem_canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        self._sem_on_mode_changed()

    # ── UI construction ─────────────────────────────────────────────

    def _sem_build_files_group(self):
        g = QGroupBox("Input images")
        gl = QVBoxLayout(g)
        self._sem_file_list = QListWidget()
        self._sem_file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._sem_file_list.currentRowChanged.connect(self._sem_on_select_image)
        gl.addWidget(self._sem_file_list)

        fb = QHBoxLayout()
        btn_add = QPushButton("Add…")
        btn_remove = QPushButton("Remove")
        btn_clear = QPushButton("Clear")
        btn_add.clicked.connect(self._sem_load_files)
        btn_remove.clicked.connect(self._sem_remove_image)
        btn_clear.clicked.connect(self._sem_clear_images)
        for b in (btn_add, btn_remove, btn_clear):
            fb.addWidget(b)
        gl.addLayout(fb)

        self._sem_info_lbl = QLabel("No image loaded")
        self._sem_info_lbl.setWordWrap(True)
        gl.addWidget(self._sem_info_lbl)

        sr = QHBoxLayout()
        sr.addWidget(QLabel("Scale (nm/px):"))
        self._sem_scale_edit = QLineEdit("")
        self._sem_scale_edit.setEnabled(False)
        self._sem_scale_edit.editingFinished.connect(self._sem_on_scale_edited)
        sr.addWidget(self._sem_scale_edit)
        gl.addLayout(sr)
        return g

    def _sem_build_mode_group(self):
        g = QGroupBox("Mode")
        gl = QVBoxLayout(g)
        self._sem_rb_auto = QRadioButton("Automated (whole image)")
        self._sem_rb_semi = QRadioButton("Semi-automated (click + detect)")
        self._sem_rb_manual = QRadioButton("Manual (2-click distance)")
        self._sem_rb_auto.setChecked(True)
        for rb, mode in ((self._sem_rb_auto, "auto"), (self._sem_rb_semi, "semiauto"),
                         (self._sem_rb_manual, "manual")):
            rb.toggled.connect(lambda checked, m=mode: checked and self._sem_set_mode(m))
            gl.addWidget(rb)
        return g

    def _sem_build_auto_group(self):
        g = QGroupBox("Automated settings")
        gl = QVBoxLayout(g)

        def row(label, default):
            hl = QHBoxLayout()
            hl.addWidget(QLabel(label))
            edit = QLineEdit(default)
            hl.addWidget(edit)
            gl.addLayout(hl)
            return edit

        self._sem_auto_threshold = row("Threshold (0-255):", "120")
        self._sem_auto_adaptive = QCheckBox("Adaptive threshold")
        gl.addWidget(self._sem_auto_adaptive)
        self._sem_auto_edge = row("Edge margin (px):", "4")
        self._sem_auto_minpts = row("Min boundary points:", "20")
        self._sem_auto_maxangle = row("Max tilt from vertical (°):", "5.0")
        self._sem_auto_skip = row("Skip IDs (comma-sep):", "")

        btn_preview = QPushButton("Preview binary mask")
        btn_preview.clicked.connect(self._sem_preview_mask)
        gl.addWidget(btn_preview)

        btn_detect = QPushButton("Detect")
        self._sem_style_primary(btn_detect)
        btn_detect.clicked.connect(self._sem_detect_current)
        gl.addWidget(btn_detect)

        btn_all = QPushButton("Detect all loaded images")
        btn_all.clicked.connect(self._sem_detect_all)
        gl.addWidget(btn_all)
        return g

    def _sem_build_semi_group(self):
        g = QGroupBox("Semi-automated settings")
        gl = QVBoxLayout(g)

        def row(label, default):
            hl = QHBoxLayout()
            hl.addWidget(QLabel(label))
            edit = QLineEdit(default)
            hl.addWidget(edit)
            gl.addLayout(hl)
            return edit

        self._sem_semi_width_nm = row("Crop width (nm):", "2000")
        self._sem_semi_length_nm = row("Crop length (nm):", "40000")
        self._sem_semi_maxangle = row("Max tilt from vertical (°):", "5.0")
        self._sem_semi_minheight = row("Min detected height (nm):", "10")

        gl.addWidget(QLabel(
            "Click near each wire to queue it, then press Detect.\n"
            "If auto-detection fails for a click, you'll be asked to "
            "click 2 points manually within the highlighted crop."
        ))
        self._sem_semi_status = QLabel("")
        self._sem_semi_status.setWordWrap(True)
        gl.addWidget(self._sem_semi_status)

        btn_detect = QPushButton("Detect clicked wires")
        self._sem_style_primary(btn_detect)
        btn_detect.clicked.connect(self._sem_semi_detect_clicked)
        gl.addWidget(btn_detect)

        btn_clear = QPushButton("Clear clicks")
        btn_clear.clicked.connect(self._sem_semi_clear_clicks)
        gl.addWidget(btn_clear)
        return g

    def _sem_build_manual_group(self):
        g = QGroupBox("Manual settings")
        gl = QVBoxLayout(g)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Measuring:"))
        self._sem_manual_type = QComboBox()
        self._sem_manual_type.addItems(_MEASURE_TYPES)
        hl.addWidget(self._sem_manual_type)
        gl.addLayout(hl)

        gl.addWidget(QLabel("Click two points on the image."))
        self._sem_manual_readout = QLabel("")
        self._sem_manual_readout.setWordWrap(True)
        gl.addWidget(self._sem_manual_readout)

        br = QHBoxLayout()
        self._sem_manual_accept = QPushButton("Accept")
        self._sem_manual_accept.setEnabled(False)
        self._sem_style_primary(self._sem_manual_accept)
        self._sem_manual_accept.clicked.connect(self._sem_manual_accept_measurement)
        btn_redo = QPushButton("Redo")
        btn_redo.clicked.connect(self._sem_manual_redo)
        br.addWidget(self._sem_manual_accept)
        br.addWidget(btn_redo)
        gl.addLayout(br)
        return g

    def _sem_build_results_group(self):
        g = QGroupBox("Results")
        gl = QVBoxLayout(g)
        self._sem_table = QTableWidget(0, 7)
        self._sem_table.setHorizontalHeaderLabels(
            ["Image", "Type", "Label", "Value 1 (nm)", "Value 2 (nm)", "Value 3", "Angle (°)"]
        )
        self._sem_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._sem_table.setMinimumHeight(160)
        gl.addWidget(self._sem_table)

        rb = QHBoxLayout()
        btn_remove = QPushButton("Remove selected")
        btn_clear = QPushButton("Clear all")
        btn_remove.clicked.connect(self._sem_remove_selected_rows)
        btn_clear.clicked.connect(self._sem_clear_rows)
        rb.addWidget(btn_remove)
        rb.addWidget(btn_clear)
        gl.addLayout(rb)

        btn_export = QPushButton("Export CSV…")
        btn_export.clicked.connect(self._sem_export_csv)
        gl.addWidget(btn_export)

        self._sem_stats_lbl = QLabel("")
        self._sem_stats_lbl.setWordWrap(True)
        gl.addWidget(self._sem_stats_lbl)
        return g

    @staticmethod
    def _sem_style_primary(button):
        bold = QFont()
        bold.setBold(True)
        button.setFont(bold)
        button.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )

    # ── File / image management ──────────────────────────────────────

    def _sem_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select SEM images", "", "SEM images (*.tif *.tiff);;All files (*.*)"
        )
        if not paths:
            return
        failed = []
        for path in paths:
            try:
                image, scale = read_sem_image(path)
            except Exception as exc:
                failed.append(f"{os.path.basename(path)}: {exc}")
                continue
            data = {
                "path": path,
                "label": os.path.basename(path),
                "image": image,
                "scale": scale,
                "auto_results": None,
                "semi_clicks": [],
            }
            self._sem_images.append(data)
            item = QListWidgetItem(data["label"])
            item.setToolTip(path)
            self._sem_file_list.addItem(item)
        if failed:
            QMessageBox.warning(self, "Some images failed to load", "\n".join(failed))
        if self._sem_current < 0 and self._sem_images:
            self._sem_file_list.setCurrentRow(0)

    def _sem_remove_image(self):
        row = self._sem_file_list.currentRow()
        if row < 0:
            return
        self._sem_file_list.takeItem(row)
        self._sem_images.pop(row)
        self._sem_rows_for_image_removed(row)
        if not self._sem_images:
            self._sem_current = -1
            self._sem_info_lbl.setText("No image loaded")
            self._sem_scale_edit.setText("")
            self._sem_scale_edit.setEnabled(False)

    def _sem_clear_images(self):
        self._sem_images.clear()
        self._sem_file_list.clear()
        self._sem_rows = []
        self._sem_current = -1
        self._sem_info_lbl.setText("No image loaded")
        self._sem_scale_edit.setText("")
        self._sem_scale_edit.setEnabled(False)
        self._sem_refresh_table()
        self._sem_redraw()

    def _sem_rows_for_image_removed(self, removed_idx):
        """Drop table rows belonging to a removed image and shift image_idx
        references down for every image after it."""
        kept = []
        for r in self._sem_rows:
            if r["image_idx"] == removed_idx:
                continue
            if r["image_idx"] > removed_idx:
                r = dict(r, image_idx=r["image_idx"] - 1)
            kept.append(r)
        self._sem_rows = kept
        self._sem_refresh_table()

    def _sem_on_select_image(self, row):
        self._sem_current = row
        if row < 0 or row >= len(self._sem_images):
            return
        data = self._sem_images[row]
        h, w = data["image"].shape
        scale_txt = f"{data['scale'] * 1e9:.4g}" if data["scale"] else ""
        self._sem_info_lbl.setText(
            f"{data['label']}\n{w} x {h} px"
            + (f"  ({data['scale'] * 1e9:.4g} nm/px)" if data["scale"] else "  (no scale found — enter manually)")
        )
        self._sem_scale_edit.setEnabled(True)
        self._sem_scale_edit.setText(scale_txt)
        self._sem_manual_pts = []
        self._sem_manual_last = None
        self._sem_manual_readout.setText("")
        self._sem_manual_accept.setEnabled(False)
        self._sem_pending_fallback = None
        self._sem_redraw()

    def _sem_on_scale_edited(self):
        if self._sem_current < 0:
            return
        try:
            nm_per_px = float(self._sem_scale_edit.text().strip())
            self._sem_images[self._sem_current]["scale"] = nm_per_px * 1e-9
        except ValueError:
            QMessageBox.warning(self, "Invalid scale", "Enter a numeric nm/px value.")

    def _sem_current_scale(self):
        if self._sem_current < 0:
            return None
        return self._sem_images[self._sem_current]["scale"]

    # ── Mode switching ────────────────────────────────────────────

    def _sem_set_mode(self, mode):
        self._sem_mode = mode
        self._sem_manual_pts = []
        self._sem_manual_last = None
        self._sem_manual_readout.setText("")
        self._sem_manual_accept.setEnabled(False)
        self._sem_pending_fallback = None
        self._sem_semi_status.setText("")
        self._sem_on_mode_changed()
        self._sem_redraw()

    def _sem_on_mode_changed(self):
        self._sem_g_auto.setVisible(self._sem_mode == "auto")
        self._sem_g_semi.setVisible(self._sem_mode == "semiauto")
        self._sem_g_manual.setVisible(self._sem_mode == "manual")

    # ── Automated detection ────────────────────────────────────────

    def _sem_read_auto_params(self):
        try:
            threshold = int(float(self._sem_auto_threshold.text()))
            edge_margin = int(float(self._sem_auto_edge.text()))
            min_pts = int(float(self._sem_auto_minpts.text()))
            max_angle = float(self._sem_auto_maxangle.text())
            skip_txt = self._sem_auto_skip.text().strip()
            skip_ids = {int(s) for s in skip_txt.split(",") if s.strip()} if skip_txt else set()
        except ValueError:
            QMessageBox.warning(self, "Invalid parameter", "Check the automated-detection settings.")
            return None
        return {
            "threshold": threshold,
            "method": "adaptive" if self._sem_auto_adaptive.isChecked() else "global",
            "edge_margin": edge_margin,
            "min_boundary_points": min_pts,
            "max_angle_deg": max_angle,
            "skip_ids": skip_ids,
        }

    def _sem_preview_mask(self):
        if self._sem_current < 0:
            return
        params = self._sem_read_auto_params()
        if params is None:
            return
        from sem_utils import _binarize
        from scipy import ndimage
        image = self._sem_images[self._sem_current]["image"]
        bw = _binarize(image, method=params["method"], threshold=params["threshold"])
        filled = ndimage.binary_fill_holes(bw)
        ax = self._sem_canvas.reset_axes()
        img_item = pg.ImageItem((filled.astype(np.uint8)) * 255)
        ax.addItem(img_item)
        ax.invertY(True)
        ax.setAspectLocked(True)
        ax.setTitle("Binary mask preview — adjust threshold, then Detect")
        self._sem_canvas.draw_idle()

    def _sem_detect_current(self):
        if self._sem_current < 0:
            QMessageBox.warning(self, "No image", "Select an image first.")
            return
        params = self._sem_read_auto_params()
        if params is None:
            return
        self._sem_run_auto_detect(self._sem_current, params)
        self._sem_redraw()
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_detect_all(self):
        if not self._sem_images:
            return
        params = self._sem_read_auto_params()
        if params is None:
            return
        for idx in range(len(self._sem_images)):
            self._sem_run_auto_detect(idx, params)
        self._sem_redraw()
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_run_auto_detect(self, idx, params):
        data = self._sem_images[idx]
        scale = data["scale"]
        if not scale:
            QMessageBox.warning(self, "No scale", f"{data['label']}: enter a scale (nm/px) first.")
            return
        results = detect_wires(data["image"], scale, **params)
        data["auto_results"] = results
        self._sem_rows = [r for r in self._sem_rows
                          if not (r["image_idx"] == idx and r["type"] == "Auto")]
        for entry in results:
            if not entry["accepted"]:
                continue
            self._sem_rows.append({
                "image_idx": idx, "image_label": data["label"], "type": "Auto",
                "label": str(entry["id"]), "val1_m": entry["width_m"], "val2_m": entry["height_m"],
                "val3": entry["aspect_ratio"], "angle_deg": entry["angle_deg"],
            })

    # ── Semi-automated detection ───────────────────────────────────

    def _sem_read_semi_params(self):
        try:
            width_nm = float(self._sem_semi_width_nm.text())
            length_nm = float(self._sem_semi_length_nm.text())
            max_angle = float(self._sem_semi_maxangle.text())
            min_height_nm = float(self._sem_semi_minheight.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid parameter", "Check the semi-automated settings.")
            return None
        return width_nm, length_nm, max_angle, min_height_nm

    def _sem_semi_detect_clicked(self):
        if self._sem_current < 0:
            return
        params = self._sem_read_semi_params()
        if params is None:
            return
        width_nm, length_nm, max_angle, min_height_nm = params
        data = self._sem_images[self._sem_current]
        scale = data["scale"]
        if not scale:
            QMessageBox.warning(self, "No scale", "Enter a scale (nm/px) first.")
            return

        for i, click in enumerate(data["semi_clicks"]):
            if click["status"] != "pending":
                continue
            crop, (x0, y0) = crop_around(data["image"], scale, click["center_px"], (width_nm, length_nm))
            best = detect_best_wire_in_crop(
                crop, scale, min_height_m=min_height_nm * 1e-9,
                max_angle_deg=max_angle, edge_margin=4, min_boundary_points=20,
            )
            if best is None:
                click["status"] = "awaiting_manual"
                h, w = crop.shape
                click["crop_rect"] = (x0, y0, x0 + w, y0 + h)
                continue
            best = dict(best)
            best["center_px"] = (best["center_px"][0] + x0, best["center_px"][1] + y0)
            best["box_points_px"] = best["box_points_px"] + np.array([x0, y0])
            click["status"] = "detected"
            click["result"] = best
            self._sem_rows.append({
                "image_idx": self._sem_current, "image_label": data["label"], "type": "Semi-auto",
                "label": f"click {i + 1}", "val1_m": best["width_m"], "val2_m": best["height_m"],
                "val3": best["aspect_ratio"], "angle_deg": best["angle_deg"],
            })

        self._sem_advance_fallback_queue()
        self._sem_redraw()
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_advance_fallback_queue(self):
        data = self._sem_images[self._sem_current]
        pending = [i for i, c in enumerate(data["semi_clicks"]) if c["status"] == "awaiting_manual"]
        if pending:
            self._sem_pending_fallback = pending[0]
            self._sem_semi_status.setText(
                f"Auto-detection failed for click {pending[0] + 1} — "
                "click 2 points inside the dashed crop to measure it manually."
            )
        else:
            self._sem_pending_fallback = None
            self._sem_semi_status.setText("")

    def _sem_semi_clear_clicks(self):
        if self._sem_current < 0:
            return
        data = self._sem_images[self._sem_current]
        data["semi_clicks"] = []
        self._sem_rows = [r for r in self._sem_rows
                          if not (r["image_idx"] == self._sem_current and r["type"].startswith("Semi-auto"))]
        self._sem_pending_fallback = None
        self._sem_semi_status.setText("")
        self._sem_redraw()
        self._sem_refresh_table()
        self._sem_refresh_stats()

    # ── Canvas click dispatch ───────────────────────────────────────

    def _sem_on_canvas_click(self, x, y, button):
        if button != Qt.LeftButton or self._sem_current < 0:
            return
        if self._sem_mode == "semiauto":
            self._sem_on_semi_click(x, y)
        elif self._sem_mode == "manual":
            self._sem_on_manual_click(x, y)

    def _sem_on_semi_click(self, x, y):
        data = self._sem_images[self._sem_current]
        if self._sem_pending_fallback is not None:
            click = data["semi_clicks"][self._sem_pending_fallback]
            pts = click.setdefault("manual_pts", [])
            pts.append((x, y))
            if len(pts) < 2:
                self._sem_redraw()
                return
            scale = data["scale"]
            meas = measure_two_points(pts[0], pts[1], scale)
            click["status"] = "manual"
            click["result"] = {"p1": pts[0], "p2": pts[1], **meas}
            idx = self._sem_pending_fallback
            self._sem_rows.append({
                "image_idx": self._sem_current, "image_label": data["label"], "type": "Semi-auto (manual)",
                "label": f"click {idx + 1}", "val1_m": meas["dx_m"], "val2_m": meas["dy_m"],
                "val3": meas["d_m"], "angle_deg": None,
            })
            self._sem_advance_fallback_queue()
            self._sem_redraw()
            self._sem_refresh_table()
            self._sem_refresh_stats()
            return

        data["semi_clicks"].append({"center_px": (x, y), "status": "pending"})
        self._sem_redraw()

    def _sem_on_manual_click(self, x, y):
        if len(self._sem_manual_pts) >= 2:
            self._sem_manual_pts = []
            self._sem_manual_last = None
            self._sem_manual_accept.setEnabled(False)
        self._sem_manual_pts.append((x, y))
        if len(self._sem_manual_pts) == 2:
            scale = self._sem_current_scale()
            if not scale:
                QMessageBox.warning(self, "No scale", "Enter a scale (nm/px) first.")
                self._sem_manual_pts = []
                self._sem_redraw()
                return
            meas = measure_two_points(self._sem_manual_pts[0], self._sem_manual_pts[1], scale)
            self._sem_manual_last = meas
            self._sem_manual_readout.setText(
                f"dx = {meas['dx_m'] * 1e9:.2f} nm\n"
                f"dy = {meas['dy_m'] * 1e9:.2f} nm\n"
                f"d = {meas['d_m'] * 1e9:.2f} nm"
            )
            self._sem_manual_accept.setEnabled(True)
        self._sem_redraw()

    def _sem_manual_accept_measurement(self):
        if self._sem_manual_last is None or self._sem_current < 0:
            return
        data = self._sem_images[self._sem_current]
        meas = self._sem_manual_last
        self._sem_rows.append({
            "image_idx": self._sem_current, "image_label": data["label"], "type": "Manual",
            "label": self._sem_manual_type.currentText(), "val1_m": meas["dx_m"], "val2_m": meas["dy_m"],
            "val3": meas["d_m"], "angle_deg": None,
        })
        self._sem_manual_pts = []
        self._sem_manual_last = None
        self._sem_manual_readout.setText("")
        self._sem_manual_accept.setEnabled(False)
        self._sem_redraw()
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_manual_redo(self):
        self._sem_manual_pts = []
        self._sem_manual_last = None
        self._sem_manual_readout.setText("")
        self._sem_manual_accept.setEnabled(False)
        self._sem_redraw()

    # ── Results table / stats / export ──────────────────────────────

    def _sem_refresh_table(self):
        rows = self._sem_rows
        self._sem_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            is_dist = row["type"] in _DIST_TYPES
            val1_txt = f"{row['val1_m'] * 1e9:.2f}" if row["val1_m"] is not None else ""
            val2_txt = f"{row['val2_m'] * 1e9:.2f}" if row["val2_m"] is not None else ""
            if row["val3"] is None:
                val3_txt = ""
            elif is_dist:
                val3_txt = f"{row['val3'] * 1e9:.2f} nm"
            else:
                val3_txt = f"{row['val3']:.3f}"
            angle_txt = f"{row['angle_deg']:.2f}" if row["angle_deg"] is not None else ""
            for c, text in enumerate([row["image_label"], row["type"], row["label"],
                                       val1_txt, val2_txt, val3_txt, angle_txt]):
                self._sem_table.setItem(r, c, QTableWidgetItem(text))

    def _sem_refresh_stats(self):
        rows = [r for r in self._sem_rows if r["type"] in ("Auto", "Semi-auto")]
        if not rows:
            self._sem_stats_lbl.setText("")
            return
        widths = np.array([r["val1_m"] for r in rows]) * 1e9
        heights = np.array([r["val2_m"] for r in rows]) * 1e9
        ars = np.array([r["val3"] for r in rows])
        self._sem_stats_lbl.setText(
            f"n = {len(rows)}\n"
            f"width  = {widths.mean():.2f} ± {widths.std():.2f} nm\n"
            f"height = {heights.mean():.2f} ± {heights.std():.2f} nm\n"
            f"aspect ratio = {ars.mean():.3f} ± {ars.std():.3f}"
        )

    def _sem_remove_selected_rows(self):
        selected = sorted({idx.row() for idx in self._sem_table.selectedIndexes()}, reverse=True)
        rows = self._sem_rows
        for r in selected:
            if 0 <= r < len(rows):
                rows.pop(r)
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_clear_rows(self):
        self._sem_rows = []
        self._sem_refresh_table()
        self._sem_refresh_stats()

    def _sem_export_csv(self):
        rows = self._sem_rows
        if not rows:
            QMessageBox.information(self, "No results", "There are no results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export results CSV", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Image", "Type", "Label", "Value 1 (nm)", "Value 2 (nm)", "Value 3", "Angle (deg)"])
            for row in rows:
                is_dist = row["type"] in _DIST_TYPES
                v1 = row["val1_m"] * 1e9 if row["val1_m"] is not None else ""
                v2 = row["val2_m"] * 1e9 if row["val2_m"] is not None else ""
                if row["val3"] is None:
                    v3 = ""
                elif is_dist:
                    v3 = row["val3"] * 1e9
                else:
                    v3 = row["val3"]
                writer.writerow([row["image_label"], row["type"], row["label"], v1, v2, v3, row["angle_deg"] or ""])

    # ── Canvas redraw ────────────────────────────────────────────────

    def _sem_redraw(self):
        if self._sem_current < 0 or self._sem_current >= len(self._sem_images):
            return
        data = self._sem_images[self._sem_current]
        ax = self._sem_canvas.reset_axes()
        image_item = pg.ImageItem(data["image"])
        ax.addItem(image_item)
        ax.invertY(True)
        ax.setAspectLocked(True)
        ax.setTitle(data["label"])

        if self._sem_mode == "auto":
            self._sem_draw_auto_overlay(ax, data)
        elif self._sem_mode == "semiauto":
            self._sem_draw_semi_overlay(ax, data)
        elif self._sem_mode == "manual":
            self._sem_draw_manual_overlay(ax)

        self._sem_canvas.draw_idle()

    def _sem_draw_auto_overlay(self, ax, data):
        results = data.get("auto_results")
        if not results:
            return
        for entry in results:
            pts = entry["contour_px"]
            xs = np.append(pts[:, 0], pts[0, 0])
            ys = np.append(pts[:, 1], pts[0, 1])
            color = (0, 200, 0) if entry["accepted"] else (110, 110, 110)
            ax.plot(xs, ys, pen=pg.mkPen(color, width=1))
            if entry["accepted"]:
                box = entry["box_points_px"]
                bx = np.append(box[:, 0], box[0, 0])
                by = np.append(box[:, 1], box[0, 1])
                ax.plot(bx, by, pen=pg.mkPen("r", width=2))
                cx, cy = entry["center_px"]
                label = pg.TextItem(str(entry["id"]), color="r", anchor=(0.5, 0.5))
                label.setPos(cx, cy)
                ax.addItem(label)

    def _sem_draw_semi_overlay(self, ax, data):
        for i, click in enumerate(data["semi_clicks"], start=1):
            cx, cy = click["center_px"]
            marker = pg.ScatterPlotItem(x=[cx], y=[cy], size=12, pen=pg.mkPen("y", width=2),
                                         brush=None, symbol="+")
            ax.addItem(marker)
            label = pg.TextItem(str(i), color="w", anchor=(0.5, 1.4))
            label.setPos(cx, cy)
            ax.addItem(label)

            status = click["status"]
            if status == "detected":
                box = click["result"]["box_points_px"]
                bx = np.append(box[:, 0], box[0, 0])
                by = np.append(box[:, 1], box[0, 1])
                ax.plot(bx, by, pen=pg.mkPen("r", width=2))
            elif status == "manual":
                p1, p2 = click["result"]["p1"], click["result"]["p2"]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], pen=pg.mkPen("c", width=2))
            elif status == "awaiting_manual":
                x0, y0, x1, y1 = click["crop_rect"]
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        pen=pg.mkPen("y", width=1, style=Qt.DashLine))
                for px, py in click.get("manual_pts", []):
                    m = pg.ScatterPlotItem(x=[px], y=[py], size=10, pen=pg.mkPen("c", width=2),
                                            brush=None, symbol="+")
                    ax.addItem(m)

    def _sem_draw_manual_overlay(self, ax):
        for px, py in self._sem_manual_pts:
            marker = pg.ScatterPlotItem(x=[px], y=[py], size=12, pen=pg.mkPen("c", width=2),
                                         brush=None, symbol="+")
            ax.addItem(marker)
        if len(self._sem_manual_pts) == 2:
            p1, p2 = self._sem_manual_pts
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], pen=pg.mkPen("c", width=2))
