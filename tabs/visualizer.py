import os
import sys
import numpy as np
import h5py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QCheckBox, QListWidget, QListWidgetItem,
    QLineEdit, QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView,
    QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, _COMPACT_BTN_STYLE, MultiLinePlotter, CategoricalScheme,
    SequentialScheme, LogAlphaRamp, PLASMA, make_pg_toolbar,
)

from pl import _HC_EV_NM


class VisualizerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Visualizer state ──────────────────────────────────────
        self._vis_files: list = []
        self._vis_data:  list = []   # {label, wl, counts, powers_W, path}

        # ── Build UI (body of original _build_vis_tab) ────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Sidebar ───────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Files")
        fl = QVBoxLayout(g_files)
        self._vis_file_list = QListWidget()
        self._vis_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fl.addWidget(self._vis_file_list)
        fb = QHBoxLayout()
        self._vis_btn_add    = QPushButton("Add…")
        self._vis_btn_remove = QPushButton("Remove")
        self._vis_btn_clear  = QPushButton("Clear")
        self._vis_btn_add.clicked.connect(self._vis_on_add)
        self._vis_btn_remove.clicked.connect(self._vis_on_remove)
        self._vis_btn_clear.clicked.connect(self._vis_on_clear)
        for b in (self._vis_btn_add, self._vis_btn_remove, self._vis_btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files)

        # Powers group
        g_powers = QGroupBox("Power steps")
        pl = QVBoxLayout(g_powers)
        self._vis_power_list = QListWidget()
        self._vis_power_list.setToolTip("Check power steps to display")
        pl.addWidget(self._vis_power_list, stretch=1)
        pb = QHBoxLayout()
        vis_btn_all  = QPushButton("All")
        vis_btn_none = QPushButton("None")
        vis_btn_all.clicked.connect(self._vis_check_all_powers)
        vis_btn_none.clicked.connect(self._vis_check_none_powers)
        pb.addWidget(vis_btn_all)
        pb.addWidget(vis_btn_none)
        pl.addLayout(pb)
        sl.addWidget(g_powers, stretch=1)

        # X axis group
        g_xaxis = QGroupBox("X axis")
        xl = QVBoxLayout(g_xaxis)
        xr = QHBoxLayout()
        self._vis_rb_energy = QRadioButton("Energy (eV)")
        self._vis_rb_wl     = QRadioButton("Wavelength (nm)")
        self._vis_rb_energy.setChecked(True)
        xr.addWidget(self._vis_rb_energy)
        xr.addWidget(self._vis_rb_wl)
        xl.addLayout(xr)
        self._vis_xauto = QCheckBox("Auto range")
        self._vis_xauto.setChecked(True)
        xl.addWidget(self._vis_xauto)
        xrl = QHBoxLayout()
        xrl.addWidget(QLabel("Min:"))
        self._vis_xmin = QLineEdit()
        self._vis_xmin.setEnabled(False)
        xrl.addWidget(self._vis_xmin)
        xrl.addWidget(QLabel("Max:"))
        self._vis_xmax = QLineEdit()
        self._vis_xmax.setEnabled(False)
        xrl.addWidget(self._vis_xmax)
        xl.addLayout(xrl)
        sl.addWidget(g_xaxis)

        # Y axis group
        g_yaxis = QGroupBox("Y axis")
        yl = QVBoxLayout(g_yaxis)
        yr = QHBoxLayout()
        self._vis_rb_linear = QRadioButton("Linear")
        self._vis_rb_log    = QRadioButton("Log")
        self._vis_rb_linear.setChecked(True)
        yr.addWidget(self._vis_rb_linear)
        yr.addWidget(self._vis_rb_log)
        yl.addLayout(yr)
        self._vis_yauto = QCheckBox("Auto range")
        self._vis_yauto.setChecked(True)
        yl.addWidget(self._vis_yauto)
        yrl = QHBoxLayout()
        yrl.addWidget(QLabel("Min:"))
        self._vis_ymin = QLineEdit()
        self._vis_ymin.setEnabled(False)
        yrl.addWidget(self._vis_ymin)
        yrl.addWidget(QLabel("Max:"))
        self._vis_ymax = QLineEdit()
        self._vis_ymax.setEnabled(False)
        yrl.addWidget(self._vis_ymax)
        yl.addLayout(yrl)
        sl.addWidget(g_yaxis)

        # Legend quantity group
        g_qty = QGroupBox("Legend quantity")
        ql = QVBoxLayout(g_qty)
        self._vis_rb_qty_power   = QRadioButton("Power")
        self._vis_rb_qty_density = QRadioButton("Power density (W/cm²)")
        self._vis_rb_qty_fluence = QRadioButton("Pump fluence (mJ/cm²)")
        self._vis_rb_qty_power.setChecked(True)
        for rb in (self._vis_rb_qty_power,
                   self._vis_rb_qty_density,
                   self._vis_rb_qty_fluence):
            ql.addWidget(rb)
        sl.addWidget(g_qty)

        # Wire controls → auto replot
        self._vis_rb_energy.toggled.connect(lambda _: self._vis_plot())
        self._vis_rb_linear.toggled.connect(lambda _: self._vis_plot())
        self._vis_xauto.toggled.connect(self._vis_on_xauto_toggled)
        self._vis_yauto.toggled.connect(self._vis_on_yauto_toggled)
        self._vis_xmin.returnPressed.connect(self._vis_plot)
        self._vis_xmax.returnPressed.connect(self._vis_plot)
        self._vis_ymin.returnPressed.connect(self._vis_plot)
        self._vis_ymax.returnPressed.connect(self._vis_plot)
        self._vis_power_list.itemChanged.connect(self._vis_plot)
        self._vis_rb_qty_power.toggled.connect(lambda _: self._vis_plot())
        self._vis_rb_qty_density.toggled.connect(lambda _: self._vis_plot())
        self._vis_rb_qty_fluence.toggled.connect(lambda _: self._vis_plot())

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        vis_btn_plot = QPushButton("Update plot")
        vis_btn_plot.setMinimumHeight(32)
        bold = QFont(); bold.setBold(True)
        vis_btn_plot.setFont(bold)
        vis_btn_plot.clicked.connect(self._vis_plot)
        sl.addWidget(vis_btn_plot)

        layout.addWidget(sidebar)

        # ── Canvas ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._vis_canvas  = PGCanvas(right)
        self._vis_toolbar = make_pg_toolbar(self._vis_canvas, right)
        rl.addWidget(self._vis_toolbar)
        rl.addWidget(self._vis_canvas, stretch=1)
        layout.addWidget(right, stretch=1)

    # ── Visualizer: file management ──────────────────────────────

    def _vis_on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDF5 files", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        for p in paths:
            if p not in self._vis_files:
                d = self._vis_load_file(p)
                if d is not None:
                    self._vis_files.append(p)
                    self._vis_data.append(d)
                    item = QListWidgetItem(d["label"])
                    item.setToolTip(p)
                    self._vis_file_list.addItem(item)
        self._vis_refresh_powers()
        self._vis_plot()

    def _vis_load_file(self, path: str):
        try:
            with h5py.File(path, "r") as f:
                if "Energy" in f:
                    wl = _HC_EV_NM / f["Energy"][:]     # eV → nm for internal use
                elif "Wavelength" in f:
                    wl = f["Wavelength"][:]
                else:
                    wl = f["wavelength_nm"][:]
                counts   = (f["SpectraDiff"]         if "SpectraDiff"        in f
                            else f["counts"])         [:]
                pwr_ds   = (f["Power_uncalibrated"] if "Power_uncalibrated" in f
                            else f["powers_W"])
                powers_W = pwr_ds[:].astype(float)
                if pwr_ds.attrs.get("units", "W") == "mW":
                    powers_W = powers_W * 1e-3   # normalise to W internally

                if "Power" in f:
                    pc_ds     = f["Power"]
                    power_cal = pc_ds[:].astype(float)
                    if pc_ds.attrs.get("units", "W") == "mW":
                        power_cal = power_cal * 1e-3
                else:
                    power_cal = None

                if "Power_density" in f:
                    pd_ds      = f["Power_density"]
                    power_dens = pd_ds[:].astype(float)
                    if pd_ds.attrs.get("units", "kW/cm^2") == "W/cm^2":
                        power_dens = power_dens * 1e-3   # old files: W/cm² → kW/cm²
                else:
                    power_dens = None

                if "Pump_fluence" in f:
                    fl_ds        = f["Pump_fluence"]
                    pump_fluence = fl_ds[:].astype(float)
                    if fl_ds.attrs.get("units", "mJ/cm^2") in ("uJ/cm^2", "µJ/cm^2"):
                        pump_fluence = pump_fluence * 1e-3   # old files: µJ → mJ
                else:
                    pump_fluence = None
            return {
                "label":         os.path.splitext(os.path.basename(path))[0],
                "wl":            wl,
                "counts":        counts,
                "powers_W":      powers_W,
                "power_cal":     power_cal,
                "power_density": power_dens,
                "pump_fluence":  pump_fluence,
                "path":          path,
            }
        except Exception as exc:
            QMessageBox.warning(self, "Load error",
                                f"Could not load:\n{path}\n\n{exc}")
            return None

    def _vis_on_remove(self):
        rows = sorted(
            [self._vis_file_list.row(i)
             for i in self._vis_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._vis_file_list.takeItem(row)
            self._vis_files.pop(row)
            self._vis_data.pop(row)
        self._vis_refresh_powers()
        self._vis_plot()

    def _vis_on_clear(self):
        self._vis_files.clear()
        self._vis_data.clear()
        self._vis_file_list.clear()
        self._vis_power_list.clear()
        self._vis_canvas._welcome()

    # ── Visualizer: power list ───────────────────────────────────

    @staticmethod
    def _vis_fmt_power(p_W: float) -> str:
        p_mW = p_W * 1e3
        if p_mW >= 1.0:
            return f"{p_mW:.4g} mW"
        p_uW = p_W * 1e6
        if p_uW >= 1.0:
            return f"{p_uW:.4g} µW"
        return f"{p_W * 1e9:.4g} nW"

    def _vis_fmt_qty(self, d: dict, idx: int) -> str:
        """Format the legend label for file *d* at power step *idx*
        using the currently selected legend quantity."""
        if self._vis_rb_qty_density.isChecked():
            arr = d.get("power_density")
            if arr is not None:
                return f"{arr[idx]:.4g} kW/cm²"
        elif self._vis_rb_qty_fluence.isChecked():
            arr = d.get("pump_fluence")
            if arr is not None:
                return f"{arr[idx]:.4g} mJ/cm²"
        # Default / fallback: calibrated power if available, else uncalibrated
        cal = d.get("power_cal")
        if cal is not None:
            return self._vis_fmt_power(cal[idx])
        return self._vis_fmt_power(d["powers_W"][idx])

    def _vis_refresh_powers(self):
        prev_checked = set()
        for i in range(self._vis_power_list.count()):
            item = self._vis_power_list.item(i)
            if item.checkState() == Qt.Checked:
                prev_checked.add(item.data(Qt.UserRole))
        first_load = self._vis_power_list.count() == 0

        all_powers = sorted(set(
            float(p) for d in self._vis_data for p in d["powers_W"]
        ))

        self._vis_power_list.blockSignals(True)
        self._vis_power_list.clear()
        for p_W in all_powers:
            item = QListWidgetItem(self._vis_fmt_power(p_W))
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(
                Qt.Checked if (first_load or p_W in prev_checked) else Qt.Unchecked
            )
            item.setData(Qt.UserRole, p_W)
            self._vis_power_list.addItem(item)
        self._vis_power_list.blockSignals(False)

    def _vis_check_all_powers(self):
        self._vis_power_list.blockSignals(True)
        for i in range(self._vis_power_list.count()):
            self._vis_power_list.item(i).setCheckState(Qt.Checked)
        self._vis_power_list.blockSignals(False)
        self._vis_plot()

    def _vis_check_none_powers(self):
        self._vis_power_list.blockSignals(True)
        for i in range(self._vis_power_list.count()):
            self._vis_power_list.item(i).setCheckState(Qt.Unchecked)
        self._vis_power_list.blockSignals(False)
        self._vis_plot()

    # ── Visualizer: axis range toggles ──────────────────────────

    def _vis_on_xauto_toggled(self, checked: bool):
        self._vis_xmin.setEnabled(not checked)
        self._vis_xmax.setEnabled(not checked)
        self._vis_plot()

    def _vis_on_yauto_toggled(self, checked: bool):
        self._vis_ymin.setEnabled(not checked)
        self._vis_ymax.setEnabled(not checked)
        self._vis_plot()

    # ── Visualizer: plot ─────────────────────────────────────────

    def _vis_plot(self):
        if not self._vis_data:
            return

        checked_powers = [
            self._vis_power_list.item(i).data(Qt.UserRole)
            for i in range(self._vis_power_list.count())
            if self._vis_power_list.item(i).checkState() == Qt.Checked
        ]
        if not checked_powers:
            self._vis_canvas.reset_axes()
            self._vis_canvas.draw_idle()
            return

        x_is_energy = self._vis_rb_energy.isChecked()
        n_files     = len(self._vis_data)
        p_sorted    = sorted(checked_powers)

        ax = self._vis_canvas.reset_axes()
        ax.setLogMode(y=self._vis_rb_log.isChecked())

        n_lines = n_files * len(p_sorted)
        ax.addLegend(labelTextSize=f"{max(4, 7 - n_lines // 5)}pt",
                     colCount=max(1, n_lines // 20))

        if n_files == 1:
            mlp = MultiLinePlotter(
                ax, SequentialScheme(vmin=p_sorted[0], vmax=p_sorted[-1], cmap=PLASMA)
            )
        else:
            # Log-normalised alpha: low power → 0.25, high power → 1.0
            mlp = MultiLinePlotter(ax, LogAlphaRamp(CategoricalScheme(), p_sorted))

        for fi, d in enumerate(self._vis_data):
            for p_W in p_sorted:
                closest = int(np.argmin(np.abs(d["powers_W"] - p_W)))
                wl      = d["wl"]
                counts  = d["counts"][:, closest]
                x       = _HC_EV_NM / wl if x_is_energy else wl
                s       = np.argsort(x)

                lbl = f"{d['label']} — {self._vis_fmt_qty(d, closest)}"
                mlp.plot(x[s], counts[s], index=fi, value=p_W, label=lbl, width=0.9)

        ax.setLabel("bottom", "Energy (eV)" if x_is_energy else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)

        if not self._vis_xauto.isChecked():
            try:
                ax.setXRange(float(self._vis_xmin.text()),
                             float(self._vis_xmax.text()), padding=0)
            except ValueError:
                pass
        if not self._vis_yauto.isChecked():
            try:
                ax.setYRange(float(self._vis_ymin.text()),
                             float(self._vis_ymax.text()), padding=0)
            except ValueError:
                pass

        ax.setTitle("PL spectra")
        self._vis_canvas.draw_idle()
