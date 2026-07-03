#!/usr/bin/env python3
"""
PL Spectrum Stitcher / Converter
PyQt5 GUI wrapping the stitch_spectra pipeline from pl.py.

Run:  python proc.py
"""

import sys
import os
import re
import numpy as np

# ------------------------------------------------------------------
# Import stitching logic from pl.py (sibling directory)
# ------------------------------------------------------------------
_PL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PL Helper")
)
if _PL_DIR not in sys.path:
    sys.path.insert(0, _PL_DIR)

from pl import (
    _parse_origin_power_series,
    _parse_origin_header,
    _stitch_counts,
    _HC_EV_NM,
    _gaussian,
    _find_symmetric_95_bounds,
)

import h5py

# ------------------------------------------------------------------
# Import nw_analysis from Plot directory
# ------------------------------------------------------------------
_PLOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plot")
)
if _PLOT_DIR not in sys.path:
    sys.path.insert(0, _PLOT_DIR)

try:
    import nw_analysis as nwa
    _NWA_AVAILABLE = True
    _NWA_ERROR = ""
except Exception as _nwa_import_exc:
    _NWA_AVAILABLE = False
    _NWA_ERROR = str(_nwa_import_exc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_header_center_disp(hdr):
    """Extract (center_nm, disp_nm) from raw_header_lines[5] and [6].

    Returns (None, None) if the lines are absent or unparseable.
    """
    raw = hdr.get("raw_header_lines", [])
    center_nm = disp_nm = None
    if len(raw) > 5:
        m = re.search(r"([\d.]+)\s*nm", raw[5])
        if m:
            center_nm = float(m.group(1))
    if len(raw) > 6:
        m = re.search(r"([\d.]+)\s*nm", raw[6])
        if m:
            disp_nm = float(m.group(1))
    return center_nm, disp_nm


def _write_origin_file(output_path, wl_out, counts_out, header_meta, powers_W):
    """Write a power-series spectrum in LabControl .origin format.

    The header is reconstructed from header_meta (raw_header_lines + computed
    center/dispersion); only the fields that exist in the .origin format are
    written — no HDF5-specific metadata.
    """
    wl_min, wl_max = float(wl_out.min()), float(wl_out.max())
    center_nm      = (wl_min + wl_max) / 2.0
    center_eV      = _HC_EV_NM / center_nm
    disp_nm        = wl_max - wl_min
    disp_eV        = abs(_HC_EV_NM / wl_min - _HC_EV_NM / wl_max)
    int_time_str   = header_meta.get("int_time_str", "0.500")
    raw            = header_meta.get("raw_header_lines", [])

    # Parse HWP positions from header_meta
    hwp_raw  = header_meta.get("hwp_raw", [])
    hwp_vals = []
    if hwp_raw:
        first_part = hwp_raw[0].partition("\t")[2] if "\t" in hwp_raw[0] else ""
        rest_parts = " ".join(l.strip() for l in hwp_raw[1:])
        array_str  = (first_part + " " + rest_parts).replace("[", "").replace("]", "")
        try:
            hwp_vals = [float(v) for v in re.split(r"\s+", array_str.strip()) if v]
        except ValueError:
            hwp_vals = []

    def _raw_line(idx, fallback=""):
        """Return raw header line idx, stripped of newline."""
        if idx < len(raw):
            return raw[idx].rstrip("\r\n")
        return fallback

    n_p = len(powers_W)

    with open(output_path, "w", encoding="utf-8", newline="\r\n") as fh:
        # ── 9-line header ─────────────────────────────────────────
        fh.write(_raw_line(0, "Date:\t") + "\n")
        fh.write(_raw_line(1, "Measurement type:\tPowerseries") + "\n")
        fh.write(_raw_line(2, "Temperature: \t0.000 K") + "\n")
        fh.write(_raw_line(3, f"Integration time:\t{int_time_str} s") + "\n")
        # Line 4: use first power value (matches original single-file convention)
        fh.write(f"Excitation power:\t{powers_W[0]*1e3:.4g} mW\n")
        # Lines 5–6: update with actual center/dispersion
        fh.write(f"Center wavelength\t{center_nm:.3f} nm / {center_eV:.3f} eV\n")
        fh.write(
            f"Dispersion window:\t{disp_nm:.3f} nm / {disp_eV:.3f} eV\n"
        )
        fh.write(_raw_line(7, "Entrance slit width:\t0.500 mm") + "\n")
        fh.write(_raw_line(8, "Exit slit width:\t0.000 mm") + "\n")

        # ── Column headers ────────────────────────────────────────
        fh.write(" \t\n")
        fh.write("Wavelength \tPowerspectrum \n")
        fh.write(f"(nm)\t(Counts/{int_time_str}s)\n")

        # ── Power / HWP rows ──────────────────────────────────────
        power_row = "Excitation power (W)\t" + "\t".join(f"{p:.7g}" for p in powers_W)
        fh.write(power_row + "\n")

        if hwp_vals and len(hwp_vals) >= n_p:
            hwp_str = " ".join(f"{v:.1f}." for v in hwp_vals[:n_p])
            fh.write(f"Power HWP Position (°)\t{hwp_str}\n")
        else:
            fh.write("Power HWP Position (°)\t\n")

        # ── Spectral data ─────────────────────────────────────────
        for i, wl in enumerate(wl_out):
            row_counts = "\t".join(str(int(round(counts_out[i, p])))
                                   for p in range(n_p))
            fh.write(f"{wl:.9f}\t{row_counts}\n")


def _write_h5_file(output_path, wl_out, counts_diff, counts_raw, header_meta,
                   powers_W, stitched=False, source_datasets=None,
                   dark_by_label=None, spot_diameter_um=None, rep_rate_mhz=None,
                   power_cal=None):
    """Write a power-series spectrum to HDF5, preserving all .origin metadata.

    Structure
    ---------
    /Wavelength                 float64 (n_wl,)           — nm
    /SpectraDiff                float64 (n_wl, n_powers)  — dark-subtracted, min→1
    /Spectra_raw                float64 (n_wl, n_powers)  — stitched, no dark sub
    /Power_uncalibrated         float64 (n_powers,)       — W
    /hwp_positions              float64 (n_powers,)       — degrees (if available)
    /darkspec                   float64 (n_wl,)           — mean dark (non-stitched only)
    /source_spectra/<label>/    group, one per input file (stitched files only)
        wavelength_nm           float64 (n_wl_i,)
        counts                  float64 (n_wl_i, n_powers)  — raw (not dark-subtracted)
        darkspec                float64 (n_wl_i,)            — mean dark (if available)
    Root attrs: format_version, stitched, date_str, int_time_str,
                center_nm, center_eV, dispersion_nm, dispersion_eV,
                spot_diameter_um (optional), rep_rate_mhz (optional),
                header_line_0 … header_line_8
    """
    wl_min, wl_max  = float(wl_out.min()), float(wl_out.max())
    center_nm       = (wl_min + wl_max) / 2.0
    center_eV       = _HC_EV_NM / center_nm
    disp_nm         = wl_max - wl_min
    disp_eV         = abs(_HC_EV_NM / wl_min - _HC_EV_NM / wl_max)
    int_time_str    = header_meta.get("int_time_str", "0.500")

    hwp_raw   = header_meta.get("hwp_raw", [])
    hwp_label = ""
    hwp_vals  = []
    if hwp_raw:
        hwp_label  = hwp_raw[0].partition("\t")[0]
        first_part = hwp_raw[0].partition("\t")[2] if "\t" in hwp_raw[0] else ""
        rest_parts = " ".join(l.strip() for l in hwp_raw[1:])
        array_str  = (first_part + " " + rest_parts).replace("[", "").replace("]", "")
        hwp_vals   = [v for v in re.split(r"\s+", array_str.strip()) if v]

    raw      = header_meta.get("raw_header_lines", [])
    has_dark = bool(dark_by_label and any(v is not None for v in dark_by_label.values()))

    with h5py.File(output_path, "w") as f:
        # ── Primary datasets ───────────────────────────────────────
        d_wl = f.create_dataset("Wavelength", data=wl_out, compression="gzip")
        d_wl.attrs["units"] = "nm"

        d_diff = f.create_dataset("SpectraDiff", data=counts_diff, compression="gzip")
        d_diff.attrs["units"]           = f"Counts/{int_time_str}s"
        d_diff.attrs["axes"]            = "Wavelength : Power_uncalibrated"
        d_diff.attrs["dark_subtracted"] = "Yes" if has_dark else "No"

        d_raw = f.create_dataset("Spectra_raw", data=counts_raw, compression="gzip")
        d_raw.attrs["units"] = f"Counts/{int_time_str}s"
        d_raw.attrs["axes"]  = "Wavelength : Power_uncalibrated"

        d_p = f.create_dataset("Power_uncalibrated", data=powers_W)
        d_p.attrs["units"] = "W"

        if hwp_vals:
            try:
                hwp_arr = np.array(
                    [float(v) for v in hwp_vals[:len(powers_W)]], dtype=float
                )
                d_h = f.create_dataset("hwp_positions", data=hwp_arr)
                d_h.attrs["label"] = hwp_label
                d_h.attrs["units"] = "degrees"
            except ValueError:
                pass

        # ── Dark spectrum (non-stitched only) ─────────────────────
        if not stitched and has_dark:
            dark_arr = next(iter(dark_by_label.values()))
            d_dk = f.create_dataset("darkspec", data=dark_arr, compression="gzip")
            d_dk.attrs["units"] = f"Counts/{int_time_str}s"

        # ── Root attributes ────────────────────────────────────────
        f.attrs["format_version"] = 1
        f.attrs["stitched"]       = "Yes" if stitched else "No"
        f.attrs["date_str"]       = header_meta.get("date_str", "")
        f.attrs["int_time_str"]   = int_time_str
        f.attrs["center_nm"]      = center_nm
        f.attrs["center_eV"]      = center_eV
        f.attrs["dispersion_nm"]  = disp_nm
        f.attrs["dispersion_eV"]  = disp_eV
        if spot_diameter_um is not None:
            f.attrs["spot_diameter_um"] = float(spot_diameter_um)
        if rep_rate_mhz is not None:
            f.attrs["rep_rate_mhz"] = float(rep_rate_mhz)
        for i, line in enumerate(raw[:9]):
            f.attrs[f"header_line_{i}"] = line.rstrip("\r\n")

        # ── Source spectra (stitched files only) ───────────────────
        if stitched and source_datasets:
            grp = f.create_group("source_spectra")
            for d in source_datasets:
                lbl = d["label"].replace("/", "_")
                sg = grp.create_group(lbl)
                sg_wl = sg.create_dataset(
                    "wavelength_nm", data=d["wl"], compression="gzip"
                )
                sg_wl.attrs["units"] = "nm"
                sg_c = sg.create_dataset(
                    "counts", data=d["counts"], compression="gzip"
                )
                sg_c.attrs["units"] = f"Counts/{int_time_str}s"
                if dark_by_label:
                    dark_arr = dark_by_label.get(d["label"])
                    if dark_arr is not None:
                        sg_dk = sg.create_dataset(
                            "darkspec", data=dark_arr, compression="gzip"
                        )
                        sg_dk.attrs["units"] = f"Counts/{int_time_str}s"

        # ── Power calibration ─────────────────────────────────────
        # power used for derived quantities; updated to calibrated value if available
        power_for_derived = powers_W

        if power_cal is not None:
            grp_c = f.create_group("power_calibration")
            for role in ("atBS", "atSample"):
                entry = power_cal.get(role)
                if entry is not None:
                    hwp_arr, pow_arr = entry
                    dh = grp_c.create_dataset(f"{role}_hwp_positions", data=hwp_arr)
                    dh.attrs["units"] = "degrees"
                    dp = grp_c.create_dataset(f"{role}_power_W", data=pow_arr)
                    dp.attrs["units"] = "W"

            # Compute Power when both sides are present.
            # Method mirrors MATLAB: fit  P_sample = a * P_BS  (linear through
            # origin, OLS) using the overlapping HWP positions, then apply
            # P_calibrated = a * Power_uncalibrated.
            cal_bs  = power_cal.get("atBS")
            cal_s   = power_cal.get("atSample")
            if cal_bs is not None and cal_s is not None:
                hwp_bs,  pow_bs  = cal_bs
                hwp_s,   pow_s   = cal_s
                hwp_bs_r = np.round(hwp_bs).astype(int)
                hwp_s_r  = np.round(hwp_s).astype(int)
                common   = np.intersect1d(hwp_bs_r, hwp_s_r)
                if len(common) >= 2:
                    idx_bs = np.array([np.where(hwp_bs_r == h)[0][0] for h in common])
                    idx_s  = np.array([np.where(hwp_s_r  == h)[0][0] for h in common])
                    x = pow_bs[idx_bs]
                    y = pow_s [idx_s ]
                    a_transmission  = float(np.dot(x, y) / np.dot(x, x))
                    power_for_derived = a_transmission * powers_W
                    d_pc = f.create_dataset("Power", data=power_for_derived)
                    d_pc.attrs["units"]         = "W"
                    d_pc.attrs["transmission"]  = a_transmission
                    d_pc.attrs["n_cal_points"]  = len(common)
                    grp_c.attrs["transmission"] = a_transmission

        # ── Derived spatial/temporal quantities ───────────────────
        if spot_diameter_um is not None:
            dspot_cm = float(spot_diameter_um) * 1e-4          # µm → cm
            power_density = 4.0 * power_for_derived / np.pi / dspot_cm ** 2
            d_pd = f.create_dataset("Power_density", data=power_density)
            d_pd.attrs["units"] = "W/cm^2"

            if rep_rate_mhz is not None:
                f_hz    = float(rep_rate_mhz) * 1e6            # MHz → Hz
                fluence = (4.0 * power_for_derived / f_hz
                           / np.pi / dspot_cm ** 2 * 1e6)      # → µJ/cm²
                d_fl = f.create_dataset("Pump_fluence", data=fluence)
                d_fl.attrs["units"] = "uJ/cm^2"


def _parse_power_calibration(filepath):
    """Parse an atBS or atSample power calibration .origin file.

    The file stores rows of  <power_W>  <hwp_pos>  <power_W>  (tab-separated).
    Returns (hwp_positions, powers_W) sorted by HWP position.
    """
    hwp, pows = [], []
    with open(filepath, encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                pw = float(parts[0])
                hp = float(parts[1])
                hwp.append(hp)
                pows.append(pw)
            except ValueError:
                continue
    if not hwp:
        raise ValueError(f"No numeric data found in {filepath}")
    hwp  = np.array(hwp,  dtype=float)
    pows = np.array(pows, dtype=float)
    order = np.argsort(hwp)
    return hwp[order], pows[order]


# ------------------------------------------------------------------
# Qt + matplotlib imports
# ------------------------------------------------------------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QRadioButton, QGroupBox, QCheckBox,
    QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame, QSizePolicy,
    QAbstractItemView, QTabWidget, QComboBox, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ------------------------------------------------------------------
# Embedded matplotlib canvas
# ------------------------------------------------------------------
class _MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._welcome()

    def reset_axes(self):
        """Clear the whole figure (removes colorbars etc.) and return a fresh axes."""
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        return self.ax

    def _welcome(self):
        self.reset_axes()
        self.ax.set_axis_off()
        self.ax.text(
            0.5, 0.5,
            "Add .origin files and click\n\"Preview\" to begin.",
            ha="center", va="center",
            transform=self.ax.transAxes,
            fontsize=12, color="#666666",
        )
        self.draw_idle()


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------
class StitchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL Spectrum Stitcher / Converter")
        self.resize(1160, 680)

        # ── State ────────────────────────────────────────────────
        self._files: list[str]   = []
        self._datasets: list[dict] = []   # {label, wl, counts, powers, path}
        self._hdrs: list[dict]   = []
        self._pairs: list[tuple] = []     # (i_a, i_b, ov_lo_wl, ov_hi_wl)
        self._spans: dict        = {}     # {(i_a,i_b): (lo_wl, hi_wl)}
        self._spans_display: dict = {}    # {(i_a,i_b): (xmin, xmax)} display units
        self._pair_idx: int      = 0
        self._x_axis: str        = "energy"
        self._span_selector      = None
        self._mode: str          = "idle"

        # dark_map: {dataset_label: dark_info_dict | None}
        # dark_info_dict = {label, path, wl, mean (shape n_wl_dark)}
        self._dark_map: dict     = {}

        # ── Visualizer state ──────────────────────────────────────
        self._vis_files: list[str]  = []
        self._vis_data:  list[dict] = []   # {label, wl, counts, powers_W, path}

        # ── Analysis state ────────────────────────────────────────
        self._ana_file: str | None  = None
        self._ana_nw                = None   # nw_analysis SimpleNamespace
        self._ana_mode              = "idle"
        # peak clicking
        self._ana_sc_ref_idx        = 0
        self._ana_sc_clicks_x       = []
        self._ana_sc_click_conn     = None
        self._ana_sc_span           = None
        self._ana_sc_span_sel       = [None]
        self._ana_sc_windows        = []
        self._ana_sc_peak_idx       = 0
        # thresholds
        self._ana_thr_queue         = []
        self._ana_thr_idx           = 0
        self._ana_thr_results       = {}
        self._ana_thr_rect_sel      = None
        self._ana_thr_sel_pts       = []
        self._ana_thr_coeffs        = None
        self._ana_thr_power         = None
        self._ana_thr_values        = None
        self._ana_thr_title         = ""
        self._ana_thr_scatter       = None

        # ── Power calibration state ──────────────────────────────
        self._cal_atbs_data     = None   # (hwp_arr, powers_W) or None
        self._cal_atsample_data = None   # (hwp_arr, powers_W) or None
        self._cal_atbs_path     = None   # str or None
        self._cal_atsample_path = None   # str or None

        # ── Spotsize state ────────────────────────────────────────
        self._ss_x            = None
        self._ss_deriv        = None
        self._ss_spans        = [None, None]
        self._ss_span_patches = [None, None]
        self._ss_selector     = None

        self._build_ui()
        self._refresh_buttons()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)
        self._tabs.addTab(self._build_stitch_tab(),     "Stitch / Convert")
        self._tabs.addTab(self._build_vis_tab(),        "Visualizer")
        self._tabs.addTab(self._build_analysis_tab(),   "Analysis")
        self._tabs.addTab(self._build_spotsize_tab(),   "Spotsize")
        self.statusBar().showMessage("Ready — add .origin files to begin.")

    def _build_stitch_tab(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Left sidebar ──────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Input files")
        fl = QVBoxLayout(g_files)
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setToolTip("Loaded .origin files (sorted by wavelength range)")
        fl.addWidget(self._file_list)
        fb = QHBoxLayout()
        self._btn_add    = QPushButton("Add…")
        self._btn_remove = QPushButton("Remove")
        self._btn_clear  = QPushButton("Clear")
        self._btn_add.clicked.connect(self._on_add_files)
        self._btn_remove.clicked.connect(self._on_remove_files)
        self._btn_clear.clicked.connect(self._on_clear_files)
        for b in (self._btn_add, self._btn_remove, self._btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        sl.addWidget(g_files)

        # X-axis group
        g_xaxis = QGroupBox("X axis")
        xl = QHBoxLayout(g_xaxis)
        self._rb_energy = QRadioButton("Energy (eV)")
        self._rb_wl     = QRadioButton("Wavelength (nm)")
        self._rb_energy.setChecked(True)
        self._rb_energy.toggled.connect(self._on_xaxis_changed)
        xl.addWidget(self._rb_energy)
        xl.addWidget(self._rb_wl)
        sl.addWidget(g_xaxis)

        # Transition spans group
        g_spans = QGroupBox("Transition spans")
        spl = QVBoxLayout(g_spans)
        self._span_list = QListWidget()
        self._span_list.setToolTip(
            "One span per overlapping pair.\nClick a row to jump to that pair."
        )
        self._span_list.itemClicked.connect(self._on_span_item_clicked)
        spl.addWidget(self._span_list)
        sl.addWidget(g_spans, stretch=1)

        # Dark spectra group
        g_dark = QGroupBox("Dark spectra")
        dl = QVBoxLayout(g_dark)
        self._dark_list = QListWidget()
        self._dark_list.setToolTip(
            "Dark spectrum matched to each input file.\n"
            "Select a row and click \"Load…\" to set manually."
        )
        self._dark_list.setMaximumHeight(110)
        dl.addWidget(self._dark_list)
        db = QHBoxLayout()
        self._btn_autosearch = QPushButton("Autosearch")
        self._btn_load_dark  = QPushButton("Load…")
        self._btn_clear_dark = QPushButton("Clear")
        self._btn_autosearch.setToolTip(
            "Search for files with 'dark' in the name in each\n"
            "input directory or its 'dark' subfolder.\n"
            "Matches by integration time and center wavelength."
        )
        self._btn_autosearch.clicked.connect(self._on_autosearch_dark)
        self._btn_load_dark.clicked.connect(self._on_load_dark)
        self._btn_clear_dark.clicked.connect(self._on_clear_dark)
        for b in (self._btn_autosearch, self._btn_load_dark, self._btn_clear_dark):
            db.addWidget(b)
        dl.addLayout(db)
        sl.addWidget(g_dark)

        # Metadata group
        g_meta = QGroupBox("Metadata (saved to HDF5)")
        ml = QVBoxLayout(g_meta)
        r_spot = QHBoxLayout()
        r_spot.addWidget(QLabel("Spot diameter (µm):"))
        self._meta_spot_diam = QLineEdit()
        self._meta_spot_diam.setPlaceholderText("optional")
        r_spot.addWidget(self._meta_spot_diam)
        ml.addLayout(r_spot)
        r_rep = QHBoxLayout()
        r_rep.addWidget(QLabel("Rep. rate (MHz):"))
        self._meta_rep_rate = QLineEdit()
        self._meta_rep_rate.setPlaceholderText("optional")
        r_rep.addWidget(self._meta_rep_rate)
        ml.addLayout(r_rep)
        sl.addWidget(g_meta)

        # Power calibration group
        g_cal = QGroupBox("Power calibration")
        cl = QVBoxLayout(g_cal)

        r_atbs = QHBoxLayout()
        r_atbs.addWidget(QLabel("atBS:"))
        self._cal_atbs_lbl = QLabel("—")
        self._cal_atbs_lbl.setWordWrap(False)
        self._cal_atbs_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_atbs.addWidget(self._cal_atbs_lbl, stretch=1)
        btn_load_atbs = QPushButton("Load…")
        btn_load_atbs.setFixedWidth(54)
        btn_load_atbs.clicked.connect(self._on_load_cal_atbs)
        r_atbs.addWidget(btn_load_atbs)
        cl.addLayout(r_atbs)

        r_ats = QHBoxLayout()
        r_ats.addWidget(QLabel("atSample:"))
        self._cal_atsample_lbl = QLabel("—")
        self._cal_atsample_lbl.setWordWrap(False)
        self._cal_atsample_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_ats.addWidget(self._cal_atsample_lbl, stretch=1)
        btn_load_ats = QPushButton("Load…")
        btn_load_ats.setFixedWidth(54)
        btn_load_ats.clicked.connect(self._on_load_cal_atsample)
        r_ats.addWidget(btn_load_ats)
        cl.addLayout(r_ats)

        btn_cal_auto = QPushButton("Autosearch")
        btn_cal_auto.clicked.connect(self._on_autosearch_cal)
        cl.addWidget(btn_cal_auto)
        sl.addWidget(g_cal)

        # Action buttons
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._btn_preview     = QPushButton("1  Preview spectra")
        self._btn_spans       = QPushButton("2  Select spans")
        self._btn_prev_result = QPushButton("3  Preview result")
        self._btn_stitch      = QPushButton("4  Save as HDF5…")
        self._btn_save_origin = QPushButton("5  Save as .origin…")
        self._btn_preview.clicked.connect(self._on_preview)
        self._btn_spans.clicked.connect(self._on_start_spans)
        self._btn_prev_result.clicked.connect(self._on_preview_result)
        self._btn_stitch.clicked.connect(self._on_stitch_save)
        self._btn_save_origin.clicked.connect(self._on_save_origin)

        bold = QFont(); bold.setBold(True)
        self._btn_stitch.setFont(bold)
        self._btn_stitch.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        for b in (self._btn_preview, self._btn_spans, self._btn_prev_result,
                  self._btn_stitch, self._btn_save_origin):
            b.setMinimumHeight(32)
            sl.addWidget(b)

        layout.addWidget(sidebar)

        # ── Right: canvas ─────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self._canvas  = _MplCanvas(right)
        self._toolbar = NavigationToolbar2QT(self._canvas, right)
        rl.addWidget(self._toolbar)
        rl.addWidget(self._canvas, stretch=1)

        # Pair navigation bar
        self._nav_bar = QFrame()
        self._nav_bar.setFrameShape(QFrame.StyledPanel)
        nav = QHBoxLayout(self._nav_bar)
        nav.setContentsMargins(8, 4, 8, 4)
        self._btn_prev  = QPushButton("◀  Prev")
        self._lbl_pair  = QLabel("Pair 1 of 1")
        self._lbl_pair.setAlignment(Qt.AlignCenter)
        self._btn_next  = QPushButton("Next  ▶")
        self._btn_done  = QPushButton("✓  Done")
        self._btn_done.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; }"
        )
        self._btn_prev.clicked.connect(self._on_prev_pair)
        self._btn_next.clicked.connect(self._on_next_pair)
        self._btn_done.clicked.connect(self._on_done_spans)
        for w in (self._btn_prev, self._lbl_pair, self._btn_next, self._btn_done):
            nav.addWidget(w)
        self._nav_bar.setVisible(False)
        rl.addWidget(self._nav_bar)

        layout.addWidget(right, stretch=1)
        return root

    # ── File management ──────────────────────────────────────────

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .origin files", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        added = 0
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                item = QListWidgetItem(os.path.basename(p))
                item.setToolTip(p)
                self._file_list.addItem(item)
                added += 1
        if added:
            self._reset_parsed()
            self._refresh_buttons()
            self.statusBar().showMessage(
                f"{len(self._files)} file(s) loaded. Click \"Preview\" to continue."
            )

    def _on_remove_files(self):
        rows = sorted(
            [self._file_list.row(i) for i in self._file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._file_list.takeItem(row)
            if row < len(self._files):
                self._files.pop(row)
        self._reset_parsed()
        self._refresh_buttons()

    def _on_clear_files(self):
        self._files.clear()
        self._file_list.clear()
        self._reset_parsed()
        self._clear_canvas()
        self._refresh_buttons()
        self.statusBar().showMessage("Files cleared.")

    def _reset_parsed(self):
        self._datasets.clear()
        self._hdrs.clear()
        self._pairs.clear()
        self._spans.clear()
        self._spans_display.clear()
        self._span_list.clear()
        self._dark_map.clear()
        self._dark_list.clear()
        self._span_selector = None
        self._mode = "idle"
        self._nav_bar.setVisible(False)

    # ── X-axis toggle ────────────────────────────────────────────

    def _on_xaxis_changed(self):
        new = "energy" if self._rb_energy.isChecked() else "wavelength"
        if new == self._x_axis:
            return
        converted = {}
        for key, (x0, x1) in self._spans_display.items():
            a, b = _HC_EV_NM / x0, _HC_EV_NM / x1
            converted[key] = (min(a, b), max(a, b))
        self._x_axis = new
        self._spans_display = converted
        if self._mode == "preview":
            self._draw_preview()
        elif self._mode == "spans":
            self._draw_pair(self._pair_idx)
        self._refresh_span_list()

    # ── Parsing ──────────────────────────────────────────────────

    def _parse_all(self) -> bool:
        """Parse all files; return True if at least 1 parsed successfully."""
        self._datasets.clear()
        self._hdrs.clear()
        errors = []
        for path in self._files:
            label = os.path.splitext(os.path.basename(path))[0]
            try:
                wl, counts, powers = _parse_origin_power_series(path)
                hdr = _parse_origin_header(path)
                self._datasets.append({
                    "label": label, "wl": wl, "counts": counts,
                    "powers": powers, "path": path,
                })
                self._hdrs.append(hdr)
            except Exception as exc:
                errors.append(f"  {label}: {exc}")
        if errors:
            QMessageBox.warning(self, "Parse errors",
                                "Could not parse:\n" + "\n".join(errors))
        if not self._datasets:
            QMessageBox.warning(self, "No valid files",
                                "None of the selected files could be parsed.")
            return False
        order = sorted(range(len(self._datasets)),
                       key=lambda i: self._datasets[i]["wl"].min())
        self._datasets = [self._datasets[i] for i in order]
        self._hdrs     = [self._hdrs[i]     for i in order]
        self._refresh_dark_list()
        return True

    def _find_pairs(self) -> bool:
        self._pairs.clear()
        for i in range(len(self._datasets) - 1):
            a, b = self._datasets[i], self._datasets[i + 1]
            ov_lo = max(a["wl"].min(), b["wl"].min())
            ov_hi = min(a["wl"].max(), b["wl"].max())
            if ov_hi > ov_lo:
                self._pairs.append((i, i + 1, ov_lo, ov_hi))
        self._refresh_span_list()
        return bool(self._pairs)

    def _refresh_span_list(self):
        self._span_list.clear()
        for k, (i_a, i_b, _, _) in enumerate(self._pairs):
            a, b = self._datasets[i_a], self._datasets[i_b]
            key = (i_a, i_b)
            if key in self._spans:
                lo_wl, hi_wl = self._spans[key]
                xmin = self._wl_to_x(hi_wl if self._x_axis == "energy" else lo_wl)
                xmax = self._wl_to_x(lo_wl if self._x_axis == "energy" else hi_wl)
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                text  = f"✅  {a['label']} ↔ {b['label']}\n    [{xmin:.4g} … {xmax:.4g}]"
                color = QColor("#c8e6c9")
            else:
                text  = f"⬜  {a['label']} ↔ {b['label']}"
                color = QColor("#ffffff")
            item = QListWidgetItem(text)
            item.setBackground(color)
            item.setData(Qt.UserRole, k)
            self._span_list.addItem(item)

    # ── Dark spectra ─────────────────────────────────────────────

    def _parse_dark_file(self, path: str) -> dict | None:
        """Parse a .origin file as a dark spectrum. Returns info dict or None."""
        try:
            wl, counts, _ = _parse_origin_power_series(path)
            hdr = _parse_origin_header(path)
            center_nm, disp_nm = _parse_header_center_disp(hdr)
            return {
                "path":       path,
                "label":      os.path.splitext(os.path.basename(path))[0],
                "wl":         wl,
                "mean":       counts.mean(axis=1),   # (n_wl,) — mean over power steps
                "int_time":   hdr.get("int_time_str", ""),
                "center_nm":  center_nm,
                "disp_nm":    disp_nm,
            }
        except Exception:
            return None

    def _darks_match(self, sig_hdr: dict, dark_info: dict) -> bool:
        """Return True if dark_info is compatible with the signal header."""
        sig_int   = sig_hdr.get("int_time_str", "")
        sig_c, sig_d = _parse_header_center_disp(sig_hdr)
        if sig_int != dark_info["int_time"]:
            return False
        if sig_c is None or dark_info["center_nm"] is None:
            return False
        if abs(sig_c - dark_info["center_nm"]) > 5.0:
            return False
        if sig_d is not None and dark_info["disp_nm"] is not None:
            if abs(sig_d - dark_info["disp_nm"]) > 5.0:
                return False
        return True

    def _autosearch_darks(self, force: bool = False):
        """Search for dark files in input directories and match them.

        force=False (default): skip datasets that already have a dark assigned
                               so manual assignments are never overwritten.
        force=True:            overwrite all entries (explicit Autosearch button).
        """
        # Collect unique directories
        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._files))

        # Find candidate .origin files with "dark" in name, or in dark/ subfolder
        candidates = []
        seen = set()
        for d in dirs:
            try:
                for fname in os.listdir(d):
                    if "dark" in fname.lower() and fname.lower().endswith(".origin"):
                        fp = os.path.join(d, fname)
                        if fp not in seen:
                            candidates.append(fp)
                            seen.add(fp)
            except OSError:
                pass
            dark_sub = os.path.join(d, "dark")
            if os.path.isdir(dark_sub):
                try:
                    for fname in os.listdir(dark_sub):
                        if "dark" in fname.lower() and fname.lower().endswith(".origin"):
                            fp = os.path.join(dark_sub, fname)
                            if fp not in seen:
                                candidates.append(fp)
                                seen.add(fp)
                except OSError:
                    pass

        # Parse candidates
        parsed = [self._parse_dark_file(p) for p in candidates]
        parsed = [d for d in parsed if d is not None]

        # Match to each dataset
        n_matched = 0
        for dataset, hdr in zip(self._datasets, self._hdrs):
            label = dataset["label"]
            if not force and self._dark_map.get(label) is not None:
                # Preserve existing assignment (manual or previous auto-match)
                n_matched += 1
                continue
            best = next((d for d in parsed if self._darks_match(hdr, d)), None)
            self._dark_map[label] = best
            if best is not None:
                n_matched += 1

        self._refresh_dark_list()
        self._refresh_buttons()
        if not candidates:
            self.statusBar().showMessage(
                "Autosearch: no dark files found (looking for 'dark' in name "
                "or files in a 'dark' subfolder)."
            )
        else:
            self.statusBar().showMessage(
                f"Autosearch: {len(candidates)} candidate(s) found, "
                f"{n_matched}/{len(self._datasets)} file(s) matched."
            )

    def _on_autosearch_dark(self):
        if not self._datasets:
            if not self._parse_all():
                return
            self._find_pairs()
        self._autosearch_darks(force=True)

    def _on_load_dark(self):
        """Manually assign a dark file to the selected entry in the dark list."""
        row = self._dark_list.currentRow()
        if row < 0 or row >= len(self._datasets):
            QMessageBox.information(self, "No selection",
                                    "Select a file in the dark list first.")
            return
        dataset = self._datasets[row]
        sig_hdr = self._hdrs[row]

        path, _ = QFileDialog.getOpenFileName(
            self, f"Dark file for '{dataset['label']}'", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if not path:
            return

        dark_info = self._parse_dark_file(path)
        if dark_info is None:
            QMessageBox.critical(self, "Parse error",
                                 f"Could not parse:\n{path}")
            return

        if not self._darks_match(sig_hdr, dark_info):
            sig_int       = sig_hdr.get("int_time_str", "?")
            sig_c, sig_d  = _parse_header_center_disp(sig_hdr)
            reply = QMessageBox.question(
                self, "Metadata mismatch",
                f"The selected dark file does not match on:\n"
                f"  integration time: signal={sig_int}  dark={dark_info['int_time']}\n"
                f"  center wavelength: signal={sig_c}  dark={dark_info['center_nm']}\n"
                f"  dispersion: signal={sig_d}  dark={dark_info['disp_nm']}\n\n"
                "Assign it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._dark_map[dataset["label"]] = dark_info
        self._refresh_dark_list()
        self._refresh_buttons()
        self.statusBar().showMessage(
            f"Dark assigned: {dataset['label']} ← {dark_info['label']}"
        )

    def _on_clear_dark(self):
        self._dark_map.clear()
        self._refresh_dark_list()
        self._refresh_buttons()
        self.statusBar().showMessage("Dark spectra cleared.")

    # ── Power calibration ─────────────────────────────────────────

    def _cal_load(self, path, role):
        """Parse a power-calibration file and store it under *role* ('atBS'/'atSample')."""
        try:
            hwp, pows = _parse_power_calibration(path)
        except Exception as exc:
            QMessageBox.critical(self, "Calibration load error",
                                 f"Could not parse:\n{path}\n\n{exc}")
            return False
        short = os.path.basename(path)
        if len(short) > 30:
            short = short[:27] + "…"
        if role == "atBS":
            self._cal_atbs_data  = (hwp, pows)
            self._cal_atbs_path  = path
            self._cal_atbs_lbl.setText(short)
        else:
            self._cal_atsample_data  = (hwp, pows)
            self._cal_atsample_path  = path
            self._cal_atsample_lbl.setText(short)
        self.statusBar().showMessage(
            f"Calibration {role}: {len(hwp)} HWP steps loaded from {short}"
        )
        return True

    def _on_load_cal_atbs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atBS calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._cal_load(path, "atBS")

    def _on_load_cal_atsample(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atSample calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._cal_load(path, "atSample")

    def _on_autosearch_cal(self):
        """Search input-file directories (and a 'Calibration' subdirectory) for
        calibration files whose names contain 'atbs' or 'atsample' (case-insensitive).
        """
        if not self._files:
            QMessageBox.information(self, "No files loaded",
                                    "Add .origin files first.")
            return

        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._files))

        search_dirs = []
        seen_dirs   = set()
        for d in dirs:
            if d not in seen_dirs:
                search_dirs.append(d)
                seen_dirs.add(d)
            # Find a 'Calibration' subfolder (case-insensitive)
            try:
                for entry in os.listdir(d):
                    if entry.lower() == "calibration":
                        sub = os.path.join(d, entry)
                        if os.path.isdir(sub) and sub not in seen_dirs:
                            search_dirs.append(sub)
                            seen_dirs.add(sub)
            except OSError:
                pass

        atbs_path = atsample_path = None
        for d in search_dirs:
            try:
                for fname in os.listdir(d):
                    if not fname.lower().endswith(".origin"):
                        continue
                    fl = fname.lower()
                    fp = os.path.join(d, fname)
                    if atbs_path is None and "atbs" in fl:
                        atbs_path = fp
                    if atsample_path is None and "atsample" in fl:
                        atsample_path = fp
            except OSError:
                pass
            if atbs_path and atsample_path:
                break

        found = []
        if atbs_path:
            if self._cal_load(atbs_path, "atBS"):
                found.append(f"atBS: {os.path.basename(atbs_path)}")
        if atsample_path:
            if self._cal_load(atsample_path, "atSample"):
                found.append(f"atSample: {os.path.basename(atsample_path)}")

        if found:
            self.statusBar().showMessage(
                "Calibration autosearch found: " + ", ".join(found)
            )
        else:
            self.statusBar().showMessage(
                "Calibration autosearch: no matching files found "
                "(looking for 'atBS'/'atSample' in name in input dirs and 'Calibration/' subfolder)."
            )

    def _refresh_dark_list(self):
        self._dark_list.clear()
        for dataset in self._datasets:
            dark_info = self._dark_map.get(dataset["label"])
            if dark_info is not None:
                text  = f"✅  {dataset['label']}\n    ← {dark_info['label']}"
                color = QColor("#c8e6c9")
            else:
                text  = f"⬜  {dataset['label']}  ← not found"
                color = QColor("#ffffff")
            item = QListWidgetItem(text)
            item.setBackground(color)
            self._dark_list.addItem(item)

    def _apply_dark(self, datasets: list) -> tuple[list, dict]:
        """Subtract matched dark from each dataset's counts index-by-index.

        Returns (ds_sub, dark_by_label) where:
          ds_sub         — list of dataset dicts with counts replaced by dark-subtracted values
          dark_by_label  — {label: dark array aligned to file pixel grid | None}
        """
        ds_sub        = []
        dark_by_label = {}
        for d in datasets:
            dark_info = self._dark_map.get(d["label"])
            if dark_info is not None:
                n_sig  = len(d["wl"])
                n_dark = len(dark_info["mean"])
                if n_dark >= n_sig:
                    dark_arr = dark_info["mean"][:n_sig]
                else:
                    # Dark is shorter: pad with zeros (no subtraction for missing pixels)
                    dark_arr = np.zeros(n_sig)
                    dark_arr[:n_dark] = dark_info["mean"]
                counts_sub = d["counts"] - dark_arr[:, np.newaxis]
                ds_sub.append({**d, "counts": counts_sub})
                dark_by_label[d["label"]] = dark_arr
            else:
                ds_sub.append(d)
                dark_by_label[d["label"]] = None
        return ds_sub, dark_by_label

    # ── Step 1 — Preview ─────────────────────────────────────────

    def _on_preview(self):
        if not self._parse_all():
            return
        self._find_pairs()
        self._autosearch_darks()
        self._mode = "preview"
        self._nav_bar.setVisible(False)
        self._draw_preview()
        self._refresh_buttons()
        n = len(self._datasets)
        n_dark = sum(1 for v in self._dark_map.values() if v is not None)
        dark_note = f", {n_dark}/{n} dark(s) matched" if n_dark else ""
        if n == 1:
            self.statusBar().showMessage(
                f"1 file loaded{dark_note}. Click \"Save as HDF5\" to convert."
            )
        else:
            self.statusBar().showMessage(
                f"{n} file(s), {len(self._pairs)} overlapping pair(s){dark_note}."
            )

    def _draw_preview(self):
        ds_plot, dark_by_label = self._apply_dark(self._datasets)
        n_dark  = sum(1 for v in dark_by_label.values() if v is not None)
        subtitle = f"dark-subtracted: {n_dark}/{len(ds_plot)}" if n_dark else "raw"

        ax = self._canvas.reset_axes()
        ax.set_axis_on()
        cmap_fn = cm.get_cmap("tab10")
        for k, d in enumerate(ds_plot):
            x   = self._wl_to_x(d["wl"])
            idx = np.argsort(x)
            ax.plot(x[idx], d["counts"][idx, -1],
                    lw=1.3, color=cmap_fn(k % 10),
                    label=f"{d['label']}  ({d['powers'][-1] * 1e3:.3g} mW)")
        ax.set_xlabel("Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Last-power spectra (preview — {subtitle})")
        self._canvas.fig.tight_layout()
        self._canvas.draw_idle()

    # ── Step 2 — Select spans ────────────────────────────────────

    def _on_start_spans(self):
        if not self._datasets:
            if not self._parse_all():
                return
        if not self._pairs and not self._find_pairs():
            QMessageBox.information(self, "No overlaps",
                                    "No overlapping wavelength ranges found.")
            return
        self._mode = "spans"
        self._pair_idx = 0
        self._nav_bar.setVisible(True)
        self._draw_pair(0)
        self._refresh_buttons()
        self.statusBar().showMessage(
            "Drag to select a transition span for each pair. "
            "Use ◀/▶ to navigate, ✓ Done when finished."
        )

    def _draw_pair(self, idx: int):
        if not (0 <= idx < len(self._pairs)):
            return
        i_a, i_b, ov_lo_wl, ov_hi_wl = self._pairs[idx]
        a, b = self._datasets[i_a], self._datasets[i_b]

        if self._span_selector is not None:
            try:
                self._span_selector.set_active(False)
            except Exception:
                pass
            self._span_selector = None

        ds_sub, _ = self._apply_dark([a, b])
        a_plot, b_plot = ds_sub

        ax = self._canvas.reset_axes()
        ax.set_axis_on()
        for d, color in zip((a_plot, b_plot), ("tab:blue", "tab:orange")):
            mask = (d["wl"] >= ov_lo_wl) & (d["wl"] <= ov_hi_wl)
            x_ov = self._wl_to_x(d["wl"][mask])
            c_ov = d["counts"][mask, -1]
            s    = np.argsort(x_ov)
            ax.plot(x_ov[s], c_ov[s], lw=1.4, color=color, label=d["label"])

        x_lo = self._wl_to_x(ov_hi_wl if self._x_axis == "energy" else ov_lo_wl)
        x_hi = self._wl_to_x(ov_lo_wl if self._x_axis == "energy" else ov_hi_wl)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Pair {idx + 1}/{len(self._pairs)}: {a['label']}  ↔  {b['label']}\n"
            "Drag to select the transition span"
        )
        self._canvas.fig.tight_layout()
        self._canvas.draw_idle()

        key = (i_a, i_b)
        if key in self._spans_display:
            x0, x1 = self._spans_display[key]
            ax.axvspan(x0, x1, alpha=0.28, color="tab:green", zorder=0)
            self._canvas.draw_idle()

        self._span_selector = SpanSelector(
            ax,
            lambda xmin, xmax, _idx=idx: self._on_span_selected(xmin, xmax, _idx),
            "horizontal",
            useblit=False,
            interactive=True,
            props=dict(facecolor="tab:green", alpha=0.22),
            handle_props=dict(color="darkgreen"),
        )
        self._lbl_pair.setText(f"Pair {idx + 1} of {len(self._pairs)}")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._pairs) - 1)
        self._span_list.setCurrentRow(idx)

    def _on_span_selected(self, xmin: float, xmax: float, pair_idx: int):
        if pair_idx != self._pair_idx or abs(xmax - xmin) < 1e-12:
            return
        i_a, i_b, _, _ = self._pairs[pair_idx]
        key = (i_a, i_b)
        self._spans_display[key] = (min(xmin, xmax), max(xmin, xmax))
        if self._x_axis == "energy":
            lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
        else:
            lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        self._spans[key] = (lo_wl, hi_wl)
        self._refresh_span_list()
        self.statusBar().showMessage(
            f"Pair {pair_idx + 1}: span [{xmin:.5g}, {xmax:.5g}] "
            f"({'eV' if self._x_axis == 'energy' else 'nm'})."
        )

    def _on_span_item_clicked(self, item):
        if self._mode != "spans":
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self._pairs):
            self._pair_idx = idx
            self._draw_pair(idx)

    def _on_prev_pair(self):
        if self._pair_idx > 0:
            self._pair_idx -= 1
            self._draw_pair(self._pair_idx)

    def _on_next_pair(self):
        if self._pair_idx < len(self._pairs) - 1:
            self._pair_idx += 1
            self._draw_pair(self._pair_idx)

    def _on_done_spans(self):
        if self._span_selector is not None:
            try:
                self._span_selector.set_active(False)
            except Exception:
                pass
            self._span_selector = None
        self._nav_bar.setVisible(False)
        self._mode = "done"
        self._refresh_buttons()
        n_sel, n_req = len(self._spans), len(self._pairs)
        if n_sel < n_req:
            self.statusBar().showMessage(
                f"Span selection done: {n_sel}/{n_req} spans defined. "
                f"{n_req - n_sel} missing pair(s) will use range midpoint."
            )
        else:
            self.statusBar().showMessage(f"All {n_sel} span(s) defined.")
        self._draw_preview()

    # ── Steps 3–5 — Preview / Save ───────────────────────────────

    def _compute_stitch(self) -> tuple | None:
        """Build output data. Dark subtraction (if assigned) always precedes stitching.

        Returns (wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label)
        or None.
          counts_diff   — dark-subtracted, min shifted to 1
          counts_raw    — stitched without dark subtraction, no min-shift
          ds_raw        — original (un-subtracted) source datasets
          dark_by_label — {label: dark_mean_on_file_wl | None}
        """
        if not self._datasets:
            if not self._parse_all():
                return None
            self._find_pairs()

        # ── Single file ──────────────────────────────────────────
        if len(self._datasets) == 1:
            d = self._datasets[0]
            ds_raw                = [d]
            ds_sub, dark_by_label = self._apply_dark(ds_raw)
            wl_out                = ds_sub[0]["wl"]
            counts_diff           = ds_sub[0]["counts"].copy()
            counts_diff           = counts_diff - counts_diff.min() + 1.0
            counts_raw            = d["counts"].copy()
            return wl_out, counts_diff, counts_raw, ds_raw, self._hdrs[0].copy(), dark_by_label

        # ── Multiple files ────────────────────────────────────────
        n_p_list = [d["counts"].shape[1] for d in self._datasets]
        n_p      = min(n_p_list)
        if len(set(n_p_list)) > 1:
            QMessageBox.information(
                self, "Power step mismatch",
                f"Files have different power-step counts: {n_p_list}.\n"
                f"All will be truncated to {n_p} steps.",
            )
        ds_raw = [
            {"label": d["label"], "wl": d["wl"],
             "counts": d["counts"][:, :n_p], "powers": d["powers"][:n_p],
             "path": d["path"]}
            for d in self._datasets
        ]

        # Build spans_wl
        spans_wl = []
        for k, (i_a, i_b, ov_lo_wl, ov_hi_wl) in enumerate(self._pairs):
            key = (i_a, i_b)
            if key in self._spans:
                spans_wl.append(self._spans[key])
            else:
                mid = (ov_lo_wl + ov_hi_wl) / 2.0
                spans_wl.append((mid, mid))

        # Stitch dark-subtracted spectrum
        ds_for_stitch, dark_by_label = self._apply_dark(ds_raw)
        try:
            wl_out, counts_diff = _stitch_counts(ds_for_stitch, spans_wl)
        except Exception as exc:
            QMessageBox.critical(self, "Stitch error", str(exc))
            return None

        # Stitch raw spectrum (no dark subtraction)
        try:
            _, counts_raw = _stitch_counts(ds_raw, spans_wl)
        except Exception:
            counts_raw = counts_diff.copy()

        # Best header (earliest date)
        best_hdr  = self._hdrs[0].copy()
        best_date = best_hdr.get("date")
        for h in self._hdrs[1:]:
            d = h.get("date")
            if d is not None and (best_date is None or d < best_date):
                best_date = d
                best_hdr["date_str"] = h.get("date_str", best_hdr.get("date_str", ""))

        counts_diff = counts_diff - counts_diff.min() + 1.0
        return wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label

    def _on_preview_result(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, _, dark_by_label = result
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)
        is_stitched = len(ds) > 1
        dark_note   = f" (dark-subtracted: {n_dark}/{len(ds)})" if n_dark else ""
        base_title  = "Stitched spectrum" if is_stitched else "Spectrum"
        self._draw_stitched(wl_out, counts_out, ds[0]["powers"],
                            title=base_title + dark_note)
        self.statusBar().showMessage(
            f"Preview: {wl_out.size} wavelengths, "
            f"{counts_out.shape[1]} power steps{dark_note}."
        )

    def _on_stitch_save(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_diff, counts_raw, ds, best_hdr, dark_by_label = result
        n_p          = counts_diff.shape[1]
        is_stitched  = len(ds) > 1
        n_dark       = sum(1 for v in dark_by_label.values() if v is not None)

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save as HDF5", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".h5"):
            out_path += ".h5"

        spot_diam = rep_rate = None
        try:
            t = self._meta_spot_diam.text().strip()
            if t:
                spot_diam = float(t)
        except ValueError:
            pass
        try:
            t = self._meta_rep_rate.text().strip()
            if t:
                rep_rate = float(t)
        except ValueError:
            pass

        power_cal = None
        if self._cal_atbs_data is not None or self._cal_atsample_data is not None:
            power_cal = {}
            if self._cal_atbs_data is not None:
                power_cal["atBS"] = self._cal_atbs_data
            if self._cal_atsample_data is not None:
                power_cal["atSample"] = self._cal_atsample_data

        try:
            _write_h5_file(
                out_path, wl_out, counts_diff, counts_raw, best_hdr, ds[0]["powers"],
                stitched=is_stitched,
                source_datasets=ds if is_stitched else None,
                dark_by_label=dark_by_label if n_dark > 0 else None,
                spot_diameter_um=spot_diam,
                rep_rate_mhz=rep_rate,
                power_cal=power_cal,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        self.statusBar().showMessage(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} power steps{dark_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._draw_stitched(wl_out, counts_diff, ds[0]["powers"], title=title)

    def _on_save_origin(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, best_hdr, dark_by_label = result
        n_p         = counts_out.shape[1]
        is_stitched = len(ds) > 1
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save as .origin", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".origin"):
            out_path += ".origin"

        try:
            _write_origin_file(
                out_path, wl_out, counts_out, best_hdr, ds[0]["powers"]
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action    = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        self.statusBar().showMessage(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} power steps{dark_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._draw_stitched(wl_out, counts_out, ds[0]["powers"], title=title)

    def _draw_stitched(self, wl_out, counts_out, powers, title="Spectrum"):
        ax = self._canvas.reset_axes()
        ax.set_axis_on()
        n_p     = counts_out.shape[1]
        cmap_fn = cm.get_cmap("plasma", n_p)
        x       = self._wl_to_x(wl_out)
        idx     = np.argsort(x)
        for p in range(n_p):
            ax.plot(x[idx], counts_out[idx, p], lw=0.8,
                    color=cmap_fn(p), alpha=0.85)
        sm = cm.ScalarMappable(
            cmap="plasma",
            norm=mcolors.Normalize(vmin=powers[0] * 1e3, vmax=powers[-1] * 1e3),
        )
        sm.set_array([])
        self._canvas.fig.colorbar(sm, ax=ax, label="Power (mW)", pad=0.02)
        ax.set_xlabel("Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self._canvas.fig.tight_layout()
        self._canvas.draw_idle()

    # ── Helpers ──────────────────────────────────────────────────

    def _wl_to_x(self, wl):
        wl = np.asarray(wl, dtype=float)
        return _HC_EV_NM / wl if self._x_axis == "energy" else wl

    def _clear_canvas(self):
        self._canvas._welcome()

    def _refresh_buttons(self):
        has1 = len(self._files) >= 1
        has2 = len(self._files) >= 2
        self._btn_preview.setEnabled(has1)
        self._btn_spans.setEnabled(has2)
        self._btn_prev_result.setEnabled(has1)
        self._btn_stitch.setEnabled(has1)
        self._btn_save_origin.setEnabled(has1)
        self._btn_remove.setEnabled(has1)
        self._btn_clear.setEnabled(has1)
        self._btn_load_dark.setEnabled(bool(self._datasets))
        self._btn_clear_dark.setEnabled(bool(self._dark_map))


    # ── Visualizer tab ───────────────────────────────────────────

    def _build_vis_tab(self):
        root = QWidget()
        layout = QHBoxLayout(root)
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
        self._vis_rb_qty_fluence = QRadioButton("Pump fluence (µJ/cm²)")
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
        self._vis_canvas  = _MplCanvas(right)
        self._vis_toolbar = NavigationToolbar2QT(self._vis_canvas, right)
        rl.addWidget(self._vis_toolbar)
        rl.addWidget(self._vis_canvas, stretch=1)
        layout.addWidget(right, stretch=1)
        return root

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

    def _vis_load_file(self, path: str) -> dict | None:
        try:
            with h5py.File(path, "r") as f:
                wl       = (f["Wavelength"]         if "Wavelength"         in f
                            else f["wavelength_nm"]) [:]
                counts   = (f["SpectraDiff"]         if "SpectraDiff"        in f
                            else f["counts"])         [:]
                powers_W = (f["Power_uncalibrated"]  if "Power_uncalibrated" in f
                            else f["powers_W"])       [:]
                power_cal    = f["Power"]        [:] if "Power"         in f else None
                power_dens   = f["Power_density"][:] if "Power_density" in f else None
                pump_fluence = f["Pump_fluence"] [:] if "Pump_fluence"  in f else None
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
                return f"{arr[idx]:.4g} W/cm²"
        elif self._vis_rb_qty_fluence.isChecked():
            arr = d.get("pump_fluence")
            if arr is not None:
                return f"{arr[idx]:.4g} µJ/cm²"
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
        cmap_file   = cm.get_cmap("tab10")
        cmap_power  = cm.get_cmap("plasma")

        # Log-normalised alpha: low power → 0.25, high power → 1.0
        p_sorted = sorted(checked_powers)
        if len(p_sorted) > 1:
            log_lo = np.log10(max(p_sorted[0],  1e-30))
            log_hi = np.log10(max(p_sorted[-1], 1e-30))
            def _alpha(p):
                t = (np.log10(max(p, 1e-30)) - log_lo) / (log_hi - log_lo + 1e-30)
                return 0.25 + 0.75 * float(t)
        else:
            def _alpha(p):
                return 1.0

        ax = self._vis_canvas.reset_axes()
        ax.set_axis_on()

        for fi, d in enumerate(self._vis_data):
            for pi, p_W in enumerate(p_sorted):
                closest = int(np.argmin(np.abs(d["powers_W"] - p_W)))
                wl      = d["wl"]
                counts  = d["counts"][:, closest]
                x       = _HC_EV_NM / wl if x_is_energy else wl
                s       = np.argsort(x)

                if n_files == 1:
                    t     = pi / max(len(p_sorted) - 1, 1)
                    color = cmap_power(t)
                    alpha = 1.0
                else:
                    color = cmap_file(fi % 10)
                    alpha = _alpha(p_W)

                lbl = f"{d['label']} — {self._vis_fmt_qty(d, closest)}"
                ax.plot(x[s], counts[s], lw=0.9, color=color,
                        alpha=alpha, label=lbl)

        ax.set_xlabel("Energy (eV)" if x_is_energy else "Wavelength (nm)")
        ax.set_ylabel("Counts")
        ax.set_yscale("log" if self._vis_rb_log.isChecked() else "linear")
        ax.grid(True, alpha=0.3)

        if not self._vis_xauto.isChecked():
            try:
                ax.set_xlim(float(self._vis_xmin.text()),
                            float(self._vis_xmax.text()))
            except ValueError:
                pass
        if not self._vis_yauto.isChecked():
            try:
                ax.set_ylim(float(self._vis_ymin.text()),
                            float(self._vis_ymax.text()))
            except ValueError:
                pass

        n_lines = n_files * len(p_sorted)
        ax.legend(fontsize=max(4, 7 - n_lines // 5), loc="best",
                  ncol=max(1, n_lines // 20))
        ax.set_title("PL spectra")
        self._vis_canvas.fig.tight_layout()
        self._vis_canvas.draw_idle()


    # ── Analysis tab ─────────────────────────────────────────────

    def _build_analysis_tab(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        if not _NWA_AVAILABLE:
            msg = QLabel(
                f"nw_analysis could not be imported:\n{_NWA_ERROR}\n\n"
                f"Expected at: {_PLOT_DIR}"
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            return root

        # ── Scrollable sidebar ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setFixedWidth(305)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_content = QWidget()
        sl = QVBoxLayout(sidebar_content)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(6)

        # File group
        g_file = QGroupBox("HDF5 file")
        gfl = QVBoxLayout(g_file)
        self._ana_file_label = QLabel("No file loaded")
        self._ana_file_label.setWordWrap(True)
        gfl.addWidget(self._ana_file_label)
        btn_load_ana = QPushButton("Load HDF5…")
        btn_load_ana.clicked.connect(self._ana_load_file)
        gfl.addWidget(btn_load_ana)
        sl.addWidget(g_file)

        # set_startconditions group
        g_sc = QGroupBox("1  Select peaks  (set_startconditions)")
        scl = QVBoxLayout(g_sc)

        r_win = QHBoxLayout()
        r_win.addWidget(QLabel("Window width:"))
        self._ana_window_width = QLineEdit("0.05")
        r_win.addWidget(self._ana_window_width)
        scl.addLayout(r_win)

        r_xu = QHBoxLayout()
        r_xu.addWidget(QLabel("X unit:"))
        self._ana_rb_ev = QRadioButton("eV")
        self._ana_rb_nm = QRadioButton("nm")
        self._ana_rb_ev.setChecked(True)
        r_xu.addWidget(self._ana_rb_ev)
        r_xu.addWidget(self._ana_rb_nm)
        scl.addLayout(r_xu)

        r_sel = QHBoxLayout()
        r_sel.addWidget(QLabel("Spectrum:"))
        self._ana_spectrum_sel = QComboBox()
        self._ana_spectrum_sel.addItems(["maxpeak", "maxsum", "last"])
        r_sel.addWidget(self._ana_spectrum_sel)
        scl.addLayout(r_sel)

        r_ys = QHBoxLayout()
        r_ys.addWidget(QLabel("Y scale:"))
        self._ana_rb_log = QRadioButton("log")
        self._ana_rb_lin = QRadioButton("linear")
        self._ana_rb_log.setChecked(True)
        r_ys.addWidget(self._ana_rb_log)
        r_ys.addWidget(self._ana_rb_lin)
        scl.addLayout(r_ys)

        self._ana_btn_sc = QPushButton("1a  Click peaks in plot…")
        self._ana_btn_sc.setEnabled(False)
        self._ana_btn_sc.clicked.connect(self._ana_start_peak_clicking)
        scl.addWidget(self._ana_btn_sc)
        self._ana_sc_lbl = QLabel("")
        self._ana_sc_lbl.setWordWrap(True)
        scl.addWidget(self._ana_sc_lbl)
        sl.addWidget(g_sc)

        # fit_nw group
        g_fit = QGroupBox("2  Fit peaks  (fit_nw)")
        fitl = QVBoxLayout(g_fit)

        r_ff = QHBoxLayout()
        r_ff.addWidget(QLabel("Function:"))
        self._ana_fitfunc = QComboBox()
        self._ana_fitfunc.addItems(
            ["gauss1", "gauss2", "gauss3", "gauss4", "lorentz1", "lorentz2"]
        )
        r_ff.addWidget(self._ana_fitfunc)
        fitl.addLayout(r_ff)

        r_bg = QHBoxLayout()
        r_bg.addWidget(QLabel("Background:"))
        self._ana_fitbg = QComboBox()
        self._ana_fitbg.addItems(["none", "linear", "constant", "raw"])
        r_bg.addWidget(self._ana_fitbg)
        fitl.addLayout(r_bg)

        self._ana_btn_fit = QPushButton("Fit peaks")
        self._ana_btn_fit.setEnabled(False)
        self._ana_btn_fit.clicked.connect(self._ana_fit)
        fitl.addWidget(self._ana_btn_fit)
        self._ana_fit_lbl = QLabel("")
        self._ana_fit_lbl.setWordWrap(True)
        fitl.addWidget(self._ana_fit_lbl)
        sl.addWidget(g_fit)

        # integrate_spectra group
        g_int = QGroupBox("3  Integrate spectra")
        intl = QVBoxLayout(g_int)

        r_ic = QHBoxLayout()
        r_ic.addWidget(QLabel("Center:"))
        self._ana_int_center = QLineEdit("1.1")
        r_ic.addWidget(self._ana_int_center)
        intl.addLayout(r_ic)

        r_iw = QHBoxLayout()
        r_iw.addWidget(QLabel("Width:"))
        self._ana_int_width = QLineEdit("0.25")
        r_iw.addWidget(self._ana_int_width)
        intl.addLayout(r_iw)

        r_ist = QHBoxLayout()
        r_ist.addWidget(QLabel("Spec type:"))
        self._ana_int_spectype = QComboBox()
        self._ana_int_spectype.addItems(["no_background", "final", "raw"])
        r_ist.addWidget(self._ana_int_spectype)
        intl.addLayout(r_ist)

        r_im = QHBoxLayout()
        r_im.addWidget(QLabel("Method:"))
        self._ana_int_method = QComboBox()
        self._ana_int_method.addItems(["trapz", "sum", "rect"])
        r_im.addWidget(self._ana_int_method)
        intl.addLayout(r_im)

        self._ana_btn_int = QPushButton("Integrate")
        self._ana_btn_int.setEnabled(False)
        self._ana_btn_int.clicked.connect(self._ana_integrate)
        intl.addWidget(self._ana_btn_int)
        self._ana_int_lbl = QLabel("")
        self._ana_int_lbl.setWordWrap(True)
        intl.addWidget(self._ana_int_lbl)
        sl.addWidget(g_int)

        # thresholds group
        g_thr = QGroupBox("4  Find thresholds")
        thrl = QVBoxLayout(g_thr)
        self._ana_thr_area  = QCheckBox("Fit area")
        self._ana_thr_int   = QCheckBox("Integral")
        self._ana_thr_max   = QCheckBox("Maximum")
        self._ana_thr_total = QCheckBox("Total area")
        self._ana_thr_area.setChecked(True)
        for cb in (self._ana_thr_area, self._ana_thr_int,
                   self._ana_thr_max, self._ana_thr_total):
            thrl.addWidget(cb)
        self._ana_btn_thr = QPushButton("Find thresholds…")
        self._ana_btn_thr.setEnabled(False)
        self._ana_btn_thr.clicked.connect(self._ana_thresholds)
        thrl.addWidget(self._ana_btn_thr)
        self._ana_thr_lbl = QLabel("")
        self._ana_thr_lbl.setWordWrap(True)
        thrl.addWidget(self._ana_thr_lbl)
        sl.addWidget(g_thr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        btn_plot_ll = QPushButton("Plot L-L curve")
        btn_plot_ll.clicked.connect(self._ana_plot_ll)
        sl.addWidget(btn_plot_ll)

        sl.addStretch(1)
        scroll.setWidget(sidebar_content)
        layout.addWidget(scroll)

        # ── Right: toolbar + canvas + action bar ─────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._ana_canvas  = _MplCanvas(right)
        self._ana_toolbar = NavigationToolbar2QT(self._ana_canvas, right)
        rl.addWidget(self._ana_toolbar)
        rl.addWidget(self._ana_canvas, stretch=1)

        # Action bar (shown during interactive phases)
        self._ana_action_bar = QFrame()
        self._ana_action_bar.setFrameShape(QFrame.StyledPanel)
        ab = QHBoxLayout(self._ana_action_bar)
        ab.setContentsMargins(8, 4, 8, 4)
        self._ana_action_lbl    = QLabel("")
        self._ana_btn_act_done  = QPushButton("✓  Done selecting peaks")
        self._ana_btn_act_conf  = QPushButton("Confirm window")
        self._ana_btn_act_skip  = QPushButton("Skip")
        self._ana_btn_act_comp  = QPushButton("Compute threshold")
        self._ana_btn_act_next  = QPushButton("Next  ▶")
        self._ana_btn_act_done.clicked.connect(self._ana_done_clicking_peaks)
        self._ana_btn_act_conf.clicked.connect(self._ana_confirm_window)
        self._ana_btn_act_skip.clicked.connect(self._ana_skip_window)
        self._ana_btn_act_comp.clicked.connect(self._ana_compute_threshold)
        self._ana_btn_act_next.clicked.connect(self._ana_next_threshold)
        for w in (self._ana_action_lbl, self._ana_btn_act_done,
                  self._ana_btn_act_conf, self._ana_btn_act_skip,
                  self._ana_btn_act_comp, self._ana_btn_act_next):
            ab.addWidget(w)
        self._ana_action_bar.setVisible(False)
        rl.addWidget(self._ana_action_bar)

        layout.addWidget(right, stretch=1)
        return root

    # ── Analysis: file loading ────────────────────────────────────

    def _ana_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load HDF5 spectrum", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not path:
            return
        try:
            with h5py.File(path, "r") as f:
                wl       = (f["Wavelength"]         if "Wavelength"         in f
                            else f["wavelength_nm"]) [:]
                # SpectraDiff = dark-subtracted; Spectra_raw = no dark sub
                spec_diff = (f["SpectraDiff"]        if "SpectraDiff"        in f
                             else f["counts"])        [:]
                spec_raw  = (f["Spectra_raw"]        if "Spectra_raw"        in f
                             else spec_diff)          [:]  # fallback: same
                powers_W  = (f["Power_uncalibrated"] if "Power_uncalibrated" in f
                             else f["powers_W"])      [:]
                spot_diam_um = float(f.attrs["spot_diameter_um"]) \
                    if "spot_diameter_um" in f.attrs else None
                rep_rate_mhz = float(f.attrs["rep_rate_mhz"]) \
                    if "rep_rate_mhz" in f.attrs else None
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        nw = nwa._new_nanowire()
        nw.name            = os.path.splitext(os.path.basename(path))[0]
        nw.wavelength      = wl.copy()
        nw.wavelength_unit = "nm"
        nw.spectra_raw     = spec_raw.copy()
        nw.spectra_diff    = spec_diff.copy()
        nw.power           = powers_W * 1e3  # → mW
        nw.specsum         = []
        if spot_diam_um is not None:
            nw.spot_radius_short = spot_diam_um / 2.0
        if rep_rate_mhz is not None:
            nw.rep_rate = rep_rate_mhz * 1e6

        self._ana_file = path
        self._ana_nw   = nw
        self._ana_mode = "idle"
        self._ana_action_bar.setVisible(False)
        self._ana_sc_cleanup()

        n_wl, n_p = spec_diff.shape
        label = os.path.basename(path)
        self._ana_file_label.setText(
            f"{label}\n{n_wl} px, {n_p} powers\n"
            f"{wl.min():.1f}–{wl.max():.1f} nm\n"
            f"{powers_W.min()*1e3:.4g}–{powers_W.max()*1e3:.4g} mW"
        )
        self._ana_btn_sc.setEnabled(True)
        self._ana_btn_fit.setEnabled(False)
        self._ana_btn_int.setEnabled(True)
        self._ana_btn_thr.setEnabled(False)
        self._ana_sc_lbl.setText("")
        self._ana_fit_lbl.setText("")
        self._ana_int_lbl.setText("")
        self._ana_thr_lbl.setText("")
        self._ana_plot_spectrum()
        self.statusBar().showMessage(
            f"Analysis: loaded {label} ({n_wl} px, {n_p} power steps)"
        )

    # ── Analysis: spectrum overview ──────────────────────────────

    def _ana_x_of_wl(self, wl):
        """Convert nm → display units (eV or nm) for the analysis tab."""
        return _HC_EV_NM / wl if self._ana_rb_ev.isChecked() else wl

    def _ana_wl_of_x(self, x_val):
        """Convert display unit → nm for the analysis tab."""
        if self._ana_rb_ev.isChecked():
            return _HC_EV_NM / x_val
        return x_val

    def _ana_plot_spectrum(self):
        nw = self._ana_nw
        if nw is None:
            return
        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        wl     = nw.wavelength
        n_p    = nw.spectra_raw.shape[1]
        cmap_f = cm.get_cmap("plasma", n_p)
        x   = self._ana_x_of_wl(wl)
        s   = np.argsort(x)
        for p in range(n_p):
            ax.plot(x[s], nw.spectra_raw[s, p], lw=0.7,
                    color=cmap_f(p), alpha=0.8)
        sm = cm.ScalarMappable(
            cmap="plasma",
            norm=mcolors.Normalize(vmin=nw.power[0], vmax=nw.power[-1]),
        )
        sm.set_array([])
        self._ana_canvas.fig.colorbar(sm, ax=ax, label="Power (mW)", pad=0.02)
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")
        ax.set_title(nw.name)
        ax.grid(True, alpha=0.3)
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()

    # ── Analysis: peak clicking (phase 1 of set_startconditions) ─

    def _ana_start_peak_clicking(self):
        nw = self._ana_nw
        if nw is None:
            return
        self._ana_sc_cleanup()

        # Choose reference spectrum
        sel = self._ana_spectrum_sel.currentText()
        if sel == "last":
            ref = nw.spectra_raw.shape[1] - 1
        elif sel == "maxpeak":
            ref = int(np.argmax(np.max(nw.spectra_raw, axis=0)))
        else:
            ref = int(np.argmax(np.sum(nw.spectra_raw, axis=0)))
        self._ana_sc_ref_idx = ref

        # Draw reference spectrum
        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        s    = np.argsort(x)
        spec = nw.spectra_raw[:, ref]
        ax.plot(x[s], spec[s], lw=1.0, color="steelblue")
        if self._ana_rb_log.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")
        ax.set_title(f"{nw.name} — left-click peaks, then ✓ Done")
        ax.grid(True, alpha=0.3)
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()

        # Connect click handler
        self._ana_sc_clicks_x = []  # x-positions (canvas units)
        self._ana_sc_click_conn = self._ana_canvas.mpl_connect(
            "button_press_event", self._ana_on_peak_click
        )
        self._ana_mode = "sc_click_peaks"

        # Show action bar — only Done button
        self._ana_action_lbl.setText("0 peaks selected")
        self._ana_btn_act_done.setVisible(True)
        self._ana_btn_act_conf.setVisible(False)
        self._ana_btn_act_skip.setVisible(False)
        self._ana_btn_act_comp.setVisible(False)
        self._ana_btn_act_next.setVisible(False)
        self._ana_action_bar.setVisible(True)
        self.statusBar().showMessage(
            "Left-click peak positions in the plot, then click ✓ Done."
        )

    def _ana_on_peak_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if self._ana_mode != "sc_click_peaks":
            return
        x_click = event.xdata
        self._ana_sc_clicks_x.append(x_click)
        n = len(self._ana_sc_clicks_x)
        self._ana_action_lbl.setText(f"{n} peak(s) selected")
        # Mark the click on the canvas
        ax = self._ana_canvas.ax
        ax.axvline(x_click, color="red", lw=1.0, linestyle="--", alpha=0.7)
        self._ana_canvas.draw_idle()

    def _ana_sc_cleanup(self):
        """Disconnect any active click handler or SpanSelector."""
        if self._ana_sc_click_conn is not None:
            try:
                self._ana_canvas.mpl_disconnect(self._ana_sc_click_conn)
            except Exception:
                pass
            self._ana_sc_click_conn = None
        if self._ana_sc_span is not None:
            try:
                self._ana_sc_span.set_active(False)
            except Exception:
                pass
            self._ana_sc_span = None

    def _ana_done_clicking_peaks(self):
        if self._ana_mode != "sc_click_peaks":
            return
        self._ana_sc_cleanup()

        if not self._ana_sc_clicks_x:
            self._ana_sc_lbl.setText("No peaks clicked — try again.")
            self._ana_action_bar.setVisible(False)
            self._ana_mode = "idle"
            return

        # Sort clicks in increasing x
        self._ana_sc_clicks_x = sorted(self._ana_sc_clicks_x)
        self._ana_sc_windows  = [None] * len(self._ana_sc_clicks_x)
        self._ana_sc_peak_idx = 0
        self._ana_mode = "sc_window"
        self._ana_show_window_for_peak(0)

    # ── Analysis: fit-window selection (phase 2) ─────────────────

    def _ana_show_window_for_peak(self, j):
        nw   = self._ana_nw
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        s    = np.argsort(x)
        ref  = self._ana_sc_ref_idx
        spec = nw.spectra_raw[:, ref]

        x_peak = self._ana_sc_clicks_x[j]
        try:
            window_half = float(self._ana_window_width.text()) / 2
        except ValueError:
            window_half = 0.025

        # Zoom window for display
        x_lo = x_peak - window_half * 3
        x_hi = x_peak + window_half * 3

        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        mask = (x[s] >= x_lo) & (x[s] <= x_hi)
        ax.plot(x[s][mask], spec[s][mask], "x-", lw=1.0, color="steelblue")
        if self._ana_rb_log.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")
        n_tot = len(self._ana_sc_clicks_x)
        ax.set_title(
            f"{nw.name} — peak {j+1}/{n_tot}\n"
            "Drag to select fit window, then Confirm"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lo, x_hi)
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()

        # SpanSelector
        self._ana_sc_span_sel = [None]  # stores (xmin, xmax)
        self._ana_sc_span = SpanSelector(
            ax,
            lambda xmin, xmax: self._ana_sc_span_sel.__setitem__(0, (xmin, xmax)),
            "horizontal",
            useblit=False,
            interactive=True,
            props=dict(facecolor="tab:green", alpha=0.22),
            handle_props=dict(color="darkgreen"),
        )

        n_done = j
        self._ana_action_lbl.setText(
            f"Peak {j+1} of {n_tot}  ({n_done} confirmed)"
        )
        self._ana_btn_act_done.setVisible(False)
        self._ana_btn_act_conf.setVisible(True)
        self._ana_btn_act_skip.setVisible(True)
        self._ana_btn_act_comp.setVisible(False)
        self._ana_btn_act_next.setVisible(False)
        self._ana_action_bar.setVisible(True)
        self.statusBar().showMessage(
            f"Drag to select fit window for peak {j+1}, then Confirm."
        )

    def _ana_confirm_window(self):
        if self._ana_mode != "sc_window":
            return
        j = self._ana_sc_peak_idx
        sel = self._ana_sc_span_sel[0] if hasattr(self, "_ana_sc_span_sel") else None
        if sel is not None and abs(sel[1] - sel[0]) > 1e-12:
            self._ana_sc_windows[j] = sel
        else:
            # Fallback: use window_width centred on the click
            try:
                hw = float(self._ana_window_width.text()) / 2
            except ValueError:
                hw = 0.025
            cx = self._ana_sc_clicks_x[j]
            self._ana_sc_windows[j] = (cx - hw, cx + hw)
        self._ana_sc_cleanup()
        self._ana_advance_window()

    def _ana_skip_window(self):
        if self._ana_mode != "sc_window":
            return
        j = self._ana_sc_peak_idx
        try:
            hw = float(self._ana_window_width.text()) / 2
        except ValueError:
            hw = 0.025
        cx = self._ana_sc_clicks_x[j]
        self._ana_sc_windows[j] = (cx - hw, cx + hw)
        self._ana_sc_cleanup()
        self._ana_advance_window()

    def _ana_advance_window(self):
        j = self._ana_sc_peak_idx + 1
        self._ana_sc_peak_idx = j
        n_tot = len(self._ana_sc_clicks_x)
        if j < n_tot:
            self._ana_show_window_for_peak(j)
        else:
            self._ana_finish_startconditions()

    def _ana_finish_startconditions(self):
        """Store start_conditions in the nw object and update UI."""
        self._ana_sc_cleanup()
        self._ana_action_bar.setVisible(False)
        self._ana_mode = "idle"

        nw   = self._ana_nw
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        n    = len(self._ana_sc_clicks_x)
        ref  = self._ana_sc_ref_idx

        sc = np.full((3, n), np.nan)
        for j in range(n):
            cx = self._ana_sc_clicks_x[j]
            peak_idx = int(np.argmin(np.abs(x - cx)))
            xmin, xmax = self._ana_sc_windows[j]
            i1 = int(np.argmin(np.abs(x - xmin)))
            i2 = int(np.argmin(np.abs(x - xmax)))
            fitwindow = max(abs(i2 - i1), 4)
            sc[:, j] = [peak_idx, fitwindow, ref]

        nw.start_conditions = sc
        nw.n_sel_peaks      = n

        self._ana_sc_lbl.setText(f"{n} peak(s) selected")
        self._ana_btn_fit.setEnabled(True)
        self._ana_btn_thr.setEnabled(False)
        self.statusBar().showMessage(
            f"Analysis: {n} peak(s) configured. Ready to fit."
        )
        self._ana_draw_peaks()

    def _ana_draw_peaks(self):
        nw = self._ana_nw
        if nw is None or nw.start_conditions is None:
            return
        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        wl  = nw.wavelength
        x   = self._ana_x_of_wl(wl)
        s   = np.argsort(x)
        ref = int(nw.start_conditions[2, 0])
        ax.plot(x[s], nw.spectra_raw[s, ref], lw=1.0, color="steelblue")
        for k in range(nw.n_sel_peaks):
            pi = int(nw.start_conditions[0, k])
            ax.axvline(x[pi], color="red", lw=1.0, linestyle="--",
                       label=f"peak {k+1}: {x[pi]:.4g}")
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")
        ax.set_title(f"{nw.name} — {nw.n_sel_peaks} selected peak(s)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()

    # ── Analysis: fit_nw ─────────────────────────────────────────

    def _ana_fit(self):
        nw = self._ana_nw
        if nw is None or nw.start_conditions is None or nw.n_sel_peaks == 0:
            QMessageBox.warning(self, "No peaks",
                                "Run set_startconditions first.")
            return
        fitfunction = self._ana_fitfunc.currentText()
        subtract_bg = self._ana_fitbg.currentText()

        self.statusBar().showMessage("Fitting… please wait.")
        QApplication.processEvents()
        try:
            nwa.fit_nw(nw,
                       subtract_fit_background=subtract_bg,
                       fitfunction=fitfunction,
                       show_progress=False)
        except Exception as exc:
            QMessageBox.critical(self, "fit_nw error", str(exc))
            return

        n_ok = sum(
            1 for j in range(nw.n_sel_peaks)
            for i in range(len(nw.power))
            if nw.fits[j][i] is not None
        )
        n_total = nw.n_sel_peaks * len(nw.power)
        self._ana_fit_lbl.setText(
            f"{fitfunction} fit done\n{n_ok}/{n_total} converged"
        )
        self._ana_btn_thr.setEnabled(True)
        self.statusBar().showMessage(
            f"Analysis: fit complete ({n_ok}/{n_total} converged)."
        )
        self._ana_plot_ll()

    # ── Analysis: integrate_spectra ──────────────────────────────

    def _ana_integrate(self):
        nw = self._ana_nw
        if nw is None:
            return
        try:
            center = float(self._ana_int_center.text())
            width  = float(self._ana_int_width.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input",
                                "Center and Width must be numbers.")
            return
        spectype = self._ana_int_spectype.currentText()
        method   = self._ana_int_method.currentText()
        try:
            values = nwa.integrate_spectra(
                nw, center=center, width=width,
                spectrumtype=spectype, method=method
            )
        except Exception as exc:
            QMessageBox.critical(self, "integrate_spectra error", str(exc))
            return
        if values is None:
            self._ana_int_lbl.setText("No spectra available")
            return
        n_valid = int(np.sum(~np.isnan(values)))
        self._ana_int_lbl.setText(
            f"Done (center={center:.4g}, width={width:.4g})\n"
            f"{n_valid}/{len(values)} valid"
        )
        self._ana_btn_thr.setEnabled(True)
        self.statusBar().showMessage(
            f"Analysis: integration done at {center:.4g} ± {width/2:.4g}."
        )
        self._ana_plot_ll()

    # ── Analysis: thresholds (native Qt version) ─────────────────

    def _ana_thresholds(self):
        nw = self._ana_nw
        if nw is None:
            return
        mode = []
        if self._ana_thr_area.isChecked():  mode.append("area")
        if self._ana_thr_int.isChecked():   mode.append("integral")
        if self._ana_thr_max.isChecked():   mode.append("max")
        if self._ana_thr_total.isChecked(): mode.append("total")
        if not mode:
            QMessageBox.warning(self, "No mode",
                                "Select at least one threshold mode.")
            return

        n_pk = nw.n_sel_peaks
        n_pw = len(nw.power)

        def _cell_to_mat(cell):
            mat = np.full((n_pk, n_pw), np.nan)
            for j in range(n_pk):
                for i in range(n_pw):
                    v = cell[j][i] if cell is not None else None
                    if v is not None and np.any(~np.isnan(v)):
                        mat[j, i] = float(np.nanmax(v))
            return mat

        # Build queue: list of (title, power_arr, values_arr)
        queue = []
        if "area" in mode and nw.peak_area is not None:
            mat = _cell_to_mat(nw.peak_area)
            for j in range(n_pk):
                vals = mat[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} fit-area", nw.power, vals))
        if "integral" in mode and nw.peak_integral is not None:
            for j in range(n_pk):
                vals = nw.peak_integral[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} integral", nw.power, vals))
        if "max" in mode and nw.peak_maximum is not None:
            for j in range(n_pk):
                vals = nw.peak_maximum[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} maximum", nw.power, vals))
        if "total" in mode and nw.total_peak_area is not None:
            vals = nw.total_peak_area
            if np.any(~np.isnan(vals)):
                queue.append(("total peak area", nw.power, vals))

        if not queue:
            QMessageBox.information(self, "Nothing to fit",
                                    "No fit results available yet.\n"
                                    "Run Fit peaks or Integrate first.")
            return

        self._ana_thr_queue   = queue
        self._ana_thr_idx     = 0
        self._ana_thr_results = {}      # {title: (thr, thr_err, slope, slope_err)}
        self._ana_thr_sel_pts = []      # selected indices for current item
        self._ana_thr_coeffs  = None    # np.polyfit result for current item
        self._ana_mode = "thr_select"
        self._ana_show_thr_item(0)

    def _ana_show_thr_item(self, idx):
        title, power, values = self._ana_thr_queue[idx]
        n_tot = len(self._ana_thr_queue)

        # Clean up any previous rect selector
        if self._ana_thr_rect_sel is not None:
            try:
                self._ana_thr_rect_sel.set_active(False)
            except Exception:
                pass
            self._ana_thr_rect_sel = None

        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        valid = ~np.isnan(values)
        self._ana_thr_scatter = ax.scatter(
            power[valid], values[valid], zorder=3, color="steelblue", s=20
        )
        ax.set_xlabel("Power (mW)")
        ax.set_ylabel("Intensity (arb.u.)")
        ax.set_title(
            f"{title}  ({idx+1}/{n_tot})\n"
            "Drag rectangle to select ASE region, then Compute threshold"
        )
        ax.grid(True, alpha=0.3)
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()

        # Store current item's data for use in compute/next
        self._ana_thr_power  = power[valid]
        self._ana_thr_values = values[valid]
        self._ana_thr_sel_pts = list(range(len(self._ana_thr_power)))  # default=all
        self._ana_thr_coeffs  = None
        self._ana_thr_title   = title

        from matplotlib.widgets import RectangleSelector
        self._ana_thr_rect_sel = RectangleSelector(
            ax,
            self._ana_on_thr_rect,
            useblit=False,
            button=[1],
            minspanx=0, minspany=0,
            spancoords="data",
            interactive=True,
        )

        self._ana_action_lbl.setText(f"Threshold {idx+1}/{n_tot}: {title}")
        self._ana_btn_act_done.setVisible(False)
        self._ana_btn_act_conf.setVisible(False)
        self._ana_btn_act_skip.setVisible(False)
        self._ana_btn_act_comp.setVisible(True)
        self._ana_btn_act_next.setVisible(True)
        self._ana_btn_act_next.setText(
            "Skip" if idx + 1 < len(self._ana_thr_queue) else "Finish"
        )
        self._ana_action_bar.setVisible(True)
        self.statusBar().showMessage(
            f"Drag to select ASE region for {title}, then Compute threshold."
        )

    def _ana_on_thr_rect(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata,  erelease.xdata])
        y0, y1 = sorted([eclick.ydata,  erelease.ydata])
        p, v = self._ana_thr_power, self._ana_thr_values
        inside = np.where(
            (p >= x0) & (p <= x1) & (v >= y0) & (v <= y1)
        )[0]
        self._ana_thr_sel_pts = inside.tolist()
        # Highlight selected points
        ax = self._ana_canvas.ax
        colors = ["steelblue"] * len(p)
        for i in self._ana_thr_sel_pts:
            colors[i] = "tomato"
        self._ana_thr_scatter.set_facecolor(colors)
        self._ana_canvas.draw_idle()

    def _ana_compute_threshold(self):
        if self._ana_mode != "thr_select":
            return
        sel = self._ana_thr_sel_pts
        p_all, v_all = self._ana_thr_power, self._ana_thr_values
        if len(sel) < 2:
            sel = list(range(len(p_all)))
        p_sel = p_all[sel]
        v_sel = v_all[sel]
        valid = ~(np.isnan(p_sel) | np.isnan(v_sel))
        p_sel, v_sel = p_sel[valid], v_sel[valid]
        if len(p_sel) < 2:
            self.statusBar().showMessage("Not enough points for fit.")
            return

        coeffs = np.polyfit(p_sel, v_sel, 1)
        thresh = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else np.nan
        self._ana_thr_coeffs = coeffs
        self._ana_thr_results[self._ana_thr_title] = (thresh, coeffs[0])

        # Redraw with fit line
        ax = self._ana_canvas.ax
        x_fit = np.linspace(0, max(p_all) * 1.1, 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r-", lw=1.5, label="fit")
        if not np.isnan(thresh):
            ax.axvline(thresh, color="black", lw=1.2, linestyle="--",
                       label=f"Threshold = {thresh:.4g} mW")
        ax.legend(fontsize=8)
        ax.set_title(
            f"{self._ana_thr_title}\n"
            f"Threshold = {thresh:.4g} mW  |  slope = {coeffs[0]:.3g}"
        )
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()
        self.statusBar().showMessage(
            f"Threshold ({self._ana_thr_title}): {thresh:.4g} mW"
        )

    def _ana_next_threshold(self):
        if self._ana_mode != "thr_select":
            return
        if self._ana_thr_rect_sel is not None:
            try:
                self._ana_thr_rect_sel.set_active(False)
            except Exception:
                pass
            self._ana_thr_rect_sel = None

        idx = self._ana_thr_idx + 1
        self._ana_thr_idx = idx
        if idx < len(self._ana_thr_queue):
            self._ana_show_thr_item(idx)
        else:
            self._ana_finish_thresholds()

    def _ana_finish_thresholds(self):
        if self._ana_thr_rect_sel is not None:
            try:
                self._ana_thr_rect_sel.set_active(False)
            except Exception:
                pass
            self._ana_thr_rect_sel = None
        self._ana_action_bar.setVisible(False)
        self._ana_mode = "idle"

        # Store results back into nw
        nw = self._ana_nw
        results = self._ana_thr_results
        parts = []
        for title, (thr, slope) in results.items():
            if not np.isnan(thr):
                parts.append(f"{title}: {thr:.4g} mW")
                # Try to map into nw fields by title prefix
                if "integral" in title:
                    nw.threshold_integral = thr
                    nw.slope_integral     = slope
                elif "maximum" in title or "max" in title:
                    nw.threshold_max = thr
                    nw.slope_max     = slope
                elif "total" in title:
                    nw.threshold = np.array([thr])
                    nw.slope     = np.array([slope])
                else:  # fit-area
                    nw.threshold = np.array(
                        [thr] if nw.threshold is None
                        else np.append(np.atleast_1d(nw.threshold), thr)
                    )

        self._ana_thr_lbl.setText("\n".join(parts) if parts else "Done")
        self.statusBar().showMessage("Analysis: thresholds done.")
        self._ana_plot_ll()

    # ── Analysis: L-L plot ───────────────────────────────────────

    def _ana_plot_ll(self):
        nw = self._ana_nw
        if nw is None:
            return
        ax = self._ana_canvas.reset_axes()
        ax.set_axis_on()
        power   = nw.power
        plotted = False

        if nw.peak_integral is not None:
            for j in range(nw.n_sel_peaks):
                vals  = nw.peak_integral[j, :]
                valid = ~np.isnan(vals)
                if valid.any():
                    ax.plot(power[valid], vals[valid], "o-", lw=1.2,
                            label=f"peak {j+1} integral")
                    plotted = True

        if nw.specsum:
            for entry in nw.specsum:
                vals  = entry["values"]
                valid = ~np.isnan(vals)
                if valid.any():
                    c = entry["center"]; w = entry["width"]
                    ax.plot(power[valid], vals[valid], "s--", lw=1.2,
                            label=f"integrate @ {c:.4g} ±{w/2:.3g}")
                    plotted = True

        def _vline(thr, lbl):
            if thr is not None:
                t = float(np.nanmean(np.atleast_1d(thr)))
                if not np.isnan(t):
                    ax.axvline(t, linestyle=":", color="red", lw=1.2, label=lbl)

        _vline(nw.threshold,          "thr (area)")
        _vline(nw.threshold_integral, "thr (int)")
        _vline(nw.threshold_max,      "thr (max)")

        if not plotted:
            ax.text(0.5, 0.5, "No results yet.\nRun Fit or Integrate first.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888888")
        else:
            ax.set_xlabel("Power (mW)")
            ax.set_ylabel("Intensity (arb.u.)")
            ax.set_title(f"{nw.name} — L-L curve")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()


# ------------------------------------------------------------------
# Spotsize tab
# ------------------------------------------------------------------

    def _build_spotsize_tab(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Sidebar ───────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        g_file = QGroupBox("Input data")
        fl = QVBoxLayout(g_file)
        self._ss_file_label = QLabel("No file loaded")
        self._ss_file_label.setWordWrap(True)
        fl.addWidget(self._ss_file_label)
        btn_load = QPushButton("Load Excel…")
        btn_load.clicked.connect(self._ss_load_file)
        fl.addWidget(btn_load)
        self._ss_is_deriv = QCheckBox("Column 2 is already a derivative")
        fl.addWidget(self._ss_is_deriv)
        sl.addWidget(g_file)

        g_par = QGroupBox("Parameters")
        pl2 = QVBoxLayout(g_par)
        r_ang = QHBoxLayout()
        r_ang.addWidget(QLabel("Angle (°):"))
        self._ss_angle = QLineEdit("45.0")
        r_ang.addWidget(self._ss_angle)
        pl2.addLayout(r_ang)
        sl.addWidget(g_par)

        g_span = QGroupBox("Span selection")
        spl2 = QVBoxLayout(g_span)
        spl2.addWidget(QLabel(
            "Drag on the plot to select a span.\n"
            "Both spans are needed before analysis."
        ))
        r_sp = QHBoxLayout()
        self._ss_rb_span1 = QRadioButton("Span 1")
        self._ss_rb_span2 = QRadioButton("Span 2")
        self._ss_rb_span1.setChecked(True)
        self._ss_rb_span1.toggled.connect(
            lambda checked: self._ss_switch_span(0) if checked else None
        )
        self._ss_rb_span2.toggled.connect(
            lambda checked: self._ss_switch_span(1) if checked else None
        )
        r_sp.addWidget(self._ss_rb_span1)
        r_sp.addWidget(self._ss_rb_span2)
        spl2.addLayout(r_sp)
        self._ss_span1_lbl = QLabel("Span 1: not set")
        self._ss_span2_lbl = QLabel("Span 2: not set")
        spl2.addWidget(self._ss_span1_lbl)
        spl2.addWidget(self._ss_span2_lbl)
        btn_clear = QPushButton("Clear spans")
        btn_clear.clicked.connect(self._ss_clear_spans)
        spl2.addWidget(btn_clear)
        sl.addWidget(g_span)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._ss_btn_analyze = QPushButton("Analyze")
        self._ss_btn_analyze.setEnabled(False)
        bold = QFont()
        bold.setBold(True)
        self._ss_btn_analyze.setFont(bold)
        self._ss_btn_analyze.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._ss_btn_analyze.setMinimumHeight(32)
        self._ss_btn_analyze.clicked.connect(self._ss_analyze)
        sl.addWidget(self._ss_btn_analyze)

        self._ss_result_lbl = QLabel("")
        self._ss_result_lbl.setWordWrap(True)
        sl.addWidget(self._ss_result_lbl)

        sl.addStretch(1)
        layout.addWidget(sidebar)

        # ── Right: canvas ─────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._ss_canvas  = _MplCanvas(right)
        self._ss_toolbar = NavigationToolbar2QT(self._ss_canvas, right)
        # Replace default welcome text
        self._ss_canvas.reset_axes()
        self._ss_canvas.ax.set_axis_off()
        self._ss_canvas.ax.text(
            0.5, 0.5, "Load an Excel file to begin.",
            ha="center", va="center", transform=self._ss_canvas.ax.transAxes,
            fontsize=12, color="#666666",
        )
        self._ss_canvas.draw_idle()
        rl.addWidget(self._ss_toolbar)
        rl.addWidget(self._ss_canvas, stretch=1)
        layout.addWidget(right, stretch=1)
        return root

    # ── Spotsize: helpers ─────────────────────────────────────────

    def _ss_setup_canvas(self):
        """Plot current derivative on canvas and install a fresh SpanSelector."""
        ax = self._ss_canvas.reset_axes()
        ax.plot(self._ss_x, self._ss_deriv, color="tab:blue", lw=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("dy/dx")
        ax.set_title("Select two spans — one per Gaussian edge")
        ax.grid(True, alpha=0.4)

        # Reapply existing span patches
        _SPAN_COLORS = ["tab:green", "tab:orange"]
        self._ss_span_patches = [None, None]
        for i, sp in enumerate(self._ss_spans):
            if sp is not None:
                xmin, xmax = sp
                self._ss_span_patches[i] = ax.axvspan(
                    xmin, xmax, alpha=0.25, color=_SPAN_COLORS[i]
                )

        self._ss_canvas.fig.tight_layout()
        self._ss_canvas.draw_idle()

        self._ss_selector = SpanSelector(
            ax, self._ss_on_span_select, "horizontal", useblit=False,
            props=dict(alpha=0.15, facecolor="lightyellow"),
        )

    def _ss_on_span_select(self, xmin, xmax):
        idx = 0 if self._ss_rb_span1.isChecked() else 1
        _SPAN_COLORS = ["tab:green", "tab:orange"]

        if self._ss_span_patches[idx] is not None:
            try:
                self._ss_span_patches[idx].remove()
            except ValueError:
                pass

        self._ss_spans[idx] = (xmin, xmax)
        ax = self._ss_canvas.ax
        self._ss_span_patches[idx] = ax.axvspan(
            xmin, xmax, alpha=0.25, color=_SPAN_COLORS[idx]
        )
        lbls = [self._ss_span1_lbl, self._ss_span2_lbl]
        lbls[idx].setText(f"Span {idx + 1}: [{xmin:.5g}, {xmax:.5g}]")

        # Auto-advance to the other span if unset
        if idx == 0 and self._ss_spans[1] is None:
            self._ss_rb_span2.setChecked(True)
        elif idx == 1 and self._ss_spans[0] is None:
            self._ss_rb_span1.setChecked(True)

        self._ss_canvas.draw_idle()
        if all(s is not None for s in self._ss_spans):
            self._ss_btn_analyze.setEnabled(True)

    def _ss_switch_span(self, idx):
        """Called when a span radio button is toggled (no-op placeholder)."""
        pass

    # ── Spotsize: actions ─────────────────────────────────────────

    def _ss_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select scan file", "",
            "Scan files (*.xlsx *.xls *.txt *.dat *.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                import pandas as pd
                df = pd.read_excel(path, header=None)
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
            else:
                data = np.loadtxt(path)
                x = data[:, 0]
                y = data[:, 1]
            order = np.argsort(x)
            x, y = x[order], y[order]
            if x.size < 10:
                QMessageBox.warning(
                    self, "Too few data points",
                    f"Only {x.size} data points after cleaning. Check the file."
                )
                return
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        self._ss_x     = x
        self._ss_deriv = y if self._ss_is_deriv.isChecked() else np.gradient(y, x)
        self._ss_spans = [None, None]
        self._ss_span1_lbl.setText("Span 1: not set")
        self._ss_span2_lbl.setText("Span 2: not set")
        self._ss_result_lbl.setText("")
        self._ss_btn_analyze.setEnabled(False)
        self._ss_rb_span1.setChecked(True)
        self._ss_file_label.setText(
            f"{os.path.basename(path)}\n"
            f"{x.size} points  x = [{x.min():.4g}, {x.max():.4g}]"
        )
        self._ss_setup_canvas()

    def _ss_clear_spans(self):
        for i, patch in enumerate(self._ss_span_patches):
            if patch is not None:
                try:
                    patch.remove()
                except ValueError:
                    pass
        self._ss_span_patches = [None, None]
        self._ss_spans        = [None, None]
        self._ss_span1_lbl.setText("Span 1: not set")
        self._ss_span2_lbl.setText("Span 2: not set")
        self._ss_btn_analyze.setEnabled(False)
        self._ss_result_lbl.setText("")
        self._ss_canvas.draw_idle()

    def _ss_analyze(self):
        from scipy.optimize import curve_fit
        from scipy.integrate import trapezoid

        x, deriv, spans = self._ss_x, self._ss_deriv, self._ss_spans
        if x is None or any(s is None for s in spans):
            return

        try:
            angle = float(self._ss_angle.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid angle",
                                "Please enter a valid angle in degrees.")
            return

        combined_mask = np.zeros(x.size, dtype=bool)
        for xmin, xmax in spans:
            combined_mask |= (x >= xmin) & (x <= xmax)

        xs, ds = x[combined_mask], deriv[combined_mask]
        if xs.size < 5:
            QMessageBox.warning(
                self, "Too few points",
                f"Combined spans contain only {xs.size} point(s) — "
                "widen the spans."
            )
            return

        peak_idx  = int(np.argmax(np.abs(ds)))
        span_width = max(s[1] - s[0] for s in spans)
        p0 = [ds[peak_idx], xs[peak_idx], span_width / 4.0]
        try:
            popt, _ = curve_fit(_gaussian, xs, ds, p0=p0, maxfev=10_000)
        except RuntimeError:
            QMessageBox.warning(self, "Fit failed",
                                "Gaussian fit failed to converge. "
                                "Try adjusting the selected spans.")
            return

        amp, mu, sigma = popt[0], popt[1], abs(popt[2])
        x_lo, x_hi    = _find_symmetric_95_bounds(x, deriv, mu)
        delta          = x_hi - x_lo
        delta_corr     = delta * np.sin(np.radians(angle))

        integral_total  = trapezoid(deriv, x)
        bounds_mask     = (x >= x_lo) & (x <= x_hi)
        integral_bounds = trapezoid(deriv[bounds_mask], x[bounds_mask])

        # ── Draw result ───────────────────────────────────────────
        ax = self._ss_canvas.reset_axes()
        ax.plot(x, deriv, color="tab:blue", lw=1.5, label="dy/dx", zorder=3)
        ax.fill_between(
            x[bounds_mask], deriv[bounds_mask], alpha=0.30, color="tab:green",
            label=f"95.4 % area  [{x_lo:.4g}, {x_hi:.4g}]",
        )
        x_fit = np.linspace(x_lo, x_hi, 500)
        ax.plot(
            x_fit, _gaussian(x_fit, amp, mu, sigma),
            "--", color="tab:green", lw=2,
            label=f"Gaussian  μ={mu:.5g}  σ={sigma:.5g}",
        )
        ax.axvline(mu, color="tab:green", lw=1.2, ls=":", zorder=4)

        # Arrow annotation
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.25 * (y_hi - y_lo))
        y_lo, y_hi = ax.get_ylim()
        span_y  = y_hi - y_lo
        y_arrow = y_hi - 0.08 * span_y
        ax.annotate(
            "", xy=(x_hi, y_arrow), xytext=(x_lo, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
        )
        ax.text(
            mu, y_arrow + 0.02 * span_y,
            f"Δx = {delta:.5g}  |  Δx × sin({angle}°) = {delta_corr:.5g}",
            ha="center", va="bottom", color="red", fontsize=10, fontweight="bold",
        )

        # Reapply span patches on new axes
        _SPAN_COLORS = ["tab:green", "tab:orange"]
        self._ss_span_patches = [None, None]
        for i, sp in enumerate(self._ss_spans):
            if sp is not None:
                self._ss_span_patches[i] = ax.axvspan(
                    sp[0], sp[1], alpha=0.15, color=_SPAN_COLORS[i],
                    label=f"Span {i + 1}",
                )

        ax.set_xlabel("x")
        ax.set_ylabel("dy/dx")
        ax.set_title(f"Δx × sin({angle}°) = {delta_corr:.5g}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        self._ss_canvas.fig.tight_layout()
        self._ss_canvas.draw_idle()

        # Reinstall SpanSelector on new axes
        self._ss_selector = SpanSelector(
            ax, self._ss_on_span_select, "horizontal", useblit=False,
            props=dict(alpha=0.15, facecolor="lightyellow"),
        )

        self._ss_result_lbl.setText(
            f"μ = {mu:.6g}\n"
            f"σ = {sigma:.6g}\n"
            f"95.4 % bounds: [{x_lo:.5g}, {x_hi:.5g}]\n"
            f"Δx = {delta:.6g}\n"
            f"Δx × sin({angle}°) = {delta_corr:.6g}\n"
            f"∫ total = {integral_total:.4g}\n"
            f"∫ window = {integral_bounds:.4g}"
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PL Spectrum Stitcher / Converter")
    app.setStyle("Fusion")
    window = StitchApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
