import os
import sys
import numpy as np
import h5py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QComboBox, QCheckBox, QLineEdit,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QFrame,
    QApplication,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import RectangleSelector
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from canvas import _MplCanvas

_PL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PL Helper")
)
if _PL_DIR not in sys.path:
    sys.path.insert(0, _PL_DIR)
from pl import _HC_EV_NM

_PLOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Plot")
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


class AnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Analysis state ────────────────────────────────────────
        self._ana_file = None
        self._ana_nw   = None   # nw_analysis SimpleNamespace
        self._ana_mode = "idle"
        # peak clicking
        self._ana_sc_ref_idx    = 0
        self._ana_sc_clicks_x   = []
        self._ana_sc_click_conn = None
        self._ana_sc_span       = None
        self._ana_sc_span_sel   = [None]
        self._ana_sc_windows    = []
        self._ana_sc_peak_idx   = 0
        # thresholds
        self._ana_thr_queue    = []
        self._ana_thr_idx      = 0
        self._ana_thr_results  = {}
        self._ana_thr_rect_sel = None
        self._ana_thr_sel_pts  = []
        self._ana_thr_coeffs   = None
        self._ana_thr_power    = None
        self._ana_thr_values   = None
        self._ana_thr_title    = ""
        self._ana_thr_scatter  = None

        # ── Build UI (body of original _build_analysis_tab) ───────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        if not _NWA_AVAILABLE:
            msg = QLabel(
                f"nw_analysis could not be imported:\n{_NWA_ERROR}\n\n"
                f"Expected at: {_PLOT_DIR}"
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            return

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
        self._ana_spectrum_sel.addItems(["last", "maxpeak", "maxsum"])
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
        self._ana_fitbg.addItems(["linear", "none", "constant", "raw"])
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

        # thresholds group
        g_thr = QGroupBox("3  Find thresholds")
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

        self._ana_btn_save = QPushButton("Save results to HDF5")
        self._ana_btn_save.setEnabled(False)
        self._ana_btn_save.clicked.connect(self._ana_save)
        sl.addWidget(self._ana_btn_save)

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
                if "Energy" in f:
                    wl = _HC_EV_NM / f["Energy"][:]     # eV → nm for internal use
                elif "Wavelength" in f:
                    wl = f["Wavelength"][:]
                else:
                    wl = f["wavelength_nm"][:]
                # SpectraDiff = dark-subtracted; Spectra_raw = no dark sub
                spec_diff = (f["SpectraDiff"]        if "SpectraDiff"        in f
                             else f["counts"])        [:]
                spec_raw  = (f["Spectra_raw"]        if "Spectra_raw"        in f
                             else spec_diff)          [:]  # fallback: same
                pwr_ana_ds = (f["Power_uncalibrated"] if "Power_uncalibrated" in f
                              else f["powers_W"])
                powers_W   = pwr_ana_ds[:].astype(float)
                _pwr_units = pwr_ana_ds.attrs.get("units", "W")
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
        powers_mW          = powers_W if _pwr_units == "mW" else powers_W * 1e3
        nw.power           = powers_mW
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
            f"{powers_mW.min():.4g}–{powers_mW.max():.4g} mW"
        )
        self._ana_btn_sc.setEnabled(True)
        self._ana_btn_fit.setEnabled(False)
        self._ana_btn_thr.setEnabled(False)
        self._ana_btn_save.setEnabled(True)
        self._ana_sc_lbl.setText("")
        self._ana_fit_lbl.setText("")
        self._ana_thr_lbl.setText("")
        self._ana_plot_spectrum()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
        # Store the (xmin, xmax) bounds in display units so _ana_integrate
        # can use them instead of the manual center/width fields.
        nw.fit_windows = list(self._ana_sc_windows)

        self._ana_sc_lbl.setText(f"{n} peak(s) selected")
        self._ana_btn_fit.setEnabled(True)
        self._ana_btn_thr.setEnabled(False)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Fitting… please wait.")
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
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Analysis: fit complete ({n_ok}/{n_total} converged). Integrating…"
            )
        QApplication.processEvents()
        self._ana_integrate()

    # ── Analysis: integrate_spectra ──────────────────────────────

    def _ana_integrate(self):
        nw = self._ana_nw
        if nw is None:
            return
        spectype = "no_background"
        method   = "trapz"

        # Reset specsum so repeated clicks don't accumulate entries.
        nw.specsum = []

        if nw.start_conditions is not None:
            # ── Per-peak tracking path (mirrors MATLAB fit_nw) ───────
            # Window width is fixed per peak; center follows the peak as
            # it shifts spectrally across power steps.
            n_peaks  = nw.n_sel_peaks
            n_wl     = nw.wavelength.shape[0]
            n_powers = len(nw.power)
            wl       = nw.wavelength  # nm, shape (n_wl,)

            spectra = nw.spectra_raw if spectype == 'raw' else nw.spectra_diff
            if spectra is None:
                return

            peak_integral = np.full((n_peaks, n_powers), np.nan)

            for j in range(n_peaks):
                center_idx = int(round(float(nw.start_conditions[0, j])))
                fitwindow  = int(round(float(nw.start_conditions[1, j])))
                fitwindow  = max(2 * (fitwindow // 2), 2)   # keep even, min 2
                ref_idx    = int(round(float(nw.start_conditions[2, j])))
                ref_idx    = max(0, min(ref_idx, n_powers - 1))

                peakindex = np.zeros(n_powers, dtype=int)
                peakindex[ref_idx] = center_idx
                if ref_idx + 1 < n_powers:
                    peakindex[ref_idx + 1] = center_idx

                backward = list(range(ref_idx, -1, -1))
                forward  = list(range(ref_idx + 1, n_powers))

                for i in backward + forward:
                    i1 = max(0, peakindex[i] - fitwindow // 2)
                    i2 = min(n_wl, peakindex[i] + fitwindow // 2)
                    if i2 <= i1:
                        continue

                    wl_seg = wl[i1:i2]
                    y_seg  = spectra[i1:i2, i].astype(float)

                    if method == 'sum' or len(wl_seg) < 2:
                        val = float(np.sum(y_seg))
                    elif method == 'trapz':
                        val = float(np.trapz(y_seg, wl_seg))
                    else:
                        val = float(np.sum(y_seg * np.gradient(wl_seg)))
                    peak_integral[j, i] = np.nan if val == 0 else val

                    next_center = i1 + int(np.argmax(y_seg))
                    if 0 < i <= ref_idx:
                        peakindex[i - 1] = next_center
                    elif i > ref_idx and i + 1 < n_powers:
                        peakindex[i + 1] = next_center

                i1_r = max(0, center_idx - fitwindow // 2)
                i2_r = min(n_wl - 1, center_idx + fitwindow // 2)
                nw.specsum.append(dict(
                    values=peak_integral[j, :].copy(),
                    center=float(wl[center_idx]) if 0 <= center_idx < n_wl else 0.0,
                    width=float(abs(wl[i2_r] - wl[i1_r])) if i2_r > i1_r else 0.0,
                    spectrumtype=spectype,
                ))

            nw.peak_integral = peak_integral
            n_valid = int(np.sum(~np.isnan(peak_integral)))
            msg = (f"Analysis: fit + integration done "
                   f"({n_peaks} peak(s), {n_valid}/{n_peaks * n_powers} valid).")

        else:
            msg = "Analysis: integration skipped (no start conditions)."

        self._ana_btn_thr.setEnabled(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg)
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
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Not enough points for fit.")
            return

        if len(p_sel) >= 3:
            coeffs, pcov_fit = np.polyfit(p_sel, v_sel, 1, cov=True)
            a, b = float(coeffs[0]), float(coeffs[1])
            sigma_a = float(np.sqrt(pcov_fit[0, 0]))
            sigma_b = float(np.sqrt(pcov_fit[1, 1]))
            if a != 0:
                # Error propagation: T = -b/a, dT/da = b/a², dT/db = -1/a
                thr_err   = 2.0 * float(np.sqrt((sigma_b / a)**2 + (sigma_a * b / a**2)**2))
            else:
                thr_err   = np.nan
            slope_err = 2.0 * sigma_a
        else:
            coeffs    = np.polyfit(p_sel, v_sel, 1)
            a, b      = float(coeffs[0]), float(coeffs[1])
            thr_err   = np.nan
            slope_err = np.nan

        thresh = -b / a if a != 0 else np.nan
        self._ana_thr_coeffs = coeffs
        self._ana_thr_results[self._ana_thr_title] = {
            "threshold":     thresh,
            "threshold_err": thr_err,
            "slope":         a,
            "slope_err":     slope_err,
            "intercept":     b,
            "sel_power":     p_sel.tolist(),
            "sel_values":    v_sel.tolist(),
        }

        # Redraw with fit line
        ax = self._ana_canvas.ax
        x_fit = np.linspace(0, max(p_all) * 1.1, 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r-", lw=1.5, label="fit")
        if not np.isnan(thresh):
            err_str = f" ± {thr_err:.2g}" if not np.isnan(thr_err) else ""
            ax.axvline(thresh, color="black", lw=1.2, linestyle="--",
                       label=f"Threshold = {thresh:.4g}{err_str} mW")
        ax.legend(fontsize=8)
        ax.set_title(
            f"{self._ana_thr_title}\n"
            f"Threshold = {thresh:.4g} mW  |  slope = {a:.3g}"
        )
        self._ana_canvas.fig.tight_layout()
        self._ana_canvas.draw_idle()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
        nw.thr_results = results   # persist full data for Save
        parts = []
        for title, data in results.items():
            thr   = data["threshold"]
            slope = data["slope"]
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
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Analysis: thresholds done.")
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

    # ── Save results ─────────────────────────────────────────────

    def _ana_save(self):
        nw = self._ana_nw
        if nw is None or self._ana_file is None:
            return

        try:
            with h5py.File(self._ana_file, "a") as f:
                # Remove stale group, recreate fresh
                if "analysis" in f:
                    del f["analysis"]
                grp = f.create_group("analysis")

                # ── Metadata attributes ───────────────────────────
                fit_fn = getattr(nw, "fit_function", "gaussian") or "gaussian"
                grp.attrs["FitFunction"] = fit_fn
                bg_type = getattr(nw, "background_type", "linear") or "linear"
                grp.attrs["FitBackground"] = bg_type

                # ── StartConditions ───────────────────────────────
                if nw.start_conditions is not None:
                    sc = grp.create_dataset(
                        "StartConditions", data=np.array(nw.start_conditions, dtype=float)
                    )
                    sc.attrs["description"] = (
                        "3 x n_peaks: [peak_pixel_index, fitwindow_pixels, ref_power_step_idx]"
                    )

                # ── FitWindows ────────────────────────────────────
                fw = getattr(nw, "fit_windows", None)
                if fw is not None:
                    fw_arr = np.array(fw, dtype=float)
                    ds = grp.create_dataset("FitWindows", data=fw_arr)
                    ds.attrs["units"] = "pixels"
                    ds.attrs["description"] = "Half-window in pixels for each peak"

                # ── PeakArea / PeakAreaErr ────────────────────────
                def _cell_to_mat_max(cell, n_pk, n_pw):
                    arr = np.full((n_pk, n_pw), np.nan)
                    for j, row in enumerate(cell):
                        for i, val in enumerate(row):
                            v = np.atleast_1d(val)
                            if v.size > 0 and not np.all(np.isnan(v)):
                                arr[j, i] = float(np.nanmax(v))
                    return arr

                def _cell_to_mat_first(cell, n_pk, n_pw):
                    arr = np.full((n_pk, n_pw), np.nan)
                    for j, row in enumerate(cell):
                        for i, val in enumerate(row):
                            v = np.atleast_1d(val)
                            if v.size > 0 and not np.all(np.isnan(v)):
                                arr[j, i] = float(v[0])
                    return arr

                pa = getattr(nw, "peak_area", None)
                if pa is not None:
                    try:
                        n_pk = len(pa); n_pw = len(pa[0]) if n_pk > 0 else 0
                        grp.create_dataset("PeakArea", data=_cell_to_mat_max(pa, n_pk, n_pw))
                    except Exception:
                        pass

                pae = getattr(nw, "peak_area_err", None)
                if pae is not None:
                    try:
                        n_pk = len(pae); n_pw = len(pae[0]) if n_pk > 0 else 0
                        grp.create_dataset("PeakAreaErr", data=_cell_to_mat_max(pae, n_pk, n_pw))
                    except Exception:
                        pass

                # ── PeakPos / PeakPosErr ──────────────────────────
                pp = getattr(nw, "peak_pos", None)
                if pp is not None:
                    try:
                        n_pk = len(pp); n_pw = len(pp[0]) if n_pk > 0 else 0
                        ds = grp.create_dataset("PeakPos",
                                                data=_cell_to_mat_first(pp, n_pk, n_pw))
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                ppe = getattr(nw, "peak_pos_err", None)
                if ppe is not None:
                    try:
                        n_pk = len(ppe); n_pw = len(ppe[0]) if n_pk > 0 else 0
                        ds = grp.create_dataset("PeakPosErr",
                                                data=_cell_to_mat_first(ppe, n_pk, n_pw))
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                # ── FWHM / FWHMErr ────────────────────────────────
                fwhm = getattr(nw, "fwhm", None) or getattr(nw, "findpeaks_fwhm", None)
                if fwhm is not None:
                    try:
                        fwhm_arr = np.array(fwhm, dtype=float)
                        ds = grp.create_dataset("FWHM", data=fwhm_arr)
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                fwhm_e = getattr(nw, "fwhm_err", None)
                if fwhm_e is not None:
                    try:
                        fwhm_e_arr = np.array(fwhm_e, dtype=float)
                        ds = grp.create_dataset("FWHMErr", data=fwhm_e_arr)
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                # ── FitParameters ─────────────────────────────────
                fits = getattr(nw, "fits", None)
                if fits is not None:
                    fp_grp = grp.create_group("FitParameters")
                    try:
                        for j, peak_fits in enumerate(fits):
                            pk_grp = fp_grp.create_group(f"peak_{j}")
                            for i, fit_ns in enumerate(peak_fits):
                                if fit_ns is not None:
                                    popt = getattr(fit_ns, "popt", fit_ns)
                                    if popt is not None:
                                        pk_grp.create_dataset(
                                            f"power_{i}",
                                            data=np.atleast_1d(popt).astype(float)
                                        )
                    except Exception:
                        pass

                # ── BackgroundData ────────────────────────────────
                fit_data = getattr(nw, "fit_data", None)
                if fit_data is not None:
                    bg_grp = grp.create_group("BackgroundData")
                    try:
                        for j, peak_data in enumerate(fit_data):
                            pk_grp = bg_grp.create_group(f"peak_{j}")
                            for i, data_arr in enumerate(peak_data):
                                if data_arr is not None:
                                    arr = np.atleast_2d(data_arr)
                                    if arr.shape[1] >= 3:
                                        pk_grp.create_dataset(f"power_{i}", data=arr[:, 2])
                    except Exception:
                        pass

                # ── PeakIntegral ──────────────────────────────────
                pi = getattr(nw, "peak_integral", None)
                if pi is not None:
                    grp.create_dataset("PeakIntegral", data=np.array(pi, dtype=float))

                # ── Specsum ───────────────────────────────────────
                specsum = getattr(nw, "specsum", None)
                if specsum:
                    ss_grp = grp.create_group("Specsum")
                    for k, entry in enumerate(specsum):
                        e_grp = ss_grp.create_group(str(k))
                        vals  = np.array(entry.get("values", []), dtype=float)
                        e_grp.create_dataset("values", data=vals)
                        e_grp.attrs["center"]      = float(entry.get("center", np.nan))
                        e_grp.attrs["width"]       = float(entry.get("width", np.nan))
                        e_grp.attrs["spectrumtype"] = str(entry.get("spectype", ""))

                # ── Thresholds ────────────────────────────────────
                thr_results = getattr(nw, "thr_results", None)
                if thr_results:
                    thr_grp  = grp.create_group("Thresholds")
                    spot_r   = getattr(nw, "spot_radius_short", None)
                    rep_rate = getattr(nw, "rep_rate", None)
                    for title, data in thr_results.items():
                        tg      = thr_grp.create_group(title)
                        thr_mW  = data.get("threshold", np.nan)
                        slope   = data.get("slope",     np.nan)
                        intcpt  = data.get("intercept", np.nan)
                        p_sel   = np.array(data.get("sel_power",  []), dtype=float)
                        v_sel   = np.array(data.get("sel_values", []), dtype=float)

                        thr_err_mW  = data.get("threshold_err", np.nan)
                        slope_err   = data.get("slope_err",     np.nan)

                        if (spot_r is not None and rep_rate is not None
                                and not np.isnan(thr_mW)
                                and float(rep_rate) > 0 and float(spot_r) > 0):
                            import math
                            d_cm = 2.0 * float(spot_r) * 1e-4
                            P_W  = float(thr_mW) * 1e-3
                            f_Hz = float(rep_rate)
                            conv = 4.0 / f_Hz / math.pi / d_cm**2 * 1e6  # mW→µJ/cm²
                            thr_fluence = conv * P_W
                            tg.attrs["Threshold"]      = thr_fluence
                            tg.attrs["ThresholdUnits"] = "uJ/cm^2"
                            if not np.isnan(thr_err_mW):
                                tg.attrs["ThresholdErr"] = conv * float(thr_err_mW) * 1e-3
                            else:
                                tg.attrs["ThresholdErr"] = np.nan
                        else:
                            tg.attrs["Threshold"]      = thr_mW
                            tg.attrs["ThresholdUnits"] = "mW"
                            tg.attrs["ThresholdErr"]   = thr_err_mW

                        tg.attrs["slope"]     = slope
                        tg.attrs["slope_err"] = slope_err if not np.isnan(slope_err) else np.nan
                        tg.attrs["intercept"] = intcpt
                        if p_sel.size:
                            tg.create_dataset("FitIntervalPower",  data=p_sel)
                        if v_sel.size:
                            tg.create_dataset("FitIntervalValues", data=v_sel)

        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        QMessageBox.information(
            self, "Saved",
            f"Analysis results written to\n{os.path.basename(self._ana_file)}"
        )
