import os
import re
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox, QLineEdit, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.widgets import SpanSelector
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from canvas import _MplCanvas

_PL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "PL Helper")
)
if _PL_DIR not in sys.path:
    sys.path.insert(0, _PL_DIR)

from pl import _HC_EV_NM, _gaussian, _find_symmetric_95_bounds


class SpotsizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Spotsize state ────────────────────────────────────────
        self._ss_x            = None
        self._ss_deriv        = None
        self._ss_spans        = [None, None]
        self._ss_span_patches = [None, None]
        self._ss_selector     = None

        # ── Build UI (body of original _build_spotsize_tab) ──────
        layout = QHBoxLayout(self)
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
