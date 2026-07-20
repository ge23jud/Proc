"""Shared PyQtGraph theming and color constants, matching the app's global
qt_material dark_teal.xml stylesheet (applied in stitchcraft.py)."""
import pyqtgraph as pg

# Matplotlib's tab10 qualitative palette (identical RGB values), used
# wherever the old code colored lines per-file via cm.get_cmap("tab10").
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Matplotlib's plasma sequential colormap, used wherever the old code colored
# lines/colorbars by a continuous value (e.g. power) via cm.get_cmap("plasma").
PLASMA = pg.colormap.get("plasma", source="matplotlib")

CANVAS_BG = "#1e1e1e"
AXIS_FG = "#cccccc"


def apply_pg_theme():
    """Call once at startup, right after qt_material's apply_stylesheet(...)
    and before any PGCanvas is constructed, so every canvas inherits
    consistent global options."""
    pg.setConfigOptions(antialias=True, background=CANVAS_BG, foreground=AXIS_FG)
