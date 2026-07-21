# Proc App — Claude Context

## What this project is
"PL Spectrum Stitcher / Converter" — a Windows PyQt5 desktop app for processing photoluminescence (PL) spectroscopy data during Benjamin Haubmann's PhD at TUM. It stitches multi-file power-series spectra into a single spectrum, converts between file formats (.origin / HDF5), visualizes spectra, and runs peak-fitting / threshold-detection analysis.

## Key facts
- **Entry point**: `stitchcraft.py` — builds the `QApplication`, applies the `qt_material` `dark_teal.xml` stylesheet, calls `plotting.apply_pg_theme()`, then shows `StitchApp`.
- **Main window**: `app.py` — `StitchApp(QMainWindow)`, a `QTabWidget` hosting the 4 tabs below.
- **Platform**: Windows-only PyQt5 app (no PySide/Qt6 constraint otherwise).
- **GitHub**: https://github.com/ge23jud/Proc.git
- **Dependencies**: pinned in `requirements.txt` (PyQt5, pyqtgraph, numpy, h5py, scipy, pandas, qt-material, opencv-python, Pillow). No `pyproject.toml`/`setup.py`.

## Tabs (`tabs/`)
- **`stitch.py`** — "Stitch / Convert": load `.origin` power-series files, preview spectra, pick per-pair overlap "transition spans" (draggable region), dark-spectrum subtraction, power calibration, save as HDF5 or `.origin`.
- **`visualizer.py`** — "Visualizer": load HDF5 files, overlay PL spectra across files/power steps with configurable axes (log/linear, manual range), colored by file or by power.
- **`analysis.py`** — "Analysis": peak-picking → fit-window selection → `fit_nw`/`integrate_spectra` (via sibling `nw_analysis.py`) → threshold detection (drag-select ASE region, linear fit, threshold ± error) → L-L curve plot → save results into the HDF5's `analysis` group. A click-driven wizard/state machine (`self._ana_mode`).
- **`spotsize.py`** — "Spotsize": load an Excel/text beam-scan file, select two spans on the derivative curve, Gaussian-fit the edge, compute the 95.4% width and beam diameter (Δx × sin(angle)).
- **`sem.py`** — "SEM": nanowire width/height/angle measurement from SEM TIFF images, a Python port of the "smarter SEM" MATLAB toolset (`\\nas.ads.mwn.de\tuze\wsi\e24\SQN\Researchers\Haubmann Benjamin\01_PhD\10_Scripts\smarter SEM`). Three modes: **Automated** (whole-image binarize → contour-trace → `cv2.minAreaRect` fit, with edge/size/tilt rejection rules and a per-image "skip IDs" list, port of `edgeDetect.m`), **Semi-automated** (click near a wire → auto-crop a physical-size window around it → detect the tallest accepted blob in the crop, falling back to a manual 2-click measurement if detection fails or the result is too short, port of `edgeDetect_semiauto.m`), and **Manual** (plain 2-click distance measurement tagged with a measurement-type label, port of `click_on_image.m`/`smarterSEM.m`). All three modes share one results table (exportable to CSV) and a live mean±std stats readout (replacing the separate `getValues.m`/`combined.m` post-processing scripts). Core image-processing logic lives in `sem_utils.py` (root, no PyQt imports); pixel scale is parsed by regex-scanning the TIFF's embedded ASCII metadata footer for `Pixel Size = <value> <unit>` (editable per-image if not found or wrong). Adds `opencv-python` and `Pillow` as dependencies.

## Shared modules
- **`io_utils.py`** — pure data I/O: `.origin`/HDF5 read-write, power-calibration parsing. No UI, no plotting.
- **`sem_utils.py`** — pure image-processing logic for the SEM tab: TIFF+scale reading, blob detection/rectangle-fit, crop, and 2-point measurement. No UI.
- **Sibling directories** (imported via `sys.path.insert`, not part of this repo):
  - `..\PL Helper\pl.py` — numeric helpers: origin-file parsing, spectrum stitching (`_stitch_counts`), Gaussian fitting, `_HC_EV_NM` (eV↔nm constant).
  - `..\Plot\nw_analysis.py` — `nw_analysis` module (`import nw_analysis as nwa`), used by `analysis.py` for `fit_nw`/`integrate_spectra` on a `SimpleNamespace` "nanowire" object (`_new_nanowire()`). Always called with `show_progress=False` from this app, so its own standalone matplotlib windows (`plt.subplots`/`ginput`) are never triggered — that module's plotting is out of scope for this app.

## Plotting architecture — `plotting/` package
All 4 tabs plot exclusively through PyQtGraph via the shared `plotting/` package (migrated off Matplotlib in 2026-07; `matplotlib` is no longer a dependency). Each tab imports only from `plotting` (never a submodule directly):

- **`canvas.py`** — `PGCanvas(pg.GraphicsLayoutWidget)`: the shared embeddable canvas. `reset_axes()` clears curves/ROIs/legend/colorbar and returns a fresh `plot_item` (full clear-and-rebuild on every redraw, matching every tab's existing redraw pattern — no incremental updates). `sigDataClicked(x, y, button)` replaces matplotlib's `button_press_event` for click-driven picking (used by `analysis.py`). `draw_idle()` is a no-op kept only so old call sites didn't need touching.
- **`theme.py`** — `apply_pg_theme()` (call once at startup, before any `PGCanvas`), `TAB10` palette, `PLASMA` colormap — matched to the app's dark `qt_material` theme.
- **`multiline.py`** — `MultiLinePlotter` + `LineColorScheme` strategy classes (`CategoricalScheme` = tab10-by-index, `SequentialScheme` = plasma-by-value, `LogAlphaRamp` = log-scaled opacity wrapper) — covers every "color N lines by file or by power" pattern across `stitch.py`/`visualizer.py`/`analysis.py`.
- **`colorbar.py`** — `GradientLegend(pg.ColorBarItem)`: a standalone plasma gradient legend (not attached to an image), used wherever lines are colored by power.
- **`regions.py`** — `DraggableSpan` (wraps `pg.LinearRegionItem`; one class serves both the interactive picker role and the "replay a stored span" role via `set_range()`) and `DraggableRect` (wraps `pg.RectROI`; a persistent resizable box — deliberately different UX from matplotlib's click-drag `RectangleSelector`, confirmed acceptable). Both track their own add/remove lifecycle since `reset_axes()` wipes them on every redraw.
- **`scatter.py`** — `RecolorableScatter`: thin wrapper over `pg.ScatterPlotItem` for per-point recoloring (analysis.py's threshold-region highlighting).
- **`annotations.py`** — `FilledRegion` (`pg.FillBetweenItem` wrapper) and `MeasurementArrow` (hand-built double-headed arrow + label composite — no built-in PyQtGraph equivalent; spotsize.py's only consumer).
- **`toolbar.py`** — `make_pg_toolbar(canvas)`: a minimal 2-button toolbar (Export Image…, Reset View) replacing matplotlib's `NavigationToolbar2QT`. Pan/zoom is handled by PyQtGraph's built-in mouse interactions (scroll-zoom, drag-pan, right-drag box-zoom) instead of dedicated buttons.

**Known pyqtgraph gotcha (worked around in `canvas.py`)**: toggling `setLogMode` right after `reset_axes()` can hit a `10**range` overflow if the ViewBox still holds a stale large linear range from whatever was plotted before the clear. `reset_axes()` explicitly resets the view to a small known range before re-enabling auto-range to avoid this.

**Known pyqtgraph limitation**: `pg.LegendItem` only renders items with an `.opts` attribute (`PlotDataItem`-like) — `LinearRegionItem`/`InfiniteLine` cannot be added via `legend.addItem(...)`. Span regions are identified by on-canvas color + sidebar labels instead of a legend entry; axvline-style markers use `InfiniteLine`'s own `label=`/`labelOpts=` instead.
