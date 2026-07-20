from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFileDialog
import pyqtgraph.exporters as pg_exporters


def make_pg_toolbar(canvas, parent=None):
    """Small 2-button toolbar replacing matplotlib's NavigationToolbar2QT:
    Export Image (PNG/SVG) and Reset View. PyQtGraph's built-in mouse
    interactions (scroll-zoom, drag-pan, right-drag box-zoom) cover pan/zoom
    without needing dedicated toolbar buttons."""
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    btn_export = QPushButton("Export Image…")
    btn_reset = QPushButton("Reset View")
    layout.addWidget(btn_export)
    layout.addWidget(btn_reset)
    layout.addStretch(1)

    def _export():
        path, _selected_filter = QFileDialog.getSaveFileName(
            widget, "Export plot image", "",
            "PNG (*.png);;SVG (*.svg)"
        )
        if not path:
            return
        if path.lower().endswith(".svg"):
            exporter = pg_exporters.SVGExporter(canvas.plot_item)
        else:
            if not path.lower().endswith(".png"):
                path += ".png"
            exporter = pg_exporters.ImageExporter(canvas.plot_item)
        exporter.export(path)

    def _reset():
        canvas.plot_item.getViewBox().autoRange()

    btn_export.clicked.connect(_export)
    btn_reset.clicked.connect(_reset)
    return widget
