import numpy as np
from PyQt5.QtGui import QFont
import pyqtgraph as pg


class FilledRegion:
    """Near-1:1 port of ax.fill_between(x, y, alpha=a): fills the area
    between a curve segment and its own zero (or given) baseline, via
    pg.FillBetweenItem."""

    def __init__(self, plot_item, x, y, brush=(0, 150, 0, 80), baseline=0.0):
        self._plot_item = plot_item
        y_arr = np.asarray(y, dtype=float)
        self._curve = pg.PlotDataItem(x, y_arr, pen=None)
        self._base = pg.PlotDataItem(x, np.full_like(y_arr, baseline), pen=None)
        self._fill = pg.FillBetweenItem(self._curve, self._base, brush=pg.mkBrush(*brush))
        plot_item.addItem(self._curve)
        plot_item.addItem(self._base)
        plot_item.addItem(self._fill)

    def remove(self):
        for item in (self._fill, self._curve, self._base):
            try:
                self._plot_item.removeItem(item)
            except Exception:
                pass


class MeasurementArrow:
    """Double-headed arrow + centered label, measuring a horizontal distance
    between two x-positions at a fixed y. PyQtGraph has no built-in
    equivalent of matplotlib's ax.annotate(arrowprops=dict(arrowstyle="<->"));
    this is a small, self-contained composite (shaft + 2 arrowheads + label),
    not a generalized annotation framework — spotsize.py is its only consumer."""

    def __init__(self, plot_item, x0, x1, y, label=None, color="r"):
        self._plot_item = plot_item
        # ArrowItem angle=0 points left by default; angle=180 points right —
        # so the left-end head (pointing further left, outward) uses angle=0
        # and the right-end head (pointing further right, outward) uses angle=180.
        self._shaft = pg.PlotDataItem([x0, x1], [y, y], pen=pg.mkPen(color, width=2))
        self._head_left = pg.ArrowItem(angle=0, pen=pg.mkPen(color), brush=color,
                                        tipAngle=30, headLen=12)
        self._head_right = pg.ArrowItem(angle=180, pen=pg.mkPen(color), brush=color,
                                         tipAngle=30, headLen=12)
        self._head_left.setPos(x0, y)
        self._head_right.setPos(x1, y)
        self._label = pg.TextItem(label or "", color=color, anchor=(0.5, 1.0))
        font = QFont()
        font.setBold(True)
        self._label.setFont(font)
        self._label.setPos((x0 + x1) / 2.0, y)
        for item in (self._shaft, self._head_left, self._head_right, self._label):
            plot_item.addItem(item)

    def set_span(self, x0, x1, y=None, label=None):
        if y is None:
            y = self._shaft.yData[0]
        self._shaft.setData([x0, x1], [y, y])
        self._head_left.setPos(x0, y)
        self._head_right.setPos(x1, y)
        self._label.setPos((x0 + x1) / 2.0, y)
        if label is not None:
            self._label.setText(label)

    def remove(self):
        for item in (self._shaft, self._head_left, self._head_right, self._label):
            try:
                self._plot_item.removeItem(item)
            except Exception:
                pass
