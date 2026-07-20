import pyqtgraph as pg


class RecolorableScatter:
    """Thin wrapper around pg.ScatterPlotItem for API symmetry with the rest
    of this package. PyQtGraph's setData(brush=[...]) already supports a
    per-point brush list natively, so set_point_colors() replaces
    ax.scatter(...).set_facecolor(colors) with no real gap to bridge."""

    def __init__(self, plot_item, default_color="steelblue", size=8):
        self._plot_item = plot_item
        self._default_color = default_color
        self._scatter = pg.ScatterPlotItem(size=size, brush=pg.mkBrush(default_color))
        self._x = []
        self._y = []
        plot_item.addItem(self._scatter)

    def set_data(self, x, y):
        self._x, self._y = list(x), list(y)
        self._scatter.setData(x=self._x, y=self._y,
                               brush=[pg.mkBrush(self._default_color)] * len(self._x))

    def set_point_colors(self, colors):
        """colors: list[str/QColor], len == number of points currently set."""
        self._scatter.setData(x=self._x, y=self._y,
                               brush=[pg.mkBrush(c) for c in colors])

    def remove(self):
        try:
            self._plot_item.removeItem(self._scatter)
        except Exception:
            pass
