from PyQt5.QtCore import QObject, pyqtSignal
import pyqtgraph as pg


class DraggableSpan(QObject):
    """SpanSelector replacement wrapping pg.LinearRegionItem(orientation='vertical').

    One instance serves both roles the old code needed two separate
    matplotlib APIs for: the live interactive picker (movable=True, like
    matplotlib.widgets.SpanSelector) and the static replay of a
    previously-picked span (like matplotlib's ax.axvspan) — for the latter,
    just call set_range() with movable=False.

    sigRegionSelected(xmin, xmax) is emitted on drag-release, mirroring
    SpanSelector's onselect callback signature.
    """

    sigRegionSelected = pyqtSignal(float, float)

    def __init__(self, plot_item, color=(0, 150, 0, 60), movable=True):
        super().__init__()
        self._plot_item = plot_item
        self._region = pg.LinearRegionItem(orientation="vertical",
                                            brush=pg.mkBrush(*color),
                                            movable=movable)
        self._region.setZValue(10)
        self._connected = False
        self._added = False

    def activate(self, initial_range=None, bounds=None):
        """Add the region to the plot and (if movable) start emitting
        sigRegionSelected on drag-release. Replaces re-installing a fresh
        SpanSelector on every redraw."""
        if bounds is not None:
            self._region.setBounds(bounds)
        if initial_range is not None:
            self._region.setRegion(initial_range)
        if not self._added:
            self._plot_item.addItem(self._region)
            self._added = True
        if self._region.movable and not self._connected:
            self._region.sigRegionChangeFinished.connect(self._on_changed)
            self._connected = True

    def deactivate(self):
        """Remove the region from the plot and stop emitting. Replaces
        SpanSelector's set_active(False) + discard."""
        if self._connected:
            try:
                self._region.sigRegionChangeFinished.disconnect(self._on_changed)
            except (TypeError, RuntimeError):
                pass
            self._connected = False
        if self._added:
            try:
                self._plot_item.removeItem(self._region)
            except Exception:
                pass
            self._added = False

    def set_range(self, xmin, xmax):
        """Show (or move) a span without triggering sigRegionSelected —
        replaces the static ax.axvspan() 'redraw a previously-picked span' use."""
        was_connected = self._connected
        if was_connected:
            self._region.sigRegionChangeFinished.disconnect(self._on_changed)
            self._connected = False
        self._region.setRegion((xmin, xmax))
        if not self._added:
            self._plot_item.addItem(self._region)
            self._added = True
        if was_connected:
            self._region.sigRegionChangeFinished.connect(self._on_changed)
            self._connected = True

    def get_range(self):
        return tuple(self._region.getRegion())

    def set_movable(self, movable):
        self._region.setMovable(movable)

    @property
    def region_item(self):
        """Underlying pg.LinearRegionItem, exposed for cases that need to
        register it with a legend (legend.addItem(span.region_item, name))."""
        return self._region

    def _on_changed(self):
        xmin, xmax = self._region.getRegion()
        self.sigRegionSelected.emit(xmin, xmax)


class DraggableRect(QObject):
    """RectangleSelector replacement wrapping pg.RectROI: a persistent,
    draggable/resizable box (confirmed acceptable UX change from matplotlib's
    click-drag-then-adjust RectangleSelector — seed with a default rect
    covering the full data range).

    sigRectSelected(x0, y0, x1, y1) is emitted on drag-release, mirroring
    RectangleSelector's (eclick, erelease) callback.
    """

    sigRectSelected = pyqtSignal(float, float, float, float)

    def __init__(self, plot_item, pen=(200, 50, 50)):
        super().__init__()
        self._plot_item = plot_item
        self._roi = pg.RectROI(pos=(0, 0), size=(1, 1), pen=pg.mkPen(pen, width=2))
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self._roi.sigRegionChangeFinished.connect(self._on_changed)
        self._added = False

    def activate(self, initial_rect):
        x0, y0, x1, y1 = initial_rect
        self._roi.setPos((x0, y0))
        self._roi.setSize((x1 - x0, y1 - y0))
        if not self._added:
            self._plot_item.addItem(self._roi)
            self._added = True

    def deactivate(self):
        if self._added:
            try:
                self._plot_item.removeItem(self._roi)
            except Exception:
                pass
            self._added = False

    def get_rect(self):
        x0, y0 = self._roi.pos()
        w, h = self._roi.size()
        return (float(x0), float(y0), float(x0 + w), float(y0 + h))

    def _on_changed(self):
        x0, y0, x1, y1 = self.get_rect()
        self.sigRectSelected.emit(x0, y0, x1, y1)
