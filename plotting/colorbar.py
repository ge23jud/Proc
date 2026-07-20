import pyqtgraph as pg


class GradientLegend(pg.ColorBarItem):
    """Standalone colormap gradient bar with a label — not attached to any
    image/scatter, purely a legend for "this line color encodes this value".
    Replaces cm.ScalarMappable(cmap=...) + fig.colorbar(sm, ax=ax, label=...).

    pg.ColorBarItem is documented primarily as an image-legend widget (paired
    with setImageItem()), but it works standalone (verified) when interactive
    is disabled — no need to hand-roll a gradient bar from scratch.
    """

    def __init__(self, cmap, vmin=0.0, vmax=1.0, label="", width=18):
        super().__init__(values=(vmin, vmax), colorMap=cmap, label=label,
                          interactive=False, width=width)

    def set_range(self, vmin, vmax):
        self.setLevels((vmin, vmax))
