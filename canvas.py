from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Compact QPushButton style for tightly-packed button rows (e.g. 3-across
# toolbar rows in a narrow sidebar). Styles such as qt_material apply large
# padding/min-width to QPushButton that clips the label text in these rows;
# this override keeps the label readable without affecting other buttons.
_COMPACT_BTN_STYLE = (
    "QPushButton { padding: 2px 4px; min-width: 0px; min-height: 20px; }"
)


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
