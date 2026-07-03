from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


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
