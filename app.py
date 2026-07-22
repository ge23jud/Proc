import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import Qt

from tabs.stitch     import StitchTab
from tabs.visualizer import VisualizerTab
from tabs.analysis   import AnalysisTab
from tabs.spotsize   import SpotsizeTab
from tabs.sem        import SemTab
from tabs.plot       import PlotTab


class StitchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PL Spectrum Stitcher / Converter")
        self.resize(1160, 680)
        self._build_ui()

    def _build_ui(self):
        tabs = QTabWidget()
        self._stitch_tab    = StitchTab(self)
        self._vis_tab       = VisualizerTab(self)
        self._ana_tab       = AnalysisTab(self)
        self._spotsize_tab  = SpotsizeTab(self)
        self._sem_tab       = SemTab(self)
        self._plot_tab      = PlotTab(self)
        tabs.addTab(self._stitch_tab,   "Stitch / Convert")
        tabs.addTab(self._vis_tab,      "Visualizer")
        tabs.addTab(self._ana_tab,      "Analysis")
        tabs.addTab(self._spotsize_tab, "Spotsize")
        tabs.addTab(self._sem_tab,      "SEM")
        tabs.addTab(self._plot_tab,     "Plot")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready — add .origin files to begin.")
