# Compact QPushButton style for tightly-packed button rows (e.g. 3-across
# toolbar rows in a narrow sidebar). Styles such as qt_material apply large
# padding/min-width to QPushButton that clips the label text in these rows;
# this override keeps the label readable without affecting other buttons.
# Relocated here (unrelated to plotting) from the old canvas.py, which this
# package replaces.
_COMPACT_BTN_STYLE = (
    "QPushButton { padding: 2px 4px; min-width: 0px; min-height: 20px; }"
)

from .theme import apply_pg_theme, TAB10, PLASMA
from .canvas import PGCanvas
from .multiline import (
    LineColorScheme, CategoricalScheme, SequentialScheme, LogAlphaRamp, MultiLinePlotter,
)
from .colorbar import GradientLegend
from .regions import DraggableSpan, DraggableRect
from .scatter import RecolorableScatter
from .annotations import FilledRegion, MeasurementArrow
from .toolbar import make_pg_toolbar

__all__ = [
    "_COMPACT_BTN_STYLE",
    "apply_pg_theme", "TAB10", "PLASMA",
    "PGCanvas",
    "LineColorScheme", "CategoricalScheme", "SequentialScheme", "LogAlphaRamp", "MultiLinePlotter",
    "GradientLegend",
    "DraggableSpan", "DraggableRect",
    "RecolorableScatter",
    "FilledRegion", "MeasurementArrow",
    "make_pg_toolbar",
]
