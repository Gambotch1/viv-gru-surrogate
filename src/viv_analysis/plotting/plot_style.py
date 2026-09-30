"""Shared figure style for all thesis figures: colours, line styles and fonts.

Figures are sized for a 16.5 cm text width (TEXT_WIDTH_IN), so they can be
included at full width without rescaling the fonts.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

CFD_COLOR       = "#222222"
MODEL_COLOR     = "#0072B2"
ERROR_COLOR     = "#D55E00"
SECONDARY_COLOR = "#009E73"
ORANGE_COLOR    = "#E69F00"
GRID_COLOR      = "#D9D9D9"

TEXT_WIDTH_CM = 16.5
TEXT_WIDTH_IN = TEXT_WIDTH_CM / 2.54


CFD_STYLE = {
    "color": CFD_COLOR,
    "linestyle": "-",
    "linewidth": 1.15,
    "label": "CFD reference",
    "zorder": 2,
}

MODEL_STYLE = {
    "color": MODEL_COLOR,
    "linestyle": (0, (4, 2)),
    "linewidth": 1.05,
    "label": "GRU prediction",
    "zorder": 3,
}

ERROR_STYLE = {
    "color": ERROR_COLOR,
    "linestyle": "-",
    "linewidth": 0.75,
    "zorder": 2,
}


def apply_thesis_style() -> None:
    """Set the matplotlib defaults used by every thesis figure."""
    plt.style.use("default")
    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.usetex": False,

        "figure.figsize": (TEXT_WIDTH_IN, 3.8),
        "figure.dpi": 120,
        "savefig.dpi": 600,

        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        "lines.linewidth": 1.15,
        "axes.linewidth": 0.75,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,

        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,

        "legend.frameon": False,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })
