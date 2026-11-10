"""Shared style for the benchmark figures (paper, light background, print).

Every solver keeps the same colour, marker and line style in every figure.
Colours follow a fixed categorical order validated for colour-vision
deficiency on adjacent pairs; markers give a second, colour-free encoding so
the curves stay distinguishable in grayscale print.
"""

import matplotlib

# Solver order = legend order = colour slot order.  Never reorder per figure.
SOLVERS = ["GaasQP", "qpOASES", "qpOASES-B", "OSQP", "DAQP", "PIQP", "Clarabel"]

COLOR = {
    "GaasQP":    "#2a78d6",  # blue
    "qpOASES":   "#eb6834",  # orange
    "qpOASES-B": "#1baf7a",  # aqua
    "OSQP":      "#eda100",  # yellow
    "DAQP":      "#e87ba4",  # magenta
    "PIQP":      "#008300",  # green
    "Clarabel":  "#6250d6",  # violet
}
MARKER = {
    "GaasQP": "o", "qpOASES": "s", "qpOASES-B": "D", "OSQP": "^",
    "DAQP": "v", "PIQP": "P", "Clarabel": "X",
}
LINESTYLE = {
    "GaasQP": "-", "qpOASES": "--", "qpOASES-B": "--", "OSQP": ":",
    "DAQP": "-.", "PIQP": (0, (5, 1, 1, 1)), "Clarabel": (0, (1, 1)),
}
LABEL = {
    "qpOASES": "qpOASES",
    "qpOASES-B": "qpOASES (QProblemB)",
}

# Solvers whose iteration counts are active-set pivots (comparable to GaasQP's).
ACTIVE_SET = ["GaasQP", "qpOASES", "qpOASES-B", "DAQP"]

W2 = 7.16  # IEEE two-column figure width (in)

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def line(ax, solver, x, y, **kw):
    """Plot one solver's series with its fixed style."""
    style = dict(color=COLOR[solver], marker=MARKER[solver], linestyle=LINESTYLE[solver],
                 label=LABEL.get(solver, solver), markeredgecolor="white",
                 markeredgewidth=0.6, zorder=4 if solver == "GaasQP" else 3)
    if solver == "GaasQP":
        style["linewidth"] = 2.0
    style.update(kw)
    return ax.plot(x, y, **style)


def shared_legend(fig, axes, ncol=4):
    """One legend above all panels, in the fixed solver order."""
    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(l, h)
    # Fixed solver order; a custom label (e.g. a merged series) sorts by the
    # solver name it starts with.
    def rank(lbl):
        for i, s in enumerate(SOLVERS):
            if lbl == LABEL.get(s, s):
                return (i, 0)
        for i, s in enumerate(SOLVERS):
            if lbl.startswith(s):
                return (i, 1)
        return (len(SOLVERS), 0)
    order = sorted(labels, key=rank)
    fig.legend([labels[l] for l in order], order, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=ncol, handlelength=2.6,
               columnspacing=1.4)
