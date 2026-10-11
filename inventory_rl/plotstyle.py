"""Shared figure style: one color per algorithm everywhere, neutral grays for the baselines."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"PPO": "#2a78d6", "A2C": "#eb6834", "DQN": "#1baf7a"}
MARKERS = {"PPO": "o", "A2C": "s", "DQN": "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"L2_p4": "L = 2, p = 4", "L2_p9": "L = 2, p = 9", "L4_p4": "L = 4, p = 4", "L4_p9": "L = 4, p = 9"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
