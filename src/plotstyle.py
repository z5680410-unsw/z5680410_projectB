"""Project-wide figure design system — Financial Times visual language.

Every exhibit in results/figures/ uses this single, consistent design system
rather than matplotlib's defaults, adapting the visual conventions the
Financial Times is known for: a warm off-white ("FT paper") background, a
restrained editorial colour palette, clean sans-serif type, left-aligned
titles with a subtitle, horizontal-only gridlines, and no chart-border box.

The same colour always means the same thing across every figure:

    EQUITY_COLOR   - anything equity-related (FT teal/blue)
    CRYPTO_COLOR   - anything crypto-related (FT orange)
    NEUTRAL_COLOR  - baseline / control / "normal" observations (grey)
    ACCENT_COLOR   - flagged observations (outliers, anomalies) - used ONLY
                     for this purpose, so the reader learns to read the
                     accent colour as "flagged" the first time and can carry
                     that reading through the whole report (FT claret/red).

Call apply_style() once before any figure is built (done on import). Two
helpers, ft_title() and ft_source(), reproduce the FT header/footer
convention and should be used in place of ax.set_title / plt.figtext.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Financial Times palette
# ---------------------------------------------------------------------------
# FT "paper" background and ink, plus an editorial categorical palette drawn
# from the FT visual-vocabulary conventions. Chosen to stay distinguishable
# under common colour-vision deficiencies (blue / orange / grey / claret).
FT_PAPER = "#FFF1E5"      # signature FT pink-cream page background
FT_INK = "#33302E"        # near-black warm ink for text
FT_GRID = "#E6D9CC"       # muted gridline that sits on the paper tone

EQUITY_COLOR = "#0F5499"  # FT blue
CRYPTO_COLOR = "#E85D04"  # FT orange
NEUTRAL_COLOR = "#8D8D8D" # mid grey - control / baseline
ACCENT_COLOR = "#990F3D"  # FT claret - flagged / anomalous ONLY

# Categorical palette for breakdowns with >2 groups (e.g. the 10 GICS
# sectors), ordered so adjacent categories stay visually distinct.
SECTOR_PALETTE = [
    "#0F5499", "#E85D04", "#0D7680", "#593380", "#990F3D",
    "#1E874B", "#B8860B", "#66605C", "#4C9C2E", "#A8326E",
]

FONT_FAMILY = "DejaVu Sans"  # ships with matplotlib; renders identically on
                              # Windows/Mac/Linux (a reproducibility concern,
                              # not just aesthetic). FT itself uses a bespoke
                              # face (Financier/Metric); DejaVu Sans is the
                              # closest always-available clean sans-serif.


def apply_style() -> None:
    """Apply the FT-inspired rcParams globally. Safe to call repeatedly."""
    mpl.rcParams.update({
        # typography
        "font.family": FONT_FAMILY,
        "font.size": 11,
        "text.color": FT_INK,
        "axes.labelcolor": FT_INK,
        "xtick.color": FT_INK,
        "ytick.color": FT_INK,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        # paper background (both the figure and the plot area)
        "figure.facecolor": FT_PAPER,
        "axes.facecolor": FT_PAPER,
        "savefig.facecolor": FT_PAPER,
        # horizontal-only gridlines, sitting behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": FT_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        # strip the box: FT charts keep only a baseline
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.edgecolor": FT_INK,
        "axes.linewidth": 0.9,
        # ticks: no tick marks, just labels (FT convention)
        "xtick.bottom": False,
        "ytick.left": False,
        # legend
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def ft_title(fig, title: str, subtitle: str | None = None,
             x: float = 0.02) -> None:
    """FT-style header: a bold left-aligned title with an optional lighter
    subtitle beneath it, plus the signature short accent rule above.

    Use INSTEAD of ax.set_title(), because FT titles sit against the figure
    edge (not centred over the axes) and carry a subtitle line.
    """
    # short accent rule above the title (a recognisable FT touch)
    fig.add_artist(plt.Line2D([x, x + 0.05], [0.965, 0.965],
                              transform=fig.transFigure,
                              color=ACCENT_COLOR, linewidth=3,
                              solid_capstyle="butt"))
    fig.text(x, 0.945, title, ha="left", va="top",
             fontsize=15, fontweight="bold", color=FT_INK)
    if subtitle:
        fig.text(x, 0.905, subtitle, ha="left", va="top",
                 fontsize=11, color=FT_INK)


def ft_source(fig, text: str, x: float = 0.02, y: float = 0.015) -> None:
    """FT-style source/footnote line at the foot of the figure: small, muted,
    left-aligned. Use INSTEAD of a plain figtext caption block.
    """
    fig.text(x, y, text, ha="left", va="bottom",
             fontsize=8, color=NEUTRAL_COLOR, wrap=True)

caption = ft_source
apply_style()
