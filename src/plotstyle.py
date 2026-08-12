"""Project-wide figure design system - Navy + Green fintech visual language.

Every exhibit in results/figures/ uses this single, consistent design system:
a dark navy "card" background, a bright high-contrast categorical palette
suited to dark UIs, clean sans-serif type, left-aligned titles with a
subtitle, horizontal-only gridlines, and no chart-border box. Matches the
Streamlit app's own navy+green theme (.streamlit/config.toml), so exhibits
look native inside the app rather than like a pasted-in light-mode image.

Colour psychology (fintech convention, e.g. Robinhood and most neobanks):
navy = trust/stability/security, green = growth/gains ("in the green").

The same colour always means the same thing across every figure:

    EQUITY_COLOR   - anything equity-related, most single-series lines
    CRYPTO_COLOR   - anything crypto-related
    NEUTRAL_COLOR  - baseline / control / "normal" observations (cool grey)
    ACCENT_COLOR   - flagged observations (outliers, anomalies, holdout,
                     drawdown/negative) - used ONLY for this purpose, so the
                     reader learns to read the accent colour as "flagged"
                     the first time and carries that reading through the
                     whole report. Deliberately NOT green, so it never
                     collides with green's "positive/growth" meaning.

Call apply_style() once before any figure is built (done on import). Two
helpers, ft_title() and ft_source(), reproduce the header/footer convention
and should be used in place of ax.set_title / plt.figtext.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Navy + Green palette
# ---------------------------------------------------------------------------
# Chart "paper" background and ink, plus an editorial categorical palette.
# Chosen to stay distinguishable under common colour-vision deficiencies and
# to read clearly against a dark navy background (bright/saturated rather
# than the muted tones a light-background palette would use).
FT_PAPER = "#132D4F"      # navy "card" background - matches the app's
                           # secondaryBackgroundColor
FT_INK = "#E8EDF5"        # light ink for text - matches the app's textColor
FT_GRID = "#2A4A73"       # subtle navy gridline, visible but not distracting

EQUITY_COLOR = "#38BDF8"  # sky blue - equity-related, most single-series lines
CRYPTO_COLOR = "#FB923C"  # orange - crypto-related
NEUTRAL_COLOR = "#94A3B8" # cool grey - control / baseline
ACCENT_COLOR = "#F87171"  # coral/red - flagged / anomalous / negative ONLY

# Categorical palette for breakdowns with >2 groups (e.g. the 10 GICS
# sectors), ordered so adjacent categories stay visually distinct against
# the navy background.
SECTOR_PALETTE = [
    "#38BDF8", "#FB923C", "#2DD4BF", "#A78BFA", "#F87171",
    "#00C896", "#FBBF24", "#94A3B8", "#F472B6", "#818CF8",
]

FONT_FAMILY = "DejaVu Sans"  # ships with matplotlib; renders identically on
                              # Windows/Mac/Linux (a reproducibility concern,
                              # not just aesthetic).


def apply_style() -> None:
    """Apply the navy+green rcParams globally. Safe to call repeatedly."""
    mpl.rcParams.update({
        # typography
        "font.family": FONT_FAMILY,
        "font.size": 9,
        "text.color": FT_INK,
        "axes.labelcolor": FT_INK,
        "xtick.color": FT_INK,
        "ytick.color": FT_INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,

        # navy "card" background (both the figure and the plot area)
        "figure.facecolor": FT_PAPER,
        "axes.facecolor": FT_PAPER,
        "savefig.facecolor": FT_PAPER,
        # horizontal-only gridlines, sitting behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": FT_GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        # strip the box: keep only a baseline
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.edgecolor": FT_GRID,
        "axes.linewidth": 0.9,
        # ticks: no tick marks, just labels
        "xtick.bottom": False,
        "ytick.left": False,
        # legend
        "legend.frameon": True,
        "legend.facecolor": FT_PAPER,
        "legend.edgecolor": FT_GRID,
        "legend.labelcolor": FT_INK,
        "legend.fontsize": 7.5,
    })


def ft_title(fig, title: str, subtitle: str | None = None,
             x: float = 0.02, title_fontsize: float = 12,
             subtitle_fontsize: float = 9) -> None:
    """Header: a bold left-aligned title with an optional lighter subtitle
    beneath it, plus a short accent rule above.

    Use INSTEAD of ax.set_title(), because these titles sit against the
    figure edge (not centred over the axes) and carry a subtitle line.

    title_fontsize/subtitle_fontsize let a caller compensate when a figure
    will be displayed at a different effective scale than the project
    default (e.g. a chart shown at full container width needs a SMALLER
    source fontsize than one shown in a half-width column, so the two look
    the same size on screen once Streamlit scales both to fit).
    """
    # short accent rule above the title
    fig.add_artist(plt.Line2D([x, x + 0.05], [0.965, 0.965],
                              transform=fig.transFigure,
                              color=ACCENT_COLOR, linewidth=3,
                              solid_capstyle="butt"))
    fig.text(x, 0.945, title, ha="left", va="top",
             fontsize=title_fontsize, fontweight="bold", color=FT_INK)
    if subtitle:
        fig.text(x, 0.905, subtitle, ha="left", va="top",
                 fontsize=subtitle_fontsize, color="#9CB3D1")


def ft_source(fig, text: str, x: float = 0.02, y: float = 0.015) -> None:
    """Source/footnote line at the foot of the figure: small, muted,
    left-aligned. Use INSTEAD of a plain figtext caption block.
    """
    fig.text(x, y, text, ha="left", va="bottom",
             fontsize=8, color="#7891B5", wrap=True)

caption = ft_source
apply_style()
