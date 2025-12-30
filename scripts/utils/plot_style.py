from __future__ import annotations

from cycler import cycler


def apply_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "text.color": "#220126",
            "axes.labelcolor": "#220126",
            "axes.edgecolor": "#495773",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.color": "#220126",
            "ytick.color": "#220126",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "grid.color": "#495773",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "axes.prop_cycle": cycler(
                "color",
                [
                    "#220126",
                    "#4A90C2",
                    "#495773",
                    "#6B73A1",
                    "#2D0140",
                    "#7A6C86",
                ],
            ),
        }
    )
