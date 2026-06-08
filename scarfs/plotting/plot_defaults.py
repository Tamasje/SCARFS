from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PALETTE: list[str] = [
    "#3A0CA3", "#4361EE", "#7209B7", "#F72585", "#CBE0D1", "#FBEE43",
    "#00A320", "#4CC9F0", "#FF6B35", "#005F73", "#9D4EDD", "#7F4F24",
]

def palette() -> list[str]:
    return list(PALETTE)

def apply_defaults() -> None:
    rc = plt.rcParams
    rc["font.size"] = 16; rc["axes.labelsize"] = 18; rc["axes.titlesize"] = 18
    rc["xtick.labelsize"] = 16; rc["ytick.labelsize"] = 16; rc["legend.fontsize"] = 16
    rc["xtick.direction"] = "in"; rc["ytick.direction"] = "in"
    rc["xtick.major.size"] = 8; rc["xtick.minor.size"] = 4
    rc["ytick.major.size"] = 8; rc["ytick.minor.size"] = 4
    rc["xtick.major.width"] = 1.5; rc["ytick.major.width"] = 1.5
    rc["savefig.dpi"] = 400; rc["savefig.bbox"] = "tight"; rc["figure.dpi"] = 100
    rc["legend.frameon"] = False
    from cycler import cycler
    rc["axes.prop_cycle"] = cycler(color=PALETTE)

def dual_temperature_axis(ax: Axes, primary: str = "C") -> Axes:
    if primary == "C":
        secax = ax.secondary_yaxis("right", functions=(lambda c: c + 273.15, lambda k: k - 273.15)); secax.set_ylabel("Temperature [K]")
    elif primary == "K":
        secax = ax.secondary_yaxis("right", functions=(lambda k: k - 273.15, lambda c: c + 273.15)); secax.set_ylabel("Temperature [°C]")
    else:
        raise ValueError(f"primary must be 'C' or 'K', got {primary!r}")
    return secax

def dual_temperature_xaxis(ax: Axes, primary: str = "C") -> Axes:
    if primary == "C":
        secax = ax.secondary_xaxis("top", functions=(lambda c: c + 273.15, lambda k: k - 273.15)); secax.set_xlabel("Temperature [K]")
    elif primary == "K":
        secax = ax.secondary_xaxis("top", functions=(lambda k: k - 273.15, lambda c: c + 273.15)); secax.set_xlabel("Temperature [°C]")
    else:
        raise ValueError(f"primary must be 'C' or 'K', got {primary!r}")
    return secax
