"""Plotting package for SCARFS benchmark figures.

Public API
----------
apply_defaults
    Apply the house-style rcParams (call once before any figure).
palette
    Return the 12-colour house palette as a list of hex strings.
dual_temperature_axis
    Attach a secondary K y-axis to an existing °C (or K) primary axis.
dual_temperature_xaxis
    Attach a secondary K x-axis to an existing °C (or K) primary x-axis.
parity_plot
    Per-species parity scatter with 1:1 line and ±10 % band.
error_vs_temperature
    Relative error vs temperature (°C primary, K secondary).
error_vs_residence_time
    Relative error vs residence time.
error_vs_conversion
    Relative error vs feed conversion (exposes near-inlet deficit RC-1).
relative_error_histogram
    Histogram of relative errors with median marker and ±10 % reference.
axial_profiles
    Overlay axial profiles (predicted vs reference) along the reactor axis.
"""

from .plot_defaults import (
    apply_defaults,
    dual_temperature_axis,
    dual_temperature_xaxis,
    palette,
)
from .figures import (
    axial_profiles,
    error_vs_conversion,
    error_vs_residence_time,
    error_vs_temperature,
    parity_plot,
    relative_error_histogram,
)

__all__ = [
    "apply_defaults",
    "palette",
    "dual_temperature_axis",
    "dual_temperature_xaxis",
    "parity_plot",
    "error_vs_temperature",
    "error_vs_residence_time",
    "error_vs_conversion",
    "relative_error_histogram",
    "axial_profiles",
]
