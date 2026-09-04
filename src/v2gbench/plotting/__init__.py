"""Plotting package — main and supplementary figures for V2G-Benchmark.

All figure modules use matplotlib + seaborn (no R/ggplot2), colorblind-friendly
palettes, and write SVG output with ``svg.fonttype='none'`` and
``font.family`` set to ``['Liberation Sans', 'Arimo', 'DejaVu Sans']``.

Submodules
----------
fig1_overview        : Figure 1 — benchmark overview (5 panels).
fig2_heatmap         : Figure 2 — model x benchmark heatmap.
fig3_distance        : Figure 3 — distance stratification (4 panels).
fig4_context         : Figure 4 — context matching.
fig5_complementarity : Figure 5 — sequence vs E2G complementarity.
fig6_integrated      : Figure 6 — integrated model comparison.
supplementary        : Supplementary figures S1–S7.
tables               : Main and supplementary table generation.
"""

from .fig1_overview import make_fig1
from .fig2_heatmap import make_fig2
from .fig3_distance import make_fig3
from .fig4_context import make_fig4
from .fig5_complementarity import make_fig5
from .fig6_integrated import make_fig6
from .supplementary import make_supplementary_figures
from .tables import (
    make_main_results_table,
    make_supplementary_tables,
    make_model_exclusions_table,
    make_mandatory_completion_matrix,
)

__all__ = [
    "make_fig1",
    "make_fig2",
    "make_fig3",
    "make_fig4",
    "make_fig5",
    "make_fig6",
    "make_supplementary_figures",
    "make_main_results_table",
    "make_supplementary_tables",
    "make_model_exclusions_table",
    "make_mandatory_completion_matrix",
]
