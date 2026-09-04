"""Statistics package — bootstrap, paired comparison and deterministic sampling.

Submodules
----------
bootstrap : unit-level bootstrap CIs and paired bootstrap p-values.
paired    : all-pairs model comparison with Benjamini–Hochberg FDR.
sampling  : SHA256-based deterministic stratified sampling and binning.
"""

from .bootstrap import (
    bootstrap_metrics,
    bootstrap_paired,
    compute_ci,
    compute_bootstrap_p_value,
)
from .paired import (
    pairwise_comparison,
    benjamini_hochberg,
    format_comparison_results,
)
from .sampling import (
    deterministic_stratified_sample,
    assign_distance_bin,
    assign_pip_bin,
    assign_nearest_rank,
)

__all__ = [
    # bootstrap
    "bootstrap_metrics",
    "bootstrap_paired",
    "compute_ci",
    "compute_bootstrap_p_value",
    # paired
    "pairwise_comparison",
    "benjamini_hochberg",
    "format_comparison_results",
    # sampling
    "deterministic_stratified_sample",
    "assign_distance_bin",
    "assign_pip_bin",
    "assign_nearest_rank",
]
