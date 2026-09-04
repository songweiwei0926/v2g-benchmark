"""Metrics package — model-agnostic evaluation metrics for V2G-Benchmark.

Submodules
----------
ranking       : MRR, Top1, Recall@k, NDCG and candidate ranking.
classification: AUPRC, AUROC, MCC.
direction     : direction-of-effect accuracy, balanced accuracy, MCC, Spearman.
effect_size   : Pearson, Spearman, R² for effect-size regression.
"""

from .ranking import (
    rank_candidates,
    compute_mrr,
    compute_top1,
    compute_recall_at_k,
    compute_ndcg,
    compute_all_ranking_metrics,
)
from .classification import (
    compute_auprc,
    compute_auroc,
    compute_mcc,
    compute_all_classification_metrics,
)
from .direction import (
    compute_direction_accuracy,
    compute_balanced_accuracy,
    compute_direction_mcc,
    compute_spearman as compute_direction_spearman,
)
from .effect_size import (
    compute_pearson,
    compute_spearman as compute_effect_spearman,
    compute_r2,
)

__all__ = [
    # ranking
    "rank_candidates",
    "compute_mrr",
    "compute_top1",
    "compute_recall_at_k",
    "compute_ndcg",
    "compute_all_ranking_metrics",
    # classification
    "compute_auprc",
    "compute_auroc",
    "compute_mcc",
    "compute_all_classification_metrics",
    # direction
    "compute_direction_accuracy",
    "compute_balanced_accuracy",
    "compute_direction_mcc",
    "compute_direction_spearman",
    # effect size
    "compute_pearson",
    "compute_effect_spearman",
    "compute_r2",
]
