"""Models package — model adapters implementing the :class:`ModelAdapter` interface.

Families
--------
* **Family 0** — baselines (:mod:`v2gbench.models.baselines`)
* **Family 1 & 2** — published prediction importers (:mod:`v2gbench.models.published`)
* **Family 3** — sequence models: Borzoi, Enformer, AlphaGenome
* **Family 4** — disease prioritization: Open Targets L2G
* **Family 5** — auto-discovered supplementary models (:mod:`v2gbench.models.discover_encode`)
* **Track E** — integrated ensembles (:mod:`v2gbench.models.integrated`)
"""

from __future__ import annotations

from .base import ModelAdapter, APPLICABILITY_STATUS, SOURCE_MODES
from .baselines import (
    RandomAdapter,
    NearestTSSAdapter,
    InverseDistanceAdapter,
    ExponentialDistanceAdapter,
    NearestExpressedAdapter,
)
from .published import (
    PublishedPredictionAdapter,
    import_abc_predictions,
    import_re2g_predictions,
    import_sce2g_predictions,
    import_epimap_predictions,
    import_graphreg_predictions,
    import_pgboost_predictions,
    import_scent_predictions,
    import_signac_predictions,
    import_archr_predictions,
    import_cicero_predictions,
    e2g_to_v2g,
    handle_missing_prediction,
    IMPORTER_REGISTRY,
)
from .borzoi import (
    BorzoiAdapter,
    load_borzoi_model,
    score_variant_borzoi,
    ensemble_borzoi_scores,
    run_borzoi_inference,
)
from .enformer import (
    EnformerAdapter,
    load_enformer_model,
    score_variant_enformer,
    match_cage_tracks,
    run_enformer_inference,
)
from .alphagenome import (
    AlphaGenomeAdapter,
    create_client,
    score_variant_alphagenome,
    compute_signed_score,
    run_alphagenome_inference,
    cache_key,
    load_cache,
    save_cache,
)
from .opentargets_l2g import (
    OpenTargetsL2GAdapter,
    import_l2g_2021,
    import_l2g_current,
)
from .discover_encode import (
    discover_encode_models,
    validate_prediction_schema,
    build_supplementary_model_registry,
)
from .integrated import (
    IntegratedRankAdapter,
    IntegratedLogisticAdapter,
    IntegratedXGBoostAdapter,
    prepare_features,
    chromosome_folds,
    train_logistic,
    train_xgboost,
    ensure_no_leakage,
)

__all__ = [
    # base
    "ModelAdapter", "APPLICABILITY_STATUS", "SOURCE_MODES",
    # baselines
    "RandomAdapter", "NearestTSSAdapter", "InverseDistanceAdapter",
    "ExponentialDistanceAdapter", "NearestExpressedAdapter",
    # published
    "PublishedPredictionAdapter", "IMPORTER_REGISTRY",
    "import_abc_predictions", "import_re2g_predictions", "import_sce2g_predictions",
    "import_epimap_predictions", "import_graphreg_predictions",
    "import_pgboost_predictions", "import_scent_predictions",
    "import_signac_predictions", "import_archr_predictions",
    "import_cicero_predictions", "e2g_to_v2g", "handle_missing_prediction",
    # borzoi
    "BorzoiAdapter", "load_borzoi_model", "score_variant_borzoi",
    "ensemble_borzoi_scores", "run_borzoi_inference",
    # enformer
    "EnformerAdapter", "load_enformer_model", "score_variant_enformer",
    "match_cage_tracks", "run_enformer_inference",
    # alphagenome
    "AlphaGenomeAdapter", "create_client", "score_variant_alphagenome",
    "compute_signed_score", "run_alphagenome_inference",
    "cache_key", "load_cache", "save_cache",
    # opentargets
    "OpenTargetsL2GAdapter", "import_l2g_2021", "import_l2g_current",
    # discover
    "discover_encode_models", "validate_prediction_schema",
    "build_supplementary_model_registry",
    # integrated
    "IntegratedRankAdapter", "IntegratedLogisticAdapter", "IntegratedXGBoostAdapter",
    "prepare_features", "chromosome_folds", "train_logistic", "train_xgboost",
    "ensure_no_leakage",
]
