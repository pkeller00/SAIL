"""Classical and multivariate spatial-autocorrelation baselines used in the paper."""

from .classical import composite_mean, global_moran_geary, local_moran_lisa
from .hotspot import HotspotResult, run_hotspot
from .multispati import MultispatiResult, multispati_pca
from .weights import KNNWeights, inverse_distance_from_graph, inverse_distance_knn

__all__ = [
    "KNNWeights",
    "inverse_distance_knn",
    "inverse_distance_from_graph",
    "global_moran_geary",
    "composite_mean",
    "local_moran_lisa",
    "multispati_pca",
    "MultispatiResult",
    "run_hotspot",
    "HotspotResult",
]
