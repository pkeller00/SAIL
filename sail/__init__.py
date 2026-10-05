"""SAIL: a learnable generalisation of spatial autocorrelation over graph-structured data.

Quick start
-----------
>>> import sail
>>> g = sail.build_graph(coords, features, method="knn", k=32)
>>> modes = sail.SAILModes(input_dim=features.shape[1], n_modes=20).fit([g])
>>> labels = sail.lisa(g, modes.model)          # (N, 20) LISA cluster ids
"""

from .feature_affinity import QueryKeyAffinity, SharedMapAffinity
from .feature_maps import GumbelSoftmaxMap, Linear, NormalisedLinear, OneHotMap, StiefelLinear
from .graph import SpatialGraph, build_graph, concat_graphs
from .lisa import LISA_LABELS, LISA_NAMES, lisa
from .local_indices import geary, getis_ord_gstar, moran, morisita_horn
from .model import SAIL, SAILOutput
from .modes import ModesConfig, SAILModes
from .utils import default_device, set_seed

__version__ = "1.0.0"

__all__ = [
    "SAIL",
    "SAILOutput",
    "SAILModes",
    "ModesConfig",
    "SpatialGraph",
    "build_graph",
    "concat_graphs",
    "lisa",
    "LISA_LABELS",
    "LISA_NAMES",
    "moran",
    "geary",
    "getis_ord_gstar",
    "morisita_horn",
    "NormalisedLinear",
    "StiefelLinear",
    "Linear",
    "OneHotMap",
    "GumbelSoftmaxMap",
    "SharedMapAffinity",
    "QueryKeyAffinity",
    "set_seed",
    "default_device",
]
