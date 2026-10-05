"""Bit-for-bit parity with the pre-release code that produced the paper results.

Skipped unless ``SAIL_LEGACY_PATH`` points at the old package directory
(the folder containing ``layers.py``, ``modes.py``, ...).
"""

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import sail

LEGACY = os.environ.get("SAIL_LEGACY_PATH")
pytestmark = [
    pytest.mark.legacy,
    pytest.mark.skipif(not LEGACY, reason="SAIL_LEGACY_PATH not set"),
]


@pytest.fixture(scope="module")
def old():
    # import submodules without running the old __init__ (which needs tiatoolbox)
    pkg = types.ModuleType("sail_legacy")
    pkg.__path__ = [LEGACY]
    sys.modules["sail_legacy"] = pkg
    return types.SimpleNamespace(
        layers=importlib.import_module("sail_legacy.layers"),
        modes=importlib.import_module("sail_legacy.modes"),
        clustering=importlib.import_module("sail_legacy.clustering"),
    )


def _graphs(n_graphs=3, d=12, k=8):
    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(n_graphs):
        n = int(rng.integers(150, 250))
        coords = rng.uniform(0, 10, size=(n, 2))
        x = rng.normal(size=(n, d)) + np.sin(coords[:, :1]) * rng.normal(size=(1, d))
        graphs.append(sail.build_graph(coords, x.astype(np.float32), method="knn", k=k))
    return graphs


CASES = [
    dict(local="geary_index", new_local="geary", objective="minimize", f_theta=None, stiefel=True),
    dict(local="moran_index", new_local="moran", objective="maximize", f_theta=None, stiefel=False),
    dict(local="geary_index", new_local="geary", objective="minimize", f_theta="QKTheta", stiefel=True),
    dict(local="getis_ord_gstar_index", new_local="gstar", objective="maximize",
         f_theta="SharedFMapTheta", stiefel=False),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['new_local']}-{c['f_theta']}-stiefel{c['stiefel']}")
def test_training_and_lisa_match(old, case):
    graphs = _graphs()
    d, h = graphs[0].num_features, 5
    common = dict(lr=0.05, epochs=25, batch_size=2)

    sail.set_seed(0)
    legacy = old.modes.SAILModes(
        input_dim=d, k=h, objective=case["objective"], use_steifel_manifold=case["stiefel"],
        f_theta=case["f_theta"], local_index_fn=getattr(old.layers.SAIL, case["local"]),
        decorrelate=True, lambda_corr=1.0, sparse=True, lambda_sparse=1.0, **common,
    )
    legacy.train(graphs)

    sail.set_seed(0)
    new = sail.SAILModes(
        d, n_modes=h, objective=case["objective"], stiefel=case["stiefel"],
        f_affinity={None: None, "QKTheta": "query_key", "SharedFMapTheta": "shared_map"}[case["f_theta"]],
        local_index=case["new_local"], lambda_corr=1.0, lambda_sparse=1.0, **common,
    ).fit(graphs, verbose=False)

    torch.testing.assert_close(new.W, legacy.model.f_map.W.detach())
    torch.testing.assert_close(new.mean_scores, legacy.best_mean_I)
    torch.testing.assert_close(new.model.temperature, legacy.model.temperature)

    torch.manual_seed(1)
    old_labels = old.clustering.lisa_clustering(graphs[0], legacy.model, permutations=99).numpy()
    torch.manual_seed(1)
    new_labels = sail.lisa(graphs[0], new.model, permutations=99)
    np.testing.assert_array_equal(new_labels, old_labels)


def test_legacy_checkpoint_roundtrip(old, tmp_path):
    graphs = _graphs()
    d = graphs[0].num_features
    sail.set_seed(0)
    legacy = old.modes.SAILModes(
        input_dim=d, k=4, epochs=5, lr=0.05, objective="minimize", use_steifel_manifold=True,
        local_index_fn=old.layers.SAIL.geary_index, decorrelate=True, sparse=True, lambda_sparse=1.0,
    )
    legacy.train(graphs)
    path = tmp_path / "legacy.pth"
    legacy.save(str(path))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from convert_legacy_checkpoints import load_legacy_modes

    new = load_legacy_modes(path, input_dim=d, local_index="geary")
    np.testing.assert_allclose(
        new.scores(graphs),
        np.stack([legacy.model([g]).detach()[0].numpy() for g in graphs]),
        rtol=1e-6, atol=1e-6,
    )
    new.save(tmp_path / "new.pt")
    again = sail.SAILModes.load(tmp_path / "new.pt")
    np.testing.assert_allclose(again.scores(graphs), new.scores(graphs))
    assert again.trained_with_stiefel
