"""Baselines: sanity checks, plus parity with the pre-release scripts when available."""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

from sail.baselines import (
    composite_mean,
    global_moran_geary,
    inverse_distance_knn,
    local_moran_lisa,
    multispati_pca,
)

pytest.importorskip("esda")
SCRIPTS = os.environ.get("SAIL_LEGACY_SCRIPTS")  # folder with sail_squidpy_4i_*.py, ORION_CRC_*.py


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 50, size=(300, 2))
    X = rng.normal(size=(300, 6)) + np.sin(coords[:, :1] / 8) * np.array([[2, 1, 0, 0, -1, 0]])
    return coords, X


def test_global_statistics_detect_structure(data):
    coords, X = data
    w = inverse_distance_knn(coords, k=8, log_distances=False)
    dist = np.linalg.norm(coords[:, None] - coords[w.neighbors], axis=-1)
    moran, geary = global_moran_geary(X, w.neighbors, dist)
    assert moran[0] > 0.3 and geary[0] < 0.7      # structured feature
    assert abs(moran[3]) < 0.1 and abs(geary[3] - 1) < 0.15  # noise feature


def test_multispati_first_axis_is_spatial(data):
    coords, X = data
    res = multispati_pca([X], [inverse_distance_knn(coords, k=8)])
    np.testing.assert_allclose(res.eigenvalues, res.variance * res.moran)
    assert res.kept[0] and abs(res.loadings[0, 0]) > 0.5


def test_local_moran_lisa_labels(data):
    coords, X = data
    labels = local_moran_lisa(X[:, 0], inverse_distance_knn(coords, k=8), permutations=199, seed=0)
    assert set(np.unique(labels)) <= {0, 1, 2, 3, 4} and (labels > 0).any()


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def legacy():
    if not SCRIPTS:
        pytest.skip("SAIL_LEGACY_SCRIPTS not set")
    sys.path.insert(0, SCRIPTS)
    return dict(
        moran4i=importlib.import_module("sail_squidpy_4i_baseline_final"),
        ms4i=importlib.import_module("sail_squidpy_4i_multispati_baseline"),
        orion=importlib.import_module("ORION_CRC_hotspot_baseline"),
    )


@pytest.mark.legacy
def test_weights_match_legacy(legacy, data):
    coords, _ = data
    old = legacy["moran4i"].build_knn_weights(coords, k=8, use_log=True)
    new = inverse_distance_knn(coords, k=8, log_distances=True).to_libpysal()
    np.testing.assert_allclose(old.sparse.toarray(), new.sparse.toarray(), rtol=1e-12)


@pytest.mark.legacy
def test_multispati_matches_legacy(legacy, data):
    coords, X = data
    Z = (X - X.mean(0)) / X.std(0)
    W = legacy["moran4i"].build_knn_weights(coords, k=8, use_log=True)
    _, old_load, old_eig, old_var, old_moran = legacy["ms4i"].multispati_pca(Z, W, Z.shape[1])
    new = multispati_pca([X], [inverse_distance_knn(coords, k=8)])
    np.testing.assert_allclose(new.eigenvalues, old_eig, rtol=1e-10)
    np.testing.assert_allclose(np.abs(new.loadings), np.abs(old_load), atol=1e-8)
    np.testing.assert_allclose(new.moran, old_moran, rtol=1e-8)


@pytest.mark.legacy
def test_global_moran_geary_matches_legacy(legacy, data):
    coords, X = data
    w = inverse_distance_knn(coords, k=8, log_distances=False)
    dist = np.linalg.norm(coords[:, None] - coords[w.neighbors], axis=-1)
    old_I, old_C = legacy["orion"].global_moran_geary(
        torch.tensor(X), torch.tensor(w.neighbors), torch.tensor(dist))
    new_I, new_C = global_moran_geary(X, w.neighbors, dist)
    np.testing.assert_allclose(new_I, old_I, rtol=1e-10)
    np.testing.assert_allclose(new_C, old_C, rtol=1e-10)


@pytest.mark.legacy
def test_local_lisa_matches_legacy(legacy, data):
    coords, X = data
    W = legacy["moran4i"].build_knn_weights(coords, k=8, use_log=True)
    w = inverse_distance_knn(coords, k=8)
    np.random.seed(3)
    old = legacy["ms4i"].local_moran_lisa(X[:, 0], W, permutations=199)
    np.random.seed(3)
    new = local_moran_lisa(X[:, 0], w, permutations=199)
    np.testing.assert_array_equal(new, old)

    np.random.seed(4)
    old_mv = legacy["moran4i"].multivariate_morans_i_labels(list(X.T), W, permutations=199)
    np.random.seed(4)
    new_mv = local_moran_lisa(composite_mean(X), w, permutations=199)
    np.testing.assert_array_equal(new_mv, old_mv)
