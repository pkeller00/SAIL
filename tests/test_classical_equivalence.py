"""Theorem IV.1: SAIL recovers classical spatial statistics exactly."""

import numpy as np
import pytest
import torch

import sail
from sail.local_indices import geary, getis_ord_gstar, moran, morisita_horn


def _row_stochastic_knn(n=60, k=6, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    x = rng.normal(size=(n, 1)) + coords[:, :1] * 3  # spatially structured
    g = sail.build_graph(coords, x, method="knn", k=k, log_distances=False)
    weights = torch.full((n, k), 1.0 / k, dtype=torch.float64)
    return g, weights, x[:, 0].astype(np.float64)


def _dense_w(neighbors, weights, n):
    W = np.zeros((n, n))
    for i in range(n):
        for j, w in zip(neighbors[i].numpy(), weights[i].numpy()):
            W[i, j] += w
    return W


def test_moran_matches_classical():
    g, w, x = _row_stochastic_knn()
    n = len(x)
    W = _dense_w(g.neighbors, w, n)
    dev = x - x.mean()
    classical = n / W.sum() * (dev @ W @ dev) / (dev @ dev)

    z = torch.tensor((dev / dev.std(ddof=0))[:, None])
    ours = moran(z, g.neighbors, w).mean().item()
    assert ours == pytest.approx(classical, abs=1e-10)


def test_geary_matches_classical():
    g, w, x = _row_stochastic_knn()
    n = len(x)
    W = _dense_w(g.neighbors, w, n)
    dev = x - x.mean()
    diff = (x[:, None] - x[None, :]) ** 2
    classical = (n - 1) * (W * diff).sum() / (2 * W.sum() * (dev @ dev))

    z = torch.tensor((dev / dev.std(ddof=1))[:, None])
    ours = geary(z, g.neighbors, w).mean().item()
    assert ours == pytest.approx(classical, abs=1e-10)


def test_gstar_matches_classical():
    g, w, x = _row_stochastic_knn()
    n = len(x)
    # G* uses self-weights: put node i in its own neighbourhood
    nbr = torch.cat([torch.arange(n)[:, None], g.neighbors], dim=1)
    k = nbr.shape[1]
    wt = torch.full((n, k), 1.0 / k, dtype=torch.float64)
    W = _dense_w(nbr, wt, n)
    xbar, s = x.mean(), x.std(ddof=0)
    wsum, w2 = W.sum(1), (W**2).sum(1)
    classical = (W @ x - xbar * wsum) / (s * np.sqrt((n * w2 - wsum**2) / (n - 1)))

    z = torch.tensor(((x - xbar) / s)[:, None])
    ours = getis_ord_gstar(z, nbr, wt, eps=0.0)[:, 0].numpy()
    np.testing.assert_allclose(ours, classical, atol=1e-10)


def test_morisita_horn_matches_classical():
    rng = np.random.default_rng(1)
    n, k, d = 40, 5, 4
    counts = rng.integers(1, 10, size=(n, d)).astype(np.float64)
    u = counts / counts.sum(1, keepdims=True)
    g = sail.build_graph(rng.uniform(size=(n, 2)), u, method="knn", k=k)
    w = torch.full((n, k), 1.0 / k, dtype=torch.float64)
    v = _dense_w(g.neighbors, w, n) @ u
    classical = 2 * (u * v).sum(1) / ((u**2).sum(1) + (v**2).sum(1))

    ours = morisita_horn(torch.tensor(u), g.neighbors, w, eps=0.0)[:, 0].numpy()
    np.testing.assert_allclose(ours, classical, atol=1e-12)


def test_model_reproduces_moran_with_fixed_uniform_affinity():
    """End-to-end through SAIL.forward: identity map, tau=0 -> uniform kNN weights."""
    g, w, x = _row_stochastic_knn()
    model = sail.SAIL(local_index="moran", learnable_temperature=False, init_temp=-1e4).double()
    g.x = g.x.double()
    g.distances = g.distances.double()
    out = model(g)
    expected = moran(torch.tensor(((x - x.mean()) / x.std(ddof=0))[:, None]), g.neighbors, w).mean()
    assert out.global_scores.item() == pytest.approx(expected.item(), abs=1e-6)
