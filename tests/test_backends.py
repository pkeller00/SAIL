"""kNN, radius and dense graphs share one forward pass; masking must be exact."""

import numpy as np
import torch

import sail


def _grid(n_side=8):
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(coords), 5)).astype(np.float32)
    return coords, x


def _model(local="geary", affinity=False, d=5):
    torch.manual_seed(0)
    f_map = sail.NormalisedLinear(d, 3)
    f_aff = sail.QueryKeyAffinity(d) if affinity else None
    return sail.SAIL(f_map=f_map, f_affinity=f_aff, local_index=local)


def _sorted(g):
    """Canonical neighbour order so different builders can be compared slot-wise."""
    order = torch.argsort(g.neighbors, dim=1)
    g.neighbors = torch.gather(g.neighbors, 1, order)
    g.distances = torch.gather(g.distances, 1, order)
    if g.mask is not None:
        g.mask = torch.gather(g.mask, 1, order)
    return g


def test_radius_matches_knn_on_regular_interior():
    # On a lattice with radius 1, interior nodes have exactly 4 neighbours.
    coords, x = _grid()
    gr = sail.build_graph(coords, x, method="radius", radius=1.01)
    assert gr.mask is not None and gr.mask.sum(1).max() == 4
    interior = (gr.mask.sum(1) == 4).numpy()
    gk = sail.build_graph(coords, x, method="knn", k=4)
    for local in ["moran", "geary", "gstar"]:
        model = _model(local)
        a = model(_sorted(gr)).local[0][interior]
        b = model(_sorted(gk)).local[0][interior]
        torch.testing.assert_close(a, b)


def test_extra_padding_changes_nothing():
    coords, x = _grid()
    g = sail.build_graph(coords, x, method="knn", k=6)
    pad = 3
    padded = sail.SpatialGraph(
        x=g.x,
        neighbors=torch.cat([g.neighbors, torch.zeros(g.num_nodes, pad, dtype=torch.long)], 1),
        distances=torch.cat([g.distances, torch.full((g.num_nodes, pad), -5.0)], 1),
        mask=torch.cat([torch.ones_like(g.neighbors, dtype=torch.bool),
                        torch.zeros(g.num_nodes, pad, dtype=torch.bool)], 1),
        kind="radius",
    )
    for affinity in [False, True]:
        model = _model("geary", affinity)
        out_a, out_b = model(g), model(padded)
        torch.testing.assert_close(out_a.local[0], out_b.local[0])
        ga = torch.autograd.grad(out_a.global_scores.sum(), model.raw_temperature)[0]
        gb = torch.autograd.grad(model(padded).global_scores.sum(), model.raw_temperature)[0]
        torch.testing.assert_close(ga, gb)


def test_dense_matches_brute_force():
    coords, x = _grid(5)
    g = sail.build_graph(coords, x, method="dense", log_distances=False)
    n = g.num_nodes
    assert g.neighbors.shape == (n, n - 1)
    model = _model("moran")
    out = model(g)

    D = torch.cdist(torch.tensor(coords), torch.tensor(coords))
    logits = -model.temperature * D
    logits.fill_diagonal_(float("-inf"))
    A = torch.softmax(logits, dim=1)
    z = model.project(g.x)
    expected = z * (A @ z)
    torch.testing.assert_close(out.local[0], expected, rtol=1e-5, atol=1e-5)


def test_isolated_radius_node_gets_zero_weight_not_nan():
    coords = np.array([[0, 0], [0.5, 0], [10, 10]], dtype=np.float32)
    x = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    g = sail.build_graph(coords, x, method="radius", radius=1.0)
    out = _model("geary")(g)
    assert torch.isfinite(out.local[0]).all()
    assert out.weights[0][2].sum() == 0


def test_mode_chunk_is_exact():
    coords, x = _grid()
    g = sail.build_graph(coords, x, method="knn", k=6)
    for local in ["moran", "geary", "gstar"]:
        model = _model(local)
        full = model(g)
        model.mode_chunk = 2
        chunked = model(g)
        torch.testing.assert_close(full.local[0], chunked.local[0])
