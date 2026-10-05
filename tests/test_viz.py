"""Smoke tests for overlays with every image source."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import sail
from sail.viz import AnnDataImage, ArrayImage, overlay, plot_modes


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(200, 2))
    x = (rng.normal(size=(200, 6)) + coords[:, :1] / 50).astype(np.float32)
    return sail.build_graph(coords, x, k=8)


def test_overlay_plain_and_lisa(graph):
    overlay(graph.coords, graph.x[:, 0], kind="continuous")
    overlay(graph.coords, np.random.default_rng(0).integers(0, 5, graph.num_nodes), kind="lisa")


def test_overlay_array_image(graph):
    img = np.zeros((50, 50, 3))
    ax = overlay(graph.coords, graph.x[:, 0], image=ArrayImage(img, scale=0.5))
    xs = ax.collections[0].get_offsets()[:, 0]
    assert xs.max() <= 50


@pytest.mark.parametrize("img_key, factor_key", [("hires", "tissue_hires_scalef"), ("my_stain", None)])
def test_overlay_anndata_image(graph, img_key, factor_key):
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(np.asarray(graph.x))
    adata.obsm["spatial"] = np.asarray(graph.coords)
    scalefactors = {factor_key: 0.25} if factor_key else {}
    adata.uns["spatial"] = {"lib": {"images": {img_key: np.ones((30, 30, 3))}, "scalefactors": scalefactors}}

    src = AnnDataImage(adata, img_key=img_key)
    assert src.scale == (0.25 if factor_key else 1.0)
    src = AnnDataImage(adata, img_key=img_key, scale=0.3)  # explicit scale wins
    overlay(adata.obsm["spatial"], graph.x[:, 1], image=src)
    with pytest.raises(KeyError):
        AnnDataImage(adata, img_key="missing")


def test_plot_modes(graph):
    sail.set_seed(0)
    modes = sail.SAILModes(graph.num_features, n_modes=3, epochs=3).fit([graph], verbose=False)
    fig = plot_modes(graph, modes, [f"f{i}" for i in range(6)], ncols=2, top_k=4)
    assert len(fig.axes) == 8  # 2 rows of modes x (map + bars) x 2 columns
    assert sum(ax.axison for ax in fig.axes) == 3  # one bar panel per mode; unused panels hidden
