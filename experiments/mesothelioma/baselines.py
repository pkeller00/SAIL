"""Comparators for the mesothelioma benchmark, on the same folds as SAIL.

* ``mesograph`` - MesoGraph (branched EdgeConv, MesoGraph.py settings); needs the
  MesoGraph repository (``data.mesograph_repo``).
* ``pins``      - positive-instance-sampling MIL (Eastwood et al., AIME 2022),
  re-implemented on the same per-cell features (instance = cell, bag = core);
  the original operates on image patches.
* ``max_mil`` / ``naive_mil`` - instance MIL with max / mean bag pooling.

Every runner writes ``<method>/fold_metrics.csv`` and ``<method>/core_predictions.csv``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import meso_data as md
import sail
from sail.config import ExperimentConfig
from sail.experiment import device_of, write_table


def _cells(g, device) -> torch.Tensor:
    """Real-cell features of a core (virtual core node dropped)."""
    return g.x.float()[:-1].to(device)


def _write(cfg, method, preds, rows):
    out = cfg.out / method
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_csv(out / "core_predictions.csv", index=False)
    write_table(pd.DataFrame(rows), out / "fold_metrics.csv", f"{method} per fold")


def _metrics(method, fold, split, df) -> dict:
    auc, ap = md.auc_ap(df["y"].map(md.binarize), df["y_pred"])
    return dict(method=method, fold=fold, split=split, auc=auc, ap=ap)


def _cross_validate(cfg, method, fit, score):
    """Shared fold loop: ``fit(fold, cores, labels) -> model``, ``score(model, cores) -> y_pred list``."""
    cores = md.load_cores(cfg.path("graphs"))
    labels = {c: md.label(g) for c, g in cores.items()}
    preds, rows = [], []
    for fold in md.load_splits(cfg.path("splits"))["folds"]:
        sail.set_seed(cfg.seed + fold["fold"])
        model = fit(fold, cores, labels)
        for split in ("val", "test"):
            df = pd.DataFrame(dict(core=fold[split], y=[labels[c] for c in fold[split]],
                                   y_pred=score(model, [cores[c] for c in fold[split]])))
            rows.append(_metrics(method, fold["fold"], split, df))
            if split == "test":
                preds.append(df.assign(fold=fold["fold"]))
    _write(cfg, method, preds, rows)


# --------------------------------------------------------------------------- #
# instance MIL
# --------------------------------------------------------------------------- #
class InstanceMLP(nn.Module):
    def __init__(self, in_dim: int, n_out: int, hidden: int = 128, p: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def _mil(cfg: ExperimentConfig, pooling: str) -> None:
    device = device_of(cfg)
    epochs = cfg.train["epochs"]

    def bag_logit(model, x):
        inst = model(x).squeeze(-1)
        return inst.max() if pooling == "max" else inst.mean()

    def score(model, graphs):
        model.eval()
        with torch.no_grad():
            return [float(torch.sigmoid(bag_logit(model, _cells(g, device)))) for g in graphs]

    def fit(fold, cores, labels):
        train = fold["train"]
        model = InstanceMLP(cores[train[0]].x.shape[1], 1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        y = np.array([md.binarize(labels[c]) for c in train])
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max((1 - y).sum(), 1) / max(y.sum(), 1)], device=device))
        best, state = -1.0, None
        for _ in range(epochs):
            model.train()
            order = np.random.permutation(len(train))
            for b in range(0, len(order), 16):
                idx = order[b:b + 16]
                opt.zero_grad()
                logits = torch.stack([bag_logit(model, _cells(cores[train[i]], device)) for i in idx])
                bce(logits, torch.tensor(y[idx], dtype=torch.float32, device=device)).backward()
                opt.step()
            auc, _ = md.auc_ap([md.binarize(labels[c]) for c in fold["val"]], score(model, [cores[c] for c in fold["val"]]))
            if not np.isnan(auc) and auc >= best:  # model selection on validation AUC
                best, state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        return model

    _cross_validate(cfg, f"{pooling}_mil", fit, score)


def max_mil(cfg):
    _mil(cfg, "max")


def naive_mil(cfg):
    _mil(cfg, "naive")


def pins(cfg: ExperimentConfig, lr: float = 5e-4, weight_decay: float = 1e-4, alpha: float = 2.0,
         per_bag: int = 64, batch: int = 256) -> None:
    """Instance classifier trained on cells sampled in proportion to predicted positivity ** alpha."""
    device = device_of(cfg)
    epochs = cfg.train["epochs"]

    def score(model, graphs):
        model.eval()
        with torch.no_grad():
            return [float(F.softmax(model(_cells(g, device)), 1)[:, 1].mean()) for g in graphs]

    def fit(fold, cores, labels):
        train = fold["train"]
        model = InstanceMLP(cores[train[0]].x.shape[1], 2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        y = np.array([md.binarize(labels[c]) for c in train])
        n = len(y)
        ce = nn.CrossEntropyLoss(weight=torch.tensor([n / (2 * max((1 - y).sum(), 1)), n / (2 * max(y.sum(), 1))],
                                                     dtype=torch.float32, device=device))
        best, state = -1.0, None
        for ep in range(epochs):
            feats, targets = [], []
            for c in np.random.permutation(train):
                x = _cells(cores[c], device)
                m = min(per_bag, len(x))
                if ep == 0:
                    idx = torch.randint(len(x), (m,), device=device)
                else:
                    model.eval()
                    with torch.no_grad():
                        w = F.softmax(model(x), 1)[:, 1].clamp_min(1e-6) ** alpha
                    idx = torch.multinomial(w / w.sum(), m, replacement=True)
                feats.append(x[idx])
                targets.append(torch.full((m,), md.binarize(labels[c]), dtype=torch.long, device=device))
            feats, targets = torch.cat(feats), torch.cat(targets)
            perm = torch.randperm(len(feats), device=device)
            model.train()
            for b in range(0, len(perm), batch):
                bi = perm[b:b + batch]
                opt.zero_grad()
                ce(model(feats[bi]), targets[bi]).backward()
                opt.step()
            auc, _ = md.auc_ap([md.binarize(labels[c]) for c in fold["val"]], score(model, [cores[c] for c in fold["val"]]))
            if not np.isnan(auc) and auc >= best:
                best, state = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        return model

    _cross_validate(cfg, "pins", fit, score)


# --------------------------------------------------------------------------- #
def mesograph(cfg: ExperimentConfig) -> None:
    """MesoGraph with the settings of MesoGraph.py (needs the MesoGraph repository)."""
    sys.path.insert(0, str(cfg.path("mesograph_repo")))
    from meso_engine import NetWrapper
    from meso_models import MesoBranched
    from torch.optim.lr_scheduler import CyclicLR
    from torch.utils.data import Sampler
    from torch_geometric.loader import DataLoader

    class StratifiedSampler(Sampler):
        """Stratified mini-batches, as in MesoGraph.py."""

        def __init__(self, labels: torch.Tensor, batch_size: int = 16) -> None:
            self.labels, self.n_splits = labels, max(int(labels.size(0) / batch_size), 2)

        def __iter__(self):
            from sklearn.model_selection import StratifiedKFold

            y = self.labels.numpy()
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
            return iter([t for _, t in skf.split(np.arange(len(y)), y)])

        def __len__(self):
            return len(self.labels)

    device = device_of(cfg)
    out = cfg.out / "mesograph"
    out.mkdir(parents=True, exist_ok=True)
    nets = {}

    def fit(fold, cores, labels):
        train = [cores[c] for c in fold["train"]]
        y = torch.tensor([labels[c] for c in fold["train"]])
        loader = DataLoader(train, batch_sampler=StratifiedSampler(y, 16))
        model = MesoBranched(dim_features=train[0].x.shape[1], dim_target=2, layers=[20, 10, 10], dropout=0,
                             pooling="mean", eps=100.0, train_eps=False, do_ls=True)
        net = NetWrapper(model, loss_function=None, device=device, save_dir=out)
        model = model.to(net.device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.02)
        sched = CyclicLR(opt, 5e-5, 25e-5, 40 * len(loader), mode="exp_range", gamma=0.8, cycle_momentum=False)
        best, *_ = net.train(train_loader=loader, max_epochs=cfg.train["epochs"], optimizer=opt, scheduler=sched,
                             clipping=None, validation_loader=DataLoader([cores[c] for c in fold["val"]], shuffle=True),
                             test_loader=DataLoader([cores[c] for c in fold["test"]], shuffle=True),
                             early_stopping=None, log_every=50)
        nets[id(best)] = net
        return best

    def score(model, graphs):
        _, df = nets[id(model)].predict(graphs, model)
        order = {str(c): i for i, c in enumerate(df["core"])}
        return [float(df["y_pred"].iloc[order[str(getattr(g, "core", ""))]]) for g in graphs]

    _cross_validate(cfg, "mesograph", fit, score)


RUNNERS = {"mesograph": mesograph, "pins": pins, "max_mil": max_mil, "naive_mil": naive_mil}
