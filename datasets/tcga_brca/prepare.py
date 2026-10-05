"""Prepare TCGA-BRCA patch graphs for ``experiments/survival/tcga_brca.yaml``.

Input: an HDF5 file with one entry per slide (index ``i``)::

    x/<i>        (N_i, d) patch features
    coords/<i>   (N_i, 2) patch coordinates
    patient_id   (n,)     TCGA patient barcodes
    time, event  (n,)     disease-specific survival (days, 0/1)

Features are z-scored per feature over all patches of all slides.

    python datasets/tcga_brca/prepare.py --h5 tcga_graphs.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sail.data import pooled_zscore, save_samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/tcga_brca"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids, xs, coords, rows = [], [], [], []
    with h5py.File(args.h5, "r") as f:
        for i in range(len(f["event"])):
            if str(i) not in f["x"]:
                continue
            pid = f["patient_id"][i]
            pid = pid.decode() if isinstance(pid, bytes) else str(pid)
            sid = f"{pid}_{i}"  # unique per slide
            ids.append(sid)
            xs.append(f["x"][str(i)][()].astype(np.float32))
            coords.append(f["coords"][str(i)][()].astype(np.float32))
            rows.append(dict(id=sid, patient=pid, time=float(f["time"][i]), event=int(f["event"][i])))

    save_samples(args.out / "samples.pt", ids, pooled_zscore(xs), coords, [f"f{j}" for j in range(xs[0].shape[1])])
    pd.DataFrame(rows).to_csv(args.out / "clinical.csv", index=False)
    print(f"{len(ids)} slides -> {args.out}")


if __name__ == "__main__":
    main()
