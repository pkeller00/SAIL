"""Prepare TCGA-CRC (COAD + READ) patch graphs for ``experiments/survival/tcga_crc.yaml``.

Input
-----
* a folder with one graph file per slide (``torch.save``-d objects with ``x``
  (N, d) UNI patch features and ``coords`` (N, 2)); the first 12 characters of
  the file name are the TCGA patient barcode;
* the TCGA-CDR survival table (Liu et al., Cell 2018) as ``.xlsx``.

Features are z-scored per feature over all patches of all slides.

    python datasets/tcga_crc/prepare.py --graph-dir graphs/ --survival TCGA-CDR.xlsx --endpoint DSS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sail.data import pooled_zscore, save_samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--survival", type=Path, required=True)
    ap.add_argument("--endpoint", default="DSS", help="TCGA-CDR endpoint: DSS, OS, PFI, DFI")
    ap.add_argument("--ext", default="g")
    ap.add_argument("--out", type=Path, default=Path("data/tcga_crc"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    surv = pd.read_excel(args.survival).set_index("bcr_patient_barcode")
    surv.index = [s[:12] for s in surv.index]

    ids, xs, coords, rows = [], [], [], []
    for path in sorted(args.graph_dir.glob(f"*.{args.ext}")):
        pid = path.stem[:12]
        if pid not in surv.index:
            continue
        event, time = surv.loc[pid, args.endpoint], surv.loc[pid, f"{args.endpoint}.time"]
        if isinstance(event, pd.Series):  # ambiguous barcode (several survival rows): skipped
            continue
        if pd.isna(event) or pd.isna(time):
            continue
        g = torch.load(path, weights_only=False)
        ids.append(path.stem)
        xs.append(np.asarray(g.x, dtype=np.float32))
        coords.append(np.asarray(g.coords, dtype=np.float32))
        rows.append(dict(id=path.stem, patient=pid, time=float(time), event=int(event)))

    save_samples(args.out / "samples.pt", ids, pooled_zscore(xs), coords, [f"f{j}" for j in range(xs[0].shape[1])])
    pd.DataFrame(rows).to_csv(args.out / "clinical.csv", index=False)
    print(f"{len(ids)} slides ({args.endpoint}) -> {args.out}")


if __name__ == "__main__":
    main()
