"""Prepare ORION-CRC (Lin et al., Nature Cancer 2023) for ``experiments/survival/orion_crc.yaml``.

Input
-----
* one segmented cell table per slide (``CRC01.db`` ... as tiatoolbox annotation
  stores with point geometries and one property per marker, or ``.csv`` /
  ``.parquet`` with ``x``, ``y`` and marker columns);
* the clinical supplementary table of the ORION paper
  (``43018_2023_576_MOESM2_ESM.xlsx``, sheet "S3 Patient Characteristics").

Processing
----------
* 17 markers: the autofluorescence (AF1) and Argo550 channels are dropped;
* each marker is scaled to [0, 1] between its cohort-wide 2nd and 98th
  percentiles (``marker_percentiles.csv``) and clipped;
* patients with < 90 days follow-up are removed, as is patient C15.

Output: ``samples.pt`` and ``clinical.csv`` (``id, time, event``; overall survival in days).

    python datasets/orion_crc/prepare.py --cells-dir /path/to/cells --clinical 43018_2023_576_MOESM2_ESM.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sail.data import save_samples

HERE = Path(__file__).resolve().parent
MARKERS = ["Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20", "PD-L1",
           "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"]
MIN_FOLLOW_UP_DAYS = 90
EXCLUDE = {"C15"}


def read_cells(path: Path) -> pd.DataFrame:
    if path.suffix == ".db":
        from tiatoolbox.annotation.storage import SQLiteStore

        df = SQLiteStore.open(path).to_dataframe()
        df["x"] = df["geometry"].apply(lambda g: g.x)
        df["y"] = df["geometry"].apply(lambda g: g.y)
        df.columns = [c.replace("properties.", "") for c in df.columns]
        return df.drop(columns=["geometry"])
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", type=Path, required=True)
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--percentiles", type=Path, default=HERE / "marker_percentiles.csv")
    ap.add_argument("--out", type=Path, default=Path("data/orion_crc"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clin = pd.read_excel(args.clinical, sheet_name="S3 Patient Characteristics", skiprows=2)
    clin = clin[["Specimen ID", "Death", "OSDays"]].dropna(subset=["OSDays"])
    clin.columns = ["id", "event", "time"]
    clin = clin[(clin["time"] >= MIN_FOLLOW_UP_DAYS) & ~clin["id"].isin(EXCLUDE)]

    pct = pd.read_csv(args.percentiles).set_index("protein").loc[MARKERS]
    lo, hi = pct["2%"].to_numpy(np.float32), pct["98%"].to_numpy(np.float32)

    ids, feats, coords = [], [], []
    files = sorted(f for f in args.cells_dir.iterdir()
                   if f.suffix in (".db", ".csv", ".parquet") and "-registered" not in f.name)
    for f in files:
        pid = f.stem.replace("CRC", "C")
        if pid not in set(clin["id"]):
            continue
        df = read_cells(f)
        ids.append(pid)
        feats.append(np.clip((df[MARKERS].to_numpy(np.float32) - lo) / (hi - lo), 0, 1))
        coords.append(df[["x", "y"]].to_numpy(np.float32))
        print(f"{pid}: {len(df):,} cells")

    save_samples(args.out / "samples.pt", ids, feats, coords, MARKERS)
    clin[clin["id"].isin(ids)][["id", "time", "event"]].to_csv(args.out / "clinical.csv", index=False)
    print(f"{len(ids)} patients -> {args.out}")


if __name__ == "__main__":
    main()
