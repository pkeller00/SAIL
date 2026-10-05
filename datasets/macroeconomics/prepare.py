"""Prepare the Global Macro Database panel for ``experiments/macroeconomics``.

Input
-----
* ``GMD_clean.xlsx`` (sheet ``data_clean``) written by ``clean_GMD.R`` from the
  Global Macro Database (Müller et al., 2025; https://www.globalmacrodata.com);
* Natural Earth 1:110m admin-0 countries shapefile (``ne_110m_admin_0_countries.shp``).

Processing
----------
* years ``--start`` to ``--end``; the 25 indicators below;
* balanced panel: countries with complete indicators in every year and a
  Natural Earth polygon (145 countries for 2000-2023);
* node position = representative point of the country polygon (lat, lon);
* each indicator is z-scored over all country-years of the panel.

Output: ``panel.csv`` with ``ISO3, countryname, year, lat, lon`` + indicators.

    python datasets/macroeconomics/prepare.py --gmd GMD_clean.xlsx --shapefile ne_110m_admin_0_countries.shp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURES = [
    "nGDP_USD", "rGDP_USD", "rGDP_pc_USD", "pop", "infl",
    "cons_GDP", "cons_USD", "hcons_GDP", "hcons_USD", "gcons_GDP", "gcons_USD",
    "inv_GDP", "inv_USD", "finv_GDP", "finv_USD",
    "exports_GDP", "exports_USD", "imports_GDP", "imports_USD", "CA_GDP", "CA_USD",
    "govexp_GDP", "govrev_GDP", "govdef_GDP", "govdebt_GDP",
]


def centroids(shapefile: Path) -> pd.DataFrame:
    import geopandas as gpd

    world = gpd.read_file(shapefile)
    world = world.set_crs(epsg=4326, allow_override=True) if world.crs is None else world.to_crs(epsg=4326)
    pts = world.geometry.representative_point()
    df = pd.DataFrame({"ISO3": world["iso_a3"].astype(str), "lat": pts.y, "lon": pts.x})
    df = df[(df["ISO3"].str.len() == 3) & (df["ISO3"] != "-99")]
    return df.drop_duplicates("ISO3", keep="first")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmd", type=Path, required=True)
    ap.add_argument("--shapefile", type=Path, required=True)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--out", type=Path, default=Path("data/macroeconomics"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.gmd, sheet_name="data_clean")
    df = df[(df["year"] >= args.start) & (df["year"] <= args.end)].copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    geo = centroids(args.shapefile)

    usable = df.dropna(subset=FEATURES)
    usable = usable[usable["ISO3"].isin(geo["ISO3"])]
    years_per_country = usable.groupby("ISO3")["year"].nunique()
    keep = years_per_country[years_per_country == usable["year"].nunique()].index
    df = df[df["ISO3"].isin(keep)].copy()
    df[FEATURES] = (df[FEATURES] - df[FEATURES].mean()) / (df[FEATURES].std() + 1e-9)

    panel = df[["ISO3", "countryname", "year", *FEATURES]].merge(geo, on="ISO3")
    panel = panel[["ISO3", "countryname", "year", "lat", "lon", *FEATURES]].sort_values(["year", "ISO3"])
    panel.to_csv(args.out / "panel.csv", index=False)
    print(f"{panel['ISO3'].nunique()} countries x {panel['year'].nunique()} years -> {args.out / 'panel.csv'}")


if __name__ == "__main__":
    main()
