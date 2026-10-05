#!/usr/bin/env bash
# Re-score the shipped paper models on the full datasets and compare with results/.
# Needs a GPU node with the data in data/ (see datasets/). Usage: bash scripts/reproduce_paper_checks.sh
set -euo pipefail
OUT=${OUT:-outputs/checks}
run() { echo "== $*"; sail-run "$@" --device cuda -v; }

run experiments/4i/config.yaml --stage eval --output-dir $OUT/4i                  # Supp. Table IV
run experiments/pd1_cscc/config.yaml --stage eval --output-dir $OUT/pd1_cscc      # Table II / Supp. Table VI
run experiments/pd1_cscc/config.yaml --stage biomarkers --output-dir $OUT/pd1_cscc
for c in orion_crc tcga_brca tcga_crc; do                                         # Table III, Supp. Tables VII-VIII
  run experiments/survival/$c.yaml --stage eval --output-dir $OUT/$c
done
run experiments/mesothelioma/config.yaml --stage eval --output-dir $OUT/mesothelioma  # Table IV (SAIL rows)
run experiments/macroeconomics/config.yaml --stage eval --output-dir $OUT/macroeconomics

python - <<'PY'
import pandas as pd
out = "outputs/checks"
def show(name, a, b, cols):
    print(f"\n{name}"); print(pd.concat({"paper": a[cols], "now": b[cols]}, axis=1).round(3).to_string())
show("4i best Dice per region",
     pd.read_csv("results/4i/sail_dice.csv").groupby("region")[["dice"]].max(),
     pd.read_csv(f"{out}/4i/sail_dice.csv").groupby("region")[["dice"]].max(), ["dice"])
for c in ["orion_crc", "tcga_brca", "tcga_crc"]:
    a = pd.read_csv(f"results/{c}/survival_all_modes.csv")
    b = pd.read_csv(f"{out}/{c}/survival_all_modes.csv")
    a["c"] = a["c-index"].map(lambda v: max(v, 1 - v)); b["c"] = b["c_index_oriented"]
    print(f"\n{c}: max |delta C-index| = {(a['c'].values - b['c'].values).__abs__().max():.4f}")
PY
