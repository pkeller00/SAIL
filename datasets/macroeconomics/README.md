# Global Macro Database

Annual macroeconomic panel (Müller, Xu, Lehbib and Chen, Global Macro
Database, 2025; https://www.globalmacrodata.com) and Natural Earth 1:110m
admin-0 country boundaries (https://www.naturalearthdata.com).

```bash
Rscript datasets/macroeconomics/clean_GMD.R GMD.xlsx data/macroeconomics/GMD_clean.xlsx
python datasets/macroeconomics/prepare.py --gmd data/macroeconomics/GMD_clean.xlsx \
    --shapefile ne_110m_admin_0_countries.shp
```

`clean_GMD.R` removes forecast flags, crisis dummies and local-currency
levels; `prepare.py` builds the balanced 2000-2023 panel (145 countries,
25 indicators, z-scored) in `data/macroeconomics/panel.csv`. Copy the
shapefile (all its sidecar files) to `data/macroeconomics/` for the maps.
