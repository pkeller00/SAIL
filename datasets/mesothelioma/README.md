# Mesothelioma TMA cell graphs (MesoGraph)

Per-core cell graphs of the St George's Hospital mesothelioma TMA cohort
released with MesoGraph (Eastwood et al., *Cell Reports Medicine* 2023):
https://github.com/measty/MesoGraph (data link in its README).

Place the per-core graphs at `data/mesothelioma/st_george_hospital/graphs/*.pkl`.
The MesoGraph baseline additionally needs the MesoGraph code:

```bash
git clone https://github.com/measty/MesoGraph third_party/MesoGraph
```

The leave-one-slide-out folds used in the paper are in
`experiments/mesothelioma/splits.json`.
