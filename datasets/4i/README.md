# 4i spatial proteomics

Iterative indirect immunofluorescence imaging (4i) of 43 protein markers,
with expert annotation of 10 subcellular compartments (Gut, Treier and
Pelkmans, *Science* 2018), as distributed by squidpy:

```python
import squidpy as sq
adata = sq.datasets.four_i()     # downloaded and cached on first use
```

No preparation is needed; `experiments/4i/config.yaml` loads it directly
(set `data.path` to a local `.h5ad` to avoid the download). Compartment labels
(`obs["cluster"]`) are used for evaluation only.
