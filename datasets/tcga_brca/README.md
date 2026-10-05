# TCGA-BRCA whole-slide images

Diagnostic H&E slides of the TCGA breast cancer cohort (GDC portal), tiled
into patches; each patch is a node with a 1,024-dimensional ShuffleNet feature
vector and its slide coordinates. Survival: disease-specific survival.

Input: an HDF5 file with `x/<i>`, `coords/<i>`, `patient_id`, `time`, `event`
(see `prepare.py`).

```bash
python datasets/tcga_brca/prepare.py --h5 tcga_graphs.h5
```
