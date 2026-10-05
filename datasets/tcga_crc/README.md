# TCGA-CRC whole-slide images

Diagnostic H&E slides of TCGA-COAD and TCGA-READ, tiled into patches with
UNI features (Chen et al., *Nature Medicine* 2024). Survival from the TCGA
Clinical Data Resource (Liu et al., *Cell* 2018).

```bash
python datasets/tcga_crc/prepare.py --graph-dir <graphs> --survival TCGA-CDR-SupplementalTableS1.xlsx --endpoint DSS
```
