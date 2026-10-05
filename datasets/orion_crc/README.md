# ORION-CRC (multiplexed immunofluorescence)

Whole-slide 18-plex immunofluorescence of colorectal cancer (Lin et al.,
*Nature Cancer* 2023). Clinical data: supplementary table
`43018_2023_576_MOESM2_ESM.xlsx`, sheet "S3 Patient Characteristics".

The public ORION release provides per-cell marker intensities for every slide.

Input: one cell table per slide (tiatoolbox annotation store `.db`
with point geometries, or `.csv`/`.parquet` with `x`, `y` and one column per
marker). `marker_percentiles.csv` holds the cohort-wide 2nd/98th percentiles
used to scale each marker.

```bash
python datasets/orion_crc/prepare.py --cells-dir <cells> --clinical 43018_2023_576_MOESM2_ESM.xlsx
```

writes `data/orion_crc/samples.pt` (17 markers per cell) and `clinical.csv`
(overall survival in days).
