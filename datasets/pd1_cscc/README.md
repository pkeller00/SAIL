# cSCC PD-1/PD-L1 blockade (CosMx)

Single-cell spatial transcriptomics (NanoString CosMx, 1,000 genes) of
cutaneous squamous cell carcinoma biopsies from patients treated with PD-1 /
PD-L1 blockade (Lee et al., *J. Immunother. Cancer* 2026; data:
Dryad, doi:10.5061/dryad.s4mw6m9jh). The labelled AnnData object used here
(cells with response labels and niche annotations) was provided by the
authors of the original study.

Place the labelled AnnData object at `data/pd1_cscc/pd1_cSCC_labelled.h5ad`.
It must contain raw counts in `X` and the `obs` columns

| column | content |
|---|---|
| `Subject_STUDY_ID` | patient |
| `TMA_ID` | tissue core |
| `Biopsy_Timepoint` | `Baseline` (pre-treatment), ... |
| `response` | `Responder` / `NonResponder` |
| `SpatialClusterNames` | niche annotation of the original study |
| `Baseline_PDL1` | clinical PD-L1 TPS |
| `x_slide_mm`, `y_slide_mm` | cell coordinates (if `obsm["spatial"]` is absent) |

Quality control (genes in >= 10 cells, cells with >= 100 counts, cores with
>= 50 cells) and feature construction are done by the experiment.
