# Prediction of 28-Day Hospital Outcomes in Sepsis-3 Cohorts

**KOSMI 2026 Fall Conference — Poster P1-07**

## Study overview

This study compares direct multiclass classification with a death-first
two-stage approach for predicting 28-day hospital outcomes in ICU patients.

Evaluation includes MIMIC-IV temporal data and external ICU cohorts
from eICU-CRD, SICdb, and NWICU.

## Prediction approaches

- **Multiclass:** Direct prediction of three hospital outcomes.
- **Two-stage:** Prediction of death followed by conditional prediction
  of prolonged hospitalization among non-death cases.

## Main findings

Across five algorithms and five evaluation cohorts, 25 paired comparisons
were performed. Nominally significant advantages were observed in
10 multiclass comparisons and one two-stage comparison.

In 23 comparisons, the absolute difference in macro AUROC was below 0.02.

These findings do not establish equivalence between prediction structures.

## Project contents

- `docs/`: GitHub Pages research website and supplementary materials.
- `research_code/`: Selected public analysis and evaluation source code.

## Data availability and reproducibility

This release does not redistribute patient-level records, raw EHR data,
trained model binaries, or restricted clinical datasets.

Source datasets must be obtained from their respective providers under
the applicable access and data-use requirements.

The available release does not independently reproduce every original
raw-to-final Sepsis-3 cohort extraction step. Some scripts require
separately authorized datasets and local configuration.

## Research website

https://flight2moon.github.io/KOSMI-2026-Sepsis-28d/

## Citation

KOSMI 2026 Fall Conference, Poster P1-07.

Please also cite the original MIMIC-IV, eICU-CRD, SICdb, and NWICU sources.

## Release notice

Source code is provided for research transparency. No license for
third-party clinical data is granted. Software licensing and contributor
permissions should be confirmed before public release.
