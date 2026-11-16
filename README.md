# KOSMI 2026 · P1-07 — Sepsis-3 28-Day Hospital Outcomes

**Prediction of 28-Day Hospital Outcomes in Patients Meeting Sepsis-3 Criteria: A Comparison of Multiclass and Two-Stage Models**

This repository is being prepared to share the reproducible **analysis code**, **aggregate evaluation results**, and **supplementary research website** for the 2026 Korean Society of Medical Informatics Fall Conference poster (P1-07).

## Study overview

The study compares direct multiclass prediction with death-first two-stage prediction of three hospital outcomes in adults identified in Sepsis-3-related ICU cohorts. Evaluation includes MIMIC-IV development/temporal data and external ICU datasets.

## Code and results

- `src/`, `analysis/`, `scripts/`, `configs/`: reviewed source code and reproducibility instructions (to be added following source audit).
- `results/`: aggregate performance tables including all 25 model-structure comparisons (to be added).
- `docs/`: public poster companion page for GitHub Pages (to be added).

**Release status:** Source files and supplementary materials are being reviewed before public upload. This repository currently does not include the raw reproducibility ZIP or patient-level datasets. No unverified data-generation pipeline is claimed to reproduce the restricted source cohorts.

## Data access and privacy

MIMIC-IV and eICU-CRD have access restrictions governed by PhysioNet. Their raw records or patient-level derivative datasets **must not be redistributed** in this public repository. Users needing those resources should obtain their own authorized access from their respective providers.

Aggregate tables and code may be published after checking provenance, contributor permissions, external licenses, and embedded credentials.

## Methods and interpretations

Comparisons use macro AUROC and patient-level paired bootstrap confidence intervals. Nominal significance should not be interpreted as proof of equivalence, and results are not adjusted for multiplicity unless explicitly labeled.

## Citation and reuse

Please cite the conference poster and original ICU database publications when reusing these methods. A source-code license will be added after confirming authorship and contributor approval.

---

**Conference:** KOSMI 2026 Fall Conference · **Poster:** P1-07
