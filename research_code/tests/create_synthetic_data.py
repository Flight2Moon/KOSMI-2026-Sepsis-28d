from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from los_baseline.constants import DEFAULT_FEATURE_NAMES


def create_dataset(path: Path, rows: int, seed: int, shift: float) -> None:
    rng = np.random.default_rng(seed)
    los_groups = np.tile(np.array([4.0, 10.0, 20.0, 35.0]), rows // 4 + 1)[:rows]
    los = np.clip(los_groups + rng.normal(0, 1.2, rows), 0.5, None)
    mortality = np.zeros(rows, dtype=np.int64)
    early_death_candidates = np.flatnonzero(los <= 28.0)
    late_death_candidates = np.flatnonzero(los > 28.0)
    mortality[early_death_candidates[::7]] = 1
    mortality[late_death_candidates[::11]] = 1
    admission_time = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        np.arange(rows) * 3, unit="h"
    )
    discharge_time = admission_time + pd.to_timedelta(los, unit="D")
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(rows),
            "hadm_id": np.arange(rows) + 100000,
            "stay_id": np.arange(rows) + 200000,
            "admittime": admission_time,
            "dischtime": discharge_time,
            "gender": rng.choice(["M", "F"], rows),
            "age": rng.normal(65 + shift, 14, rows),
            "sofa_score": rng.integers(2, 16, rows),
            "charlson_comorbidity_index": rng.integers(0, 10, rows),
            "aps3": rng.integers(5, 120, rows),
            "heart_rate_mean": rng.normal(85 + shift, 15, rows),
            "sbp_mean": rng.normal(115, 18, rows),
            "dbp_mean": rng.normal(65, 10, rows),
            "resp_rate_mean": rng.normal(20, 5, rows),
            "temperature_mean": rng.normal(37, 0.8, rows),
            "hemoglobin_mean": rng.normal(11.5, 2, rows),
            "platelets_mean": rng.normal(220, 70, rows),
            "wbc_mean": rng.normal(12, 4, rows),
            "bun_mean": rng.normal(28 + shift, 12, rows),
            "creatinine_mean": np.clip(rng.normal(1.4, 0.6, rows), 0.2, None),
            "glucose_mean": rng.normal(140, 35, rows),
            "chloride_mean": rng.normal(103, 4, rows),
            "sodium_mean": rng.normal(138, 3, rows),
            "potassium_mean": rng.normal(4.1, 0.5, rows),
            "LOS": los,
            "hospital_expire_flag": mortality,
        }
    )
    frame.loc[rng.choice(rows, max(1, rows // 40), replace=False), "bun_mean"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="infer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()
    output = Path(args.output_dir)
    create_dataset(output / "mimic.csv", args.rows, 42, 0.0)
    eicu_path = output / "eicu.csv.gz"
    create_dataset(eicu_path, args.rows, 43, 2.0)


if __name__ == "__main__":
    main()
