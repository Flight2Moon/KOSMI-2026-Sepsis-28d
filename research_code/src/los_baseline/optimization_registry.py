from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationRegistry(AbstractContextManager["OptimizationRegistry"]):
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS studies (
                study_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                seed INTEGER NOT NULL,
                shared_split_hash TEXT,
                objective_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trials (
                study_id TEXT NOT NULL,
                search_stage TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                tunable_parameters_json TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                fold_metrics_json TEXT,
                aggregate_metrics_json TEXT,
                selection_score REAL,
                duration_seconds REAL,
                failure_type TEXT,
                failure_message TEXT,
                traceback_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (study_id, search_stage, trial_number),
                FOREIGN KEY (study_id) REFERENCES studies(study_id)
            );

            CREATE TABLE IF NOT EXISTS validation_candidates (
                study_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                source_stage TEXT NOT NULL,
                source_trial_number INTEGER NOT NULL,
                model_config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                complexity_score REAL NOT NULL,
                eligible INTEGER NOT NULL,
                selected INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (study_id, candidate_id),
                FOREIGN KEY (study_id) REFERENCES studies(study_id)
            );
            """
        )
        self.connection.commit()

    def register_study(
        self,
        study_id: str,
        task_name: str,
        model_name: str,
        seed: int,
        shared_split_hash: str | None,
        objective: Mapping[str, Any],
    ) -> None:
        now = _utc_now()
        self.connection.execute(
            """
            INSERT INTO studies (
                study_id, task_name, model_name, seed, shared_split_hash,
                objective_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_id) DO UPDATE SET
                task_name=excluded.task_name,
                model_name=excluded.model_name,
                seed=excluded.seed,
                shared_split_hash=excluded.shared_split_hash,
                objective_json=excluded.objective_json,
                updated_at=excluded.updated_at
            """,
            (
                study_id,
                task_name,
                model_name,
                int(seed),
                shared_split_hash,
                json.dumps(dict(objective), sort_keys=True),
                now,
                now,
            ),
        )
        self.connection.commit()

    def trial_exists(
        self,
        study_id: str,
        search_stage: str,
        trial_number: int,
        completed_only: bool = True,
    ) -> bool:
        query = (
            "SELECT status FROM trials WHERE study_id=? AND search_stage=? "
            "AND trial_number=?"
        )
        row = self.connection.execute(
            query, (study_id, search_stage, int(trial_number))
        ).fetchone()
        if row is None:
            return False
        return row["status"] == "completed" if completed_only else True

    def upsert_trial(
        self,
        *,
        study_id: str,
        search_stage: str,
        trial_number: int,
        status: str,
        tunable_parameters: Mapping[str, Any],
        model_config: Mapping[str, Any],
        fold_metrics: list[Mapping[str, Any]] | None = None,
        aggregate_metrics: Mapping[str, Any] | None = None,
        selection_score: float | None = None,
        duration_seconds: float | None = None,
        failure_type: str | None = None,
        failure_message: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        now = _utc_now()
        self.connection.execute(
            """
            INSERT INTO trials (
                study_id, search_stage, trial_number, status,
                tunable_parameters_json, model_config_json,
                fold_metrics_json, aggregate_metrics_json, selection_score,
                duration_seconds, failure_type, failure_message, traceback_text,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_id, search_stage, trial_number) DO UPDATE SET
                status=excluded.status,
                tunable_parameters_json=excluded.tunable_parameters_json,
                model_config_json=excluded.model_config_json,
                fold_metrics_json=excluded.fold_metrics_json,
                aggregate_metrics_json=excluded.aggregate_metrics_json,
                selection_score=excluded.selection_score,
                duration_seconds=excluded.duration_seconds,
                failure_type=excluded.failure_type,
                failure_message=excluded.failure_message,
                traceback_text=excluded.traceback_text,
                updated_at=excluded.updated_at
            """,
            (
                study_id,
                search_stage,
                int(trial_number),
                status,
                json.dumps(dict(tunable_parameters), sort_keys=True, default=str),
                json.dumps(dict(model_config), sort_keys=True, default=str),
                json.dumps(fold_metrics, sort_keys=True, default=str)
                if fold_metrics is not None
                else None,
                json.dumps(dict(aggregate_metrics), sort_keys=True, default=str)
                if aggregate_metrics is not None
                else None,
                selection_score,
                duration_seconds,
                failure_type,
                failure_message,
                traceback_text,
                now,
                now,
            ),
        )
        self.connection.commit()

    def completed_trials(
        self,
        study_id: str,
        search_stages: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = [study_id]
        query = "SELECT * FROM trials WHERE study_id=? AND status='completed'"
        stages = tuple(search_stages or ())
        if stages:
            placeholders = ",".join("?" for _ in stages)
            query += f" AND search_stage IN ({placeholders})"
            parameters.extend(stages)
        query += " ORDER BY selection_score DESC, search_stage, trial_number"
        rows = self.connection.execute(query, parameters).fetchall()
        return [self._decode_trial(row) for row in rows]

    def all_trials(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM trials WHERE study_id=? "
            "ORDER BY search_stage, trial_number",
            (study_id,),
        ).fetchall()
        return [self._decode_trial(row) for row in rows]

    def _decode_trial(self, row: sqlite3.Row) -> dict[str, Any]:
        output = dict(row)
        for key in (
            "tunable_parameters_json",
            "model_config_json",
            "fold_metrics_json",
            "aggregate_metrics_json",
        ):
            value = output.pop(key)
            output[key.removesuffix("_json")] = json.loads(value) if value else None
        return output

    def replace_validation_candidates(
        self,
        study_id: str,
        candidates: Iterable[Mapping[str, Any]],
    ) -> None:
        now = _utc_now()
        self.connection.execute(
            "DELETE FROM validation_candidates WHERE study_id=?", (study_id,)
        )
        for candidate in candidates:
            self.connection.execute(
                """
                INSERT INTO validation_candidates (
                    study_id, candidate_id, source_stage, source_trial_number,
                    model_config_json, metrics_json, complexity_score,
                    eligible, selected, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id,
                    str(candidate["candidate_id"]),
                    str(candidate["source_stage"]),
                    int(candidate["source_trial_number"]),
                    json.dumps(candidate["model_config"], sort_keys=True, default=str),
                    json.dumps(candidate["metrics"], sort_keys=True, default=str),
                    float(candidate.get("complexity_score", 0.0)),
                    int(bool(candidate.get("eligible_for_selection", False))),
                    int(bool(candidate.get("selected", False))),
                    now,
                    now,
                ),
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
