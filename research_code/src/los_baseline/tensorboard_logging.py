from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NullSummaryWriter:
    def __getattr__(self, name: str):
        def no_operation(*args: Any, **kwargs: Any) -> None:
            return None
        return no_operation

    def close(self) -> None:
        return None

    def flush(self) -> None:
        return None


def create_summary_writer(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exception:
        print(
            "[WARNING] TensorBoard is unavailable. The experiment will "
            "continue with file-based logging only."
        )
        print(f"[WARNING] TensorBoard initialization error: {exception}")
        return NullSummaryWriter()


def log_metric_dictionary(
    writer: Any,
    split_name: str,
    threshold_view: str,
    metrics: dict[str, Any],
    step: int = 0,
) -> None:
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                writer.add_scalar(
                    f"{split_name}/{threshold_view}/{name}",
                    float(value),
                    step,
                )
            except Exception:
                continue


def log_figure_dictionary(
    writer: Any,
    split_name: str,
    figures: dict[str, Any],
    step: int = 0,
) -> None:
    for name, figure in figures.items():
        writer.add_figure(f"{split_name}/{name}", figure, step)


def log_json_payload(
    writer: Any,
    tag: str,
    payload: dict[str, Any],
    step: int = 0,
) -> None:
    writer.add_text(
        tag,
        "```json\n" + json.dumps(payload, indent=2, default=str) + "\n```",
        step,
    )


def log_metadata(writer: Any, metadata: dict[str, Any]) -> None:
    log_json_payload(writer, "experiment/metadata", metadata, step=0)
