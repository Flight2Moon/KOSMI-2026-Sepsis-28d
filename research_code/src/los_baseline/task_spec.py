from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TaskType = Literal["binary", "multiclass"]
PopulationType = Literal["all", "survivors"]


@dataclass(frozen=True)
class TaskDefinition:
    task_name: str
    task_type: TaskType
    population: PopulationType
    num_classes: int
    class_names: tuple[str, ...]
    threshold_required: bool
    positive_class: int | None = None

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty.")
        if self.task_type not in {"binary", "multiclass"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.population not in {"all", "survivors"}:
            raise ValueError(f"Unsupported population: {self.population}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names length must match num_classes: "
                f"{len(self.class_names)} != {self.num_classes}"
            )
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique.")
        if self.task_type == "binary":
            if self.num_classes != 2:
                raise ValueError("Binary tasks must define exactly 2 classes.")
            if not self.threshold_required:
                raise ValueError("Binary tasks must require a decision threshold.")
            if self.positive_class not in {0, 1}:
                raise ValueError("Binary tasks must define positive_class as 0 or 1.")
        else:
            if self.threshold_required:
                raise ValueError("Multiclass tasks must use argmax, not a threshold.")
            if self.positive_class is not None:
                raise ValueError("Multiclass tasks must not define positive_class.")

    @property
    def expected_classes(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXP1_FLAT_3CLASS_TASK = TaskDefinition(
    task_name="exp1_flat_3class",
    task_type="multiclass",
    population="all",
    num_classes=3,
    class_names=(
        "A_discharged_alive",
        "B_died",
        "C_still_admitted",
    ),
    threshold_required=False,
)

STAGE1_MORTALITY_TASK = TaskDefinition(
    task_name="stage1_mortality",
    task_type="binary",
    population="all",
    num_classes=2,
    class_names=("A_or_C_non_death", "B_died"),
    threshold_required=True,
    positive_class=1,
)

STAGE2_LONG_STAY_TASK = TaskDefinition(
    task_name="stage2_long_stay",
    task_type="binary",
    population="survivors",
    num_classes=2,
    class_names=("A_discharged_alive", "C_still_admitted"),
    threshold_required=True,
    positive_class=1,
)

TASK_DEFINITIONS = {
    task.task_name: task
    for task in (
        EXP1_FLAT_3CLASS_TASK,
        STAGE1_MORTALITY_TASK,
        STAGE2_LONG_STAY_TASK,
    )
}


def get_task_definition(task_name: str) -> TaskDefinition:
    try:
        return TASK_DEFINITIONS[task_name]
    except KeyError as exception:
        raise KeyError(
            f"Unknown EXP123 task: {task_name}. "
            f"Available: {sorted(TASK_DEFINITIONS)}"
        ) from exception
