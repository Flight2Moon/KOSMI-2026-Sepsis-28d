from __future__ import annotations

import unittest

from los_baseline.task_spec import (
    EXP1_FLAT_3CLASS_TASK,
    STAGE1_MORTALITY_TASK,
    STAGE2_LONG_STAY_TASK,
    TaskDefinition,
    get_task_definition,
)


class TaskDefinitionTests(unittest.TestCase):
    def test_frozen_task_contracts(self) -> None:
        self.assertEqual(EXP1_FLAT_3CLASS_TASK.expected_classes, (0, 1, 2))
        self.assertFalse(EXP1_FLAT_3CLASS_TASK.threshold_required)
        self.assertEqual(STAGE1_MORTALITY_TASK.expected_classes, (0, 1))
        self.assertTrue(STAGE1_MORTALITY_TASK.threshold_required)
        self.assertEqual(STAGE2_LONG_STAY_TASK.population, "survivors")

    def test_lookup_returns_frozen_definition(self) -> None:
        self.assertIs(
            get_task_definition("stage1_mortality"),
            STAGE1_MORTALITY_TASK,
        )

    def test_invalid_multiclass_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskDefinition(
                task_name="invalid",
                task_type="multiclass",
                population="all",
                num_classes=3,
                class_names=("a", "b", "c"),
                threshold_required=True,
            )


if __name__ == "__main__":
    unittest.main()
