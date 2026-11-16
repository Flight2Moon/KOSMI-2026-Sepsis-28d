"""Imbalance strategies for EXP123.

The package uses lazy exports because training_engine imports the lightweight
sampling contract, while the legacy undersampling module imports optimization
helpers that themselves import training_engine. Eager package imports would
therefore create a circular dependency during CLI startup.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NaturalSampler": (".sampling", "NaturalSampler"),
    "SampledTrainingData": (".sampling", "SampledTrainingData"),
    "TrainingSampler": (".sampling", "TrainingSampler"),
    "assert_natural_split": (".sampling", "assert_natural_split"),
    "sampling_manifest_payload": (".sampling", "sampling_manifest_payload"),
    "synthetic_mask": (".sampling", "synthetic_mask"),
    "ClassCSmoteSampler": (".oversampling", "ClassCSmoteSampler"),
    "SyntheticPoolManifest": (".oversampling", "SyntheticPoolManifest"),
    "build_class_c_synthetic_pool": (".oversampling", "build_class_c_synthetic_pool"),
    "load_synthetic_pool": (".oversampling", "load_synthetic_pool"),
    "save_synthetic_pool": (".oversampling", "save_synthetic_pool"),
    "MajorityUndersamplingSampler": (".undersampling", "MajorityUndersamplingSampler"),
    "UndersamplingPlan": (".undersampling", "UndersamplingPlan"),
    "majority_undersample_indices": (".undersampling", "majority_undersample_indices"),
    "run_undersampling_search": (".undersampling", "run_undersampling_search"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
