"""Study-specific modules for the professor-provided 28-day Sepsis-3 study.

The Study28 package starts from frozen analysis-ready CSV files, never recreates
or edits labels, and separates development/model-freeze work from external
validation and later adaptation/sensitivity analyses.
"""

from .audit import AuditResult, assert_required_external_sites, audit_study_datasets
from .contract import LABEL_TO_Y, PROHIBITED_MODEL_COLUMNS, Y_TO_LABEL
from .datasets import DatasetRegistry, DatasetSpec, load_dataset_registry
from .external_protocol import Study28ExternalProtocol, load_external_protocol
from .protocol import Study28Protocol, load_study28_protocol

__all__ = [
    "LABEL_TO_Y",
    "Y_TO_LABEL",
    "PROHIBITED_MODEL_COLUMNS",
    "DatasetRegistry",
    "DatasetSpec",
    "load_dataset_registry",
    "AuditResult",
    "audit_study_datasets",
    "assert_required_external_sites",
    "Study28Protocol",
    "load_study28_protocol",
    "Study28ExternalProtocol",
    "load_external_protocol",
]
