"""
nlp_engine/__init__.py
Re-exports for backward compatibility with existing imports.
"""
from .entities import (
    Symptom,
    Medication,
    Vital,
    Diagnosis,
    ClinicalEntities,
)
from .core import MedicalNLPEngine

__all__ = [
    "Symptom",
    "Medication",
    "Vital",
    "Diagnosis",
    "ClinicalEntities",
    "MedicalNLPEngine",
]
