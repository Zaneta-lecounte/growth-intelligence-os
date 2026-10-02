"""Hypothesis Evidence Validator (diagnosis layer).

Source of truth: specs/hypothesis-evidence-validator.md
"""
from gios.modules.hypothesis_evidence_validator.analysis import MODULE
from gios.modules.hypothesis_evidence_validator.pipeline import ValidationResult, save, scored_hypothesis, validate

__all__ = ["MODULE", "ValidationResult", "save", "scored_hypothesis", "validate"]
