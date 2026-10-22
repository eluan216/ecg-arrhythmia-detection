"""Evaluation protocols for ECG arrhythmia detection."""
from src.eval.inter_patient import run_inter_patient_rf, save_patient_split

__all__ = ["run_inter_patient_rf", "save_patient_split"]
