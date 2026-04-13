"""
================================================================================
ADVERSARIALLY RESILIENT DETECTION PIPELINE — UPSC-AI INTEGRATION
================================================================================
Ported from: Adversarially-resilient-detection-pipelines
Applied to:  UPSC-AI RAG System

Components:
    - drift_detectors   : ADWIN, Page-Hinkley, KS, MMD (same algorithms as SOC pipeline)
    - adversarial_detector : Text-based adversarial query detection
    - conformal_scorer  : Split conformal prediction on similarity scores
    - soc_monitor       : Central SOC state machine (STABLE → SUSPICIOUS → EVASION_LOCKED)
================================================================================
"""

from .drift_detectors import (
    ADWINDetector,
    PageHinkleyDetector,
    KSDetector,
    MMDDetector,
    ConceptDriftEngine,
)
from .adversarial_detector import AdversarialQueryDetector
from .conformal_scorer import ConformalScorer
from .soc_monitor import SOCMonitor, SOCState, SOCAlert

__all__ = [
    "ADWINDetector",
    "PageHinkleyDetector",
    "KSDetector",
    "MMDDetector",
    "ConceptDriftEngine",
    "AdversarialQueryDetector",
    "ConformalScorer",
    "SOCMonitor",
    "SOCState",
    "SOCAlert",
]
