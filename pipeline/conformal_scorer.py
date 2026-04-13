"""
================================================================================
CONFORMAL UNCERTAINTY SCORER
================================================================================
Ported from: src/risk_management_engine.py (ConformalEngine — SPLIT backend)

Applies split conformal prediction to RAG retrieval scores.
Given a calibration set of known-good similarity scores, it computes a
nonconformity score at inference time and flags queries whose scores fall
outside the (1-alpha) prediction interval.

In the SOC pipeline this wraps RSCP / online CP. Here we use SPLIT CP —
the lightest backend — because RAG retrieval scores are scalar and we want
a fast, explainable confidence bound.

Usage:
    scorer = ConformalScorer(alpha=0.10)
    scorer.calibrate(normal_scores)           # list of float, calibration set
    result = scorer.score(query_max_sim)      # at inference
================================================================================
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ConformalResult:
    max_similarity: float
    nonconformity: float        # how "strange" this score is vs calibration
    p_value: float              # fraction of calibration scores MORE nonconforming
    is_anomalous: bool          # True if p_value < alpha
    lower_bound: float          # prediction interval lower bound
    upper_bound: float          # prediction interval upper bound
    calibration_n: int


class ConformalScorer:
    """
    Split Conformal Prediction scorer for RAG similarity scores.

    Calibration: store nonconformity scores (1 - sim) from normal queries.
    Inference:   compute nonconformity for new query; flag if p-value < alpha.

    A low p-value means: "this query's retrieval similarity is more unusual
    than alpha% of normal queries" → likely adversarial or OOD.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.calibrated = False
        self._cal_nonconformity: Optional[np.ndarray] = None
        self._lower: float = 0.0
        self._upper: float = 1.0

    # ------------------------------------------------------------------
    # Calibration (call once with normal-traffic scores)
    # ------------------------------------------------------------------

    def calibrate(self, normal_scores: List[float]) -> None:
        """
        Calibrate on normal-traffic similarity scores.

        Parameters
        ----------
        normal_scores : list of max_similarity values from legitimate queries
        """
        if len(normal_scores) < 5:
            raise ValueError("Need at least 5 calibration samples.")

        arr = np.array(normal_scores, dtype=float)
        # Nonconformity = how far below the typical range a score falls
        self._cal_nonconformity = 1.0 - arr          # high nc = low sim = unusual

        # Prediction interval: (alpha/2, 1-alpha/2) quantiles of calibration scores
        self._lower = float(np.quantile(arr, self.alpha / 2))
        self._upper = float(np.quantile(arr, 1.0 - self.alpha / 2))
        self.calibrated = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, max_similarity: float) -> ConformalResult:
        """
        Score a single query by its max retrieval similarity.

        Returns a ConformalResult with anomaly flag and p-value.
        """
        if not self.calibrated:
            # Return a neutral result before calibration
            return ConformalResult(
                max_similarity=max_similarity,
                nonconformity=1.0 - max_similarity,
                p_value=1.0,
                is_anomalous=False,
                lower_bound=0.0,
                upper_bound=1.0,
                calibration_n=0,
            )

        nc = 1.0 - max_similarity
        # p-value = fraction of calibration nonconformities >= nc
        p_value = float(np.mean(self._cal_nonconformity >= nc))

        return ConformalResult(
            max_similarity=max_similarity,
            nonconformity=nc,
            p_value=p_value,
            is_anomalous=p_value < self.alpha,
            lower_bound=self._lower,
            upper_bound=self._upper,
            calibration_n=len(self._cal_nonconformity),
        )

    # ------------------------------------------------------------------
    # Batch calibration update (online-style)
    # ------------------------------------------------------------------

    def update_calibration(self, new_scores: List[float]) -> None:
        """
        Extend calibration set with newly observed normal scores.
        Recomputes the prediction interval.
        """
        if not self.calibrated:
            self.calibrate(new_scores)
            return

        existing = 1.0 - self._cal_nonconformity
        combined = np.concatenate([existing, new_scores])
        self.calibrate(combined.tolist())
