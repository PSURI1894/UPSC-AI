"""
================================================================================
SOC MONITOR — CENTRAL PIPELINE ORCHESTRATOR
================================================================================
Mirrors the RiskThermostat + SOCDashboard from the main SOC pipeline,
adapted for the UPSC-AI RAG domain.

State machine (same FSM as the SOC pipeline):
    STABLE          → normal traffic, all detectors quiet
    SUSPICIOUS      → 1+ signals triggered (adversarial OR drift OR conformal)
    EVASION_LOCKED  → multiple signals or repeated adversarial queries
    FAILURE         → system error or critical breach

Signal sources (parallel to IDS pipeline):
    1. AdversarialQueryDetector  (text patterns, OOD, score anomaly)
    2. ConceptDriftEngine        (ADWIN + PH + KS + MMD consensus)
    3. ConformalScorer           (split-CP p-value on similarity scores)

The monitor auto-calibrates from the first N_CALIBRATION legitimate queries.
================================================================================
"""

import time
import uuid
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from .adversarial_detector import AdversarialQueryDetector, DetectionResult
from .conformal_scorer import ConformalScorer, ConformalResult
from .drift_detectors import ADWINDetector, PageHinkleyDetector


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class SOCState(Enum):
    STABLE         = "STABLE"
    SUSPICIOUS     = "SUSPICIOUS"
    EVASION_LOCKED = "EVASION_LOCKED"
    FAILURE        = "FAILURE"


_STATE_COLORS = {
    SOCState.STABLE:         "\033[92m",   # green
    SOCState.SUSPICIOUS:     "\033[93m",   # yellow
    SOCState.EVASION_LOCKED: "\033[91m",   # red
    SOCState.FAILURE:        "\033[95m",   # magenta
}
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class SOCAlert:
    alert_id: str
    timestamp: float
    state: SOCState
    query: str
    signals: List[str]                  # which detectors fired
    adversarial_result: Optional[DetectionResult]
    conformal_result: Optional[ConformalResult]
    drift_signals: Dict[str, bool]
    severity: float                     # 0.0 – 1.0
    action: str                         # recommended SOC action
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "query": self.query[:120],
            "signals": self.signals,
            "severity": round(self.severity, 3),
            "action": self.action,
            "adversarial": {
                "is_adversarial": self.adversarial_result.is_adversarial if self.adversarial_result else False,
                "attack_types": self.adversarial_result.attack_types if self.adversarial_result else [],
                "confidence": self.adversarial_result.confidence if self.adversarial_result else 0.0,
            },
            "conformal": {
                "max_similarity": self.conformal_result.max_similarity if self.conformal_result else None,
                "p_value": self.conformal_result.p_value if self.conformal_result else None,
                "is_anomalous": self.conformal_result.is_anomalous if self.conformal_result else False,
            },
            "drift": self.drift_signals,
        }


# ---------------------------------------------------------------------------
# SOC Monitor
# ---------------------------------------------------------------------------

class SOCMonitor:
    """
    Central orchestrator — wraps all pipeline detectors and manages the
    SOC state machine.

    Auto-calibration: the first `n_calibration` queries with similarity
    scores above `calibration_min_score` are used to build the conformal
    baseline. During this warm-up phase the state is always STABLE.
    """

    N_CALIBRATION = 10           # queries needed before full detection activates
    COOLDOWN_AFTER_LOCK = 3      # queries before EVASION_LOCKED → SUSPICIOUS

    def __init__(self, alpha: float = 0.10, logger: Optional[logging.Logger] = None):
        self.alpha = alpha
        self.logger = logger or logging.getLogger("SOCMonitor")

        # Sub-detectors
        self.adv_detector = AdversarialQueryDetector()
        self.conformal = ConformalScorer(alpha=alpha)
        self.adwin = ADWINDetector()
        self.ph = PageHinkleyDetector()

        # State machine
        self._state = SOCState.STABLE
        self._state_changed_at: float = time.time()
        self._lock_counter: int = 0          # queries since EVASION_LOCKED

        # Calibration
        self._calibration_scores: List[float] = []
        self._calibrated: bool = False

        # History
        self._alerts: deque = deque(maxlen=500)
        self._query_count: int = 0
        self._adversarial_count: int = 0
        self._drift_count: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        max_similarity: float,
        all_scores: Optional[List[float]] = None,
    ) -> SOCAlert:
        """
        Process one query through the full pipeline.

        Parameters
        ----------
        query          : raw user question
        max_similarity : highest similarity score returned by retriever
        all_scores     : all similarity scores across both books

        Returns
        -------
        SOCAlert with current state and all detector signals
        """
        self._query_count += 1
        signals: List[str] = []
        drift_signals: Dict[str, bool] = {}

        # ── Auto-calibration warm-up ────────────────────────────────
        if not self._calibrated:
            if max_similarity > 0.45:   # only learn from clearly-relevant queries
                self._calibration_scores.append(max_similarity)
            if len(self._calibration_scores) >= self.N_CALIBRATION:
                self.conformal.calibrate(self._calibration_scores)
                self._calibrated = True
                self.logger.info(
                    f"[SOCMonitor] Conformal calibration complete on "
                    f"{len(self._calibration_scores)} queries. "
                    f"Interval: [{self.conformal._lower:.3f}, {self.conformal._upper:.3f}]"
                )

        # ── 1. Adversarial query detection ───────────────────────────
        adv_result: DetectionResult = self.adv_detector.inspect(
            query, max_similarity=max_similarity, all_scores=all_scores
        )
        if adv_result.is_adversarial:
            signals.extend([f"ADV:{t}" for t in adv_result.attack_types])
            self._adversarial_count += 1

        # ── 2. Conformal scoring ─────────────────────────────────────
        conf_result: ConformalResult = self.conformal.score(max_similarity)
        if conf_result.is_anomalous and self._calibrated:
            signals.append(f"CONFORMAL:p={conf_result.p_value:.3f}")

        # ── 3. Streaming drift detectors ─────────────────────────────
        # Use (1 - max_similarity) as the "error" signal; high = bad retrieval
        error_signal = 1.0 - max_similarity
        adwin_drift = self.adwin.update(error_signal)
        ph_drift = self.ph.update(error_signal)

        drift_signals = {"adwin": adwin_drift, "page_hinkley": ph_drift}
        if adwin_drift or ph_drift:
            self._drift_count += 1
            signals.append(f"DRIFT:{'ADWIN' if adwin_drift else ''}{'PH' if ph_drift else ''}")

        # ── 4. State machine transition ──────────────────────────────
        new_state = self._transition(signals)

        # ── 5. Severity + action ─────────────────────────────────────
        severity = self._compute_severity(signals, adv_result, conf_result)
        action = self._recommend_action(new_state, adv_result)

        alert = SOCAlert(
            alert_id=uuid.uuid4().hex[:10],
            timestamp=time.time(),
            state=new_state,
            query=query,
            signals=signals,
            adversarial_result=adv_result,
            conformal_result=conf_result,
            drift_signals=drift_signals,
            severity=severity,
            action=action,
        )

        self._alerts.append(alert)

        if signals:
            self.logger.warning(
                f"[SOCMonitor] {new_state.value} | signals={signals} | "
                f"severity={severity:.2f} | query='{query[:60]}...'"
            )

        return alert

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @staticmethod
    def _has_pattern_attack(signals: List[str]) -> bool:
        """Only explicit injection/jailbreak patterns count as hard attacks."""
        return any(s.startswith("ADV:PROMPT") or s.startswith("ADV:JAIL") for s in signals)

    def _transition(self, signals: List[str]) -> SOCState:
        has_pattern = self._has_pattern_attack(signals)
        n_signals = len(signals)

        if self._state == SOCState.EVASION_LOCKED:
            self._lock_counter += 1
            # Unlock after cooldown queries with no explicit pattern attacks
            # (score anomalies / OOD alone do not reset the cooldown)
            if self._lock_counter >= self.COOLDOWN_AFTER_LOCK and not has_pattern:
                self._state = SOCState.SUSPICIOUS
                self._lock_counter = 0
            return self._state

        if n_signals == 0:
            if self._state == SOCState.SUSPICIOUS:
                self._state = SOCState.STABLE
            return self._state

        # EVASION_LOCKED only on explicit pattern attacks (injection / jailbreak)
        # Score anomaly + OOD alone → SUSPICIOUS, never EVASION_LOCKED
        if has_pattern:
            self._state = SOCState.EVASION_LOCKED
            self._lock_counter = 0
        elif n_signals >= 1:
            self._state = SOCState.SUSPICIOUS

        return self._state

    # ------------------------------------------------------------------
    # Severity + action
    # ------------------------------------------------------------------

    def _compute_severity(
        self,
        signals: List[str],
        adv: DetectionResult,
        conf: ConformalResult,
    ) -> float:
        if not signals:
            return 0.0

        base = min(0.3 * len(signals), 0.7)
        base += adv.confidence * 0.2
        if conf.is_anomalous and self._calibrated:
            base += (1.0 - conf.p_value) * 0.1
        return min(round(base, 3), 1.0)

    def _recommend_action(self, state: SOCState, adv: DetectionResult) -> str:
        if state == SOCState.STABLE:
            return "PASS"
        if state == SOCState.SUSPICIOUS:
            return "LOG_AND_MONITOR"
        if state == SOCState.EVASION_LOCKED:
            if "PROMPT_INJECTION" in adv.attack_types or "JAILBREAK" in adv.attack_types:
                return "BLOCK_AND_ALERT"
            return "RATE_LIMIT"
        return "ESCALATE"

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------

    @property
    def state(self) -> SOCState:
        return self._state

    def get_status(self) -> Dict:
        recent = list(self._alerts)[-20:]
        return {
            "state": self._state.value,
            "query_count": self._query_count,
            "adversarial_count": self._adversarial_count,
            "drift_count": self._drift_count,
            "calibrated": self._calibrated,
            "calibration_progress": f"{len(self._calibration_scores)}/{self.N_CALIBRATION}",
            "score_baseline": self.adv_detector.get_score_baseline(),
            "conformal_interval": {
                "lower": round(self.conformal._lower, 4) if self._calibrated else None,
                "upper": round(self.conformal._upper, 4) if self._calibrated else None,
            },
            "recent_alerts": [a.to_dict() for a in recent if a.signals],
        }

    def get_full_report(self) -> str:
        """Generate a SOC-style terminal report."""
        color = _STATE_COLORS.get(self._state, "")
        lines = [
            "",
            "=" * 70,
            f"  ADVERSARIALLY RESILIENT PIPELINE — SOC REPORT",
            "=" * 70,
            f"  Current State : {color}{self._state.value}{_RESET}",
            f"  Total Queries : {self._query_count}",
            f"  Adversarial   : {self._adversarial_count}",
            f"  Drift Events  : {self._drift_count}",
            f"  Calibrated    : {self._calibrated}",
        ]

        if self._calibrated:
            lines += [
                f"  CP Interval   : [{self.conformal._lower:.3f}, {self.conformal._upper:.3f}]",
            ]

        baseline = self.adv_detector.get_score_baseline()
        if baseline["n"] > 0:
            lines += [
                f"  Score Baseline: mean={baseline['mean']:.3f}  std={baseline['std']:.3f}",
            ]

        # Last 5 alerts
        recent_alerts = [a for a in list(self._alerts)[-10:] if a.signals]
        if recent_alerts:
            lines += ["", "  RECENT ALERTS:", "  " + "-" * 66]
            for a in recent_alerts[-5:]:
                c = _STATE_COLORS.get(a.state, "")
                lines.append(
                    f"  {c}[{a.state.value}]{_RESET} "
                    f"signals={a.signals}  sev={a.severity:.2f}  "
                    f"action={a.action}"
                )
                lines.append(f"    query: \"{a.query[:65]}\"")

        lines += ["=" * 70, ""]
        return "\n".join(lines)

    def reset(self):
        """Reset state machine (keep calibration)."""
        self._state = SOCState.STABLE
        self._lock_counter = 0
        self._alerts.clear()
        self._query_count = 0
        self._adversarial_count = 0
        self._drift_count = 0
