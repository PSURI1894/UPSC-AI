"""
================================================================================
ADVERSARIAL QUERY DETECTOR
================================================================================
Detects adversarial and malicious queries targeting the UPSC-AI RAG system.

Attack taxonomy (mirrors the SOC pipeline's attack library):
    1. PROMPT_INJECTION  — attempts to override system instructions
    2. JAILBREAK         — tries to remove content restrictions
    3. OOD_QUERY         — out-of-distribution topic (not history-related)
    4. SCORE_ANOMALY     — retrieval scores statistically inconsistent with
                           normal traffic (too low = OOD, too high = possible
                           embedding inversion attack)
    5. REPETITION_FLOOD  — same or near-identical query sent repeatedly
================================================================================
"""

import re
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


# ---------------------------------------------------------------------------
# Attack patterns
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS = [
    r"ignore (your|all|previous|the) (instructions?|rules?|constraints?|system prompt)",
    r"disregard (your|all|previous) (instructions?|rules?)",
    r"forget (everything|all|your instructions?)",
    r"you are now",
    r"new persona",
    r"act as (if )?you (are|were|have no)",
    r"override (your|all) (settings?|instructions?|constraints?)",
    r"system prompt",
    r"<\s*system\s*>",
    r"\[system\]",
    r"###\s*instruction",
    r"do not follow",
    r"bypass (your|all)",
]

_JAILBREAK_PATTERNS = [
    r"without (any|your) restrictions?",
    r"no (ethical|moral|content) (guidelines?|filters?|restrictions?)",
    r"pretend you (are|have no|don't have)",
    r"roleplay as",
    r"DAN mode",
    r"developer mode",
    r"jailbreak",
    r"unrestricted (mode|ai|assistant)",
    r"hypothetically (speaking|if you could)",
    r"in a fictional world where",
    r"imagine you are an ai that",
]

_OOD_TOPIC_PATTERNS = [
    # Non-history topics unrelated to UPSC curriculum
    r"\b(cryptocurrency|bitcoin|ethereum|nft|blockchain)\b",
    r"\b(machine learning|neural network|deep learning|llm|gpt|chatgpt)\b",
    r"\b(recipe|cooking|food|restaurant)\b",
    r"\b(movie|film|bollywood|celebrity|actor|actress)\b",
    r"\b(cricket|ipl|football|sports team)\b",
    r"\b(stock market|share price|sensex|nifty)\b",
    r"\b(hack|exploit|malware|ransomware|vulnerability)\b",
    r"\b(password|credentials|login|phishing)\b",
]

_HISTORY_INDICATORS = [
    r"\b(dynasty|emperor|king|queen|ruler|kingdom|empire)\b",
    r"\b(century|bc|ad|ancient|medieval|colonial|independence)\b",
    r"\b(india|mughal|british|partition|harappan|vedic|maurya)\b",
    r"\b(gandhi|nehru|patel|ambedkar|bose|tilak)\b",
    r"\b(upsc|ias|prelims|mains|history|civilization)\b",
    r"\b(war|battle|treaty|revolution|movement|struggle)\b",
    r"\b(religion|culture|art|architecture|economy|trade)\b",
]


@dataclass
class DetectionResult:
    is_adversarial: bool
    attack_types: List[str]
    confidence: float           # 0.0 = clean, 1.0 = certain attack
    score_anomaly: bool
    is_ood: bool
    is_repeat: bool
    details: Dict[str, object] = field(default_factory=dict)


class AdversarialQueryDetector:
    """
    Stateful adversarial query detector for the UPSC-AI RAG system.

    Maintains a short-term query history to detect repetition floods and
    tracks retrieval score baselines to flag statistical anomalies.
    """

    def __init__(
        self,
        score_low_threshold: float = 0.42,
        score_high_threshold: float = 0.95,
        repeat_window: int = 50,
        repeat_max: int = 3,
    ):
        self.score_low_threshold = score_low_threshold
        self.score_high_threshold = score_high_threshold
        self.repeat_window = repeat_window
        self.repeat_max = repeat_max

        # Rolling history for repeat detection
        self._recent_hashes: deque = deque(maxlen=repeat_window)
        self._score_history: deque = deque(maxlen=200)

        # Compiled regex for speed
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS]
        self._jailbreak_re = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]
        self._ood_re = [re.compile(p, re.IGNORECASE) for p in _OOD_TOPIC_PATTERNS]
        self._history_re = [re.compile(p, re.IGNORECASE) for p in _HISTORY_INDICATORS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(
        self,
        query: str,
        max_similarity: Optional[float] = None,
        all_scores: Optional[List[float]] = None,
    ) -> DetectionResult:
        """
        Inspect a single query and its retrieval scores.

        Parameters
        ----------
        query          : raw user query string
        max_similarity : highest similarity score across all retrieved docs
        all_scores     : list of all similarity scores for this query
        """
        attack_types: List[str] = []
        details: Dict[str, object] = {}

        # 1. Pattern matching
        if self._matches_any(query, self._injection_re):
            attack_types.append("PROMPT_INJECTION")
            details["injection_matched"] = True

        if self._matches_any(query, self._jailbreak_re):
            attack_types.append("JAILBREAK")
            details["jailbreak_matched"] = True

        # 2. OOD detection — topic outside history scope
        is_ood = self._is_ood(query)
        if is_ood:
            attack_types.append("OOD_QUERY")

        # 3. Retrieval score anomaly
        score_anomaly = False
        if max_similarity is not None:
            self._score_history.append(max_similarity)
            if max_similarity < self.score_low_threshold:
                score_anomaly = True
                attack_types.append("SCORE_ANOMALY_LOW")
                details["max_similarity"] = max_similarity
            elif max_similarity > self.score_high_threshold:
                score_anomaly = True
                attack_types.append("SCORE_ANOMALY_HIGH")
                details["max_similarity"] = max_similarity

        # 4. Repetition flood detection
        q_hash = self._hash(query)
        repeat_count = sum(1 for h in self._recent_hashes if h == q_hash)
        is_repeat = repeat_count >= self.repeat_max
        if is_repeat:
            attack_types.append("REPETITION_FLOOD")
            details["repeat_count"] = repeat_count + 1
        self._recent_hashes.append(q_hash)

        # Aggregate confidence
        confidence = self._compute_confidence(attack_types, max_similarity)

        return DetectionResult(
            is_adversarial=len(attack_types) > 0,
            attack_types=attack_types,
            confidence=confidence,
            score_anomaly=score_anomaly,
            is_ood=is_ood,
            is_repeat=is_repeat,
            details=details,
        )

    def get_score_baseline(self) -> Dict[str, float]:
        """Return rolling statistics on similarity scores seen so far."""
        if not self._score_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        arr = np.array(self._score_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matches_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    def _is_ood(self, query: str) -> bool:
        has_ood_signal = self._matches_any(query, self._ood_re)
        has_history_signal = self._matches_any(query, self._history_re)

        if has_ood_signal:
            return True
        # Very short queries with no history keywords are suspicious
        if len(query.split()) < 4 and not has_history_signal:
            return False  # too short to classify confidently
        return False

    def _compute_confidence(self, attack_types: List[str], max_sim: Optional[float]) -> float:
        if not attack_types:
            return 0.0

        base = min(0.4 * len(attack_types), 0.8)

        # Score-based boost
        if max_sim is not None and max_sim < self.score_low_threshold:
            gap = self.score_low_threshold - max_sim
            base = min(base + gap * 2, 1.0)

        # Hard patterns = higher confidence
        if "PROMPT_INJECTION" in attack_types or "JAILBREAK" in attack_types:
            base = min(base + 0.2, 1.0)

        return round(base, 3)

    @staticmethod
    def _hash(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
