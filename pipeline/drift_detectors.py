"""
================================================================================
CONCEPT DRIFT DETECTION ENGINE — Ported from SOC Pipeline
================================================================================
Same algorithms used in the Adversarially Resilient SOC Pipeline,
now applied to monitor query/retrieval score distributions in the UPSC-AI
RAG system.

Algorithms:
    - ADWIN (Adaptive Windowing)         : streaming mean shift
    - Page-Hinkley                       : sequential change-point
    - Kolmogorov-Smirnov (KS)            : per-feature distributional shift
    - Maximum Mean Discrepancy (MMD)     : RBF kernel two-sample test

Consensus rule: drift triggered when >= 2 detectors agree.
================================================================================
"""

import numpy as np
from scipy import stats
from collections import deque
from typing import List, Dict, Optional


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) for drift detection.
    Monitors the running mean of a scalar stream (e.g. similarity scores).
    """

    def __init__(self, delta: float = 0.002, window_size: int = 200):
        self.delta = delta
        self.window_size = window_size
        self.stream: deque = deque(maxlen=window_size)

    def update(self, v: float) -> bool:
        self.stream.append(v)
        if len(self.stream) < 10:
            return False

        data = list(self.stream)
        n = len(data)
        step = max(1, n // 10)

        for cut in range(step, n - step + 1, step):
            w1, w2 = data[:cut], data[cut:]
            n1, n2 = len(w1), len(w2)
            mu1, mu2 = np.mean(w1), np.mean(w2)
            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            epsilon = np.sqrt(0.5 / m * np.log(4.0 / self.delta))

            if abs(mu1 - mu2) > epsilon:
                for _ in range(cut):
                    if self.stream:
                        self.stream.popleft()
                return True

        return False

    def reset(self):
        self.stream.clear()


class PageHinkleyDetector:
    """
    Page-Hinkley test — detects upward shifts in error/score streams.
    Applied here to monitor the no-answer rate or low-similarity rate.
    """

    def __init__(self, delta: float = 0.005, lambda_threshold: float = 20, alpha: float = 0.9999):
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.sum = 0.0
        self.n = 0

    def update(self, x: float) -> bool:
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        if self.sum > self.lambda_threshold:
            self.reset()
            return True
        return False

    def reset(self):
        self.sum = 0.0
        self.x_mean = 0.0
        self.n = 0


class KSDetector:
    """
    Kolmogorov-Smirnov test — detects distributional shift in feature vectors.
    Applied here to query embedding distributions.
    """

    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        self.reference_data = reference_data
        self.alpha = alpha
        self.n_features = reference_data.shape[1]
        self._last_drifted_features: List[int] = []

    def detect(self, current_data: np.ndarray) -> bool:
        corrected_alpha = self.alpha / self.n_features
        drifted = []

        for i in range(self.n_features):
            _, p_val = stats.ks_2samp(self.reference_data[:, i], current_data[:, i])
            if p_val < corrected_alpha:
                drifted.append(i)

        self._last_drifted_features = drifted
        return len(drifted) > 0

    @property
    def drifted_features(self) -> List[int]:
        return self._last_drifted_features


class MMDDetector:
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.
    Detects when the current query embedding distribution diverges
    from the calibration (normal-traffic) reference.
    """

    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05, bandwidth: Optional[float] = None):
        self.reference_data = reference_data
        self.alpha = alpha

        if bandwidth is None:
            subset = reference_data[: min(200, len(reference_data))]
            dists = np.linalg.norm(subset[:, None] - subset[None, :], axis=-1)
            self.bandwidth = float(np.median(dists[dists > 0])) + 1e-8
        else:
            self.bandwidth = bandwidth

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        sq_X = np.sum(X ** 2, axis=1, keepdims=True)
        sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        dists_sq = sq_X + sq_Y.T - 2 * X @ Y.T
        return np.exp(-dists_sq / (2 * self.bandwidth ** 2))

    def _kernel_mean(self, X: np.ndarray, Y: np.ndarray) -> float:
        K = self._rbf_kernel(X, Y)
        if np.array_equal(X, Y):
            np.fill_diagonal(K, 0)
            n = K.shape[0]
            return float(K.sum() / (n * (n - 1))) if n > 1 else 0.0
        return float(K.mean())

    def detect(self, current_data: np.ndarray) -> bool:
        max_n = 200
        ref = self.reference_data[:max_n]
        curr = current_data[:max_n]

        mmd_sq = (
            self._kernel_mean(ref, ref)
            + self._kernel_mean(curr, curr)
            - 2 * self._kernel_mean(ref, curr)
        )

        combined = np.vstack([ref, curr])
        n_ref = len(ref)
        null_mmds = np.zeros(50)

        for p in range(50):
            perm = np.random.permutation(len(combined))
            px, py = combined[perm[:n_ref]], combined[perm[n_ref:]]
            null_mmds[p] = (
                self._kernel_mean(px, px)
                + self._kernel_mean(py, py)
                - 2 * self._kernel_mean(px, py)
            )

        threshold = np.percentile(null_mmds, (1 - self.alpha) * 100)
        return float(mmd_sq) > float(threshold)


class ConceptDriftEngine:
    """
    Multi-signal consensus drift engine — same architecture as the SOC pipeline.
    Triggers alert only when >= consensus_threshold detectors agree.

    Applied here to:
        - current_batch     : query embedding matrix (n_queries, embed_dim)
        - prediction_errors : 1 - max_similarity per query (high = suspicious)
    """

    def __init__(self, reference_features: np.ndarray, consensus_threshold: int = 2):
        self.adwin = ADWINDetector()
        self.ph = PageHinkleyDetector()
        self.ks = KSDetector(reference_features)
        self.mmd = MMDDetector(reference_features)
        self.consensus_threshold = consensus_threshold
        self._last_results: Dict[str, bool] = {}

    def evaluate(self, current_batch: np.ndarray, prediction_errors: List[float]) -> bool:
        adwin_hits = sum(1 for e in prediction_errors if self.adwin.update(e))
        ph_hits = sum(1 for e in prediction_errors if self.ph.update(e))

        adwin_drift = adwin_hits > 0
        ph_drift = ph_hits > 0
        ks_drift = self.ks.detect(current_batch)
        mmd_drift = self.mmd.detect(current_batch)

        self._last_results = {
            "adwin": adwin_drift,
            "page_hinkley": ph_drift,
            "ks": ks_drift,
            "mmd": mmd_drift,
        }

        votes = sum(self._last_results.values())
        return votes >= self.consensus_threshold

    @property
    def last_results(self) -> Dict[str, bool]:
        return dict(self._last_results)
