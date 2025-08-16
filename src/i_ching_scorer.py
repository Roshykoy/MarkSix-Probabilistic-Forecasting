from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence


class IChingScorer:
    """Compute traditional probabilities and score prediction sets.

    The scorer derives number frequencies from historical draws, applies the
    classic five‑element modifiers ``(i % 5)`` (with multiples of five mapped to
    ``5``), and normalizes the resulting weights with additive smoothing.  It
    can optionally combine temporal weights or external machine‑learning
    predictions before re‑normalising.  A small utility is provided to score
    candidate sets based on spread, sum balance, frequency and recent activity.
    """

    def __init__(
        self,
        config: Dict,
        historical_draws: Sequence[Iterable[int]],
        smoothing: float = 1.0,
    ) -> None:
        self.config = config
        self.num_numbers = config["num_lotto_numbers"]
        self.historical_draws = [list(draw) for draw in historical_draws]
        self.smoothing = smoothing

        # raw occurrence counts for each number
        self.frequencies: Dict[int, int] = self._compute_frequencies()
        # probabilities incorporating element modifiers
        self.probabilities: Dict[int, float] = self.calculate_probabilities()

    # ------------------------------------------------------------------
    # Probability calculations
    def _compute_frequencies(self) -> Dict[int, int]:
        counts = Counter()
        for draw in self.historical_draws:
            counts.update(draw)
        return {i: counts.get(i, 0) for i in range(1, self.num_numbers + 1)}

    def calculate_probabilities(self, smoothing: Optional[float] = None) -> Dict[int, float]:
        """Calculate probabilities with optional smoothing."""
        if smoothing is None:
            smoothing = self.smoothing

        weighted: Dict[int, float] = {}
        total = 0.0
        for i in range(1, self.num_numbers + 1):
            modifier = i % 5 or 5  # five‑element multiplier
            count = self.frequencies.get(i, 0)
            value = (count + smoothing) * modifier
            weighted[i] = value
            total += value

        self.probabilities = {i: v / total for i, v in weighted.items()}
        return self.probabilities

    # ------------------------------------------------------------------
    # Ensemble combination
    def ensemble_probabilities(
        self,
        temporal_weights: Optional[Dict[int, float]] = None,
        ml_predictions: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """Combine stored probabilities with optional temporal or ML weights."""
        combined = self.probabilities.copy()
        for i in combined.keys():
            if temporal_weights:
                combined[i] *= temporal_weights.get(i, 1.0)
            if ml_predictions:
                combined[i] *= ml_predictions.get(i, 1.0)

        total = sum(combined.values())
        return {i: v / total for i, v in combined.items()}

    # ------------------------------------------------------------------
    # Scoring utilities
    def score(self, number_set: Sequence[int]) -> float:
        """Simple score: average probability of the numbers."""
        return float(mean([self.probabilities[num] for num in number_set]))

    def score_prediction_sets(
        self,
        candidate_sets: Sequence[Sequence[int]],
        recent_draws: Optional[Sequence[Sequence[int]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """Evaluate candidate sets by spread, sum balance, frequency and recency."""
        if weights is None:
            weights = {"spread": 0.25, "sum": 0.25, "frequency": 0.25, "recent": 0.25}

        recent_numbers: set[int] = set()
        if recent_draws:
            for draw in recent_draws:
                recent_numbers.update(draw)

        scores: List[Dict] = []
        max_number = self.num_numbers
        for s in candidate_sets:
            s_sorted = sorted(s)
            k = len(s_sorted)
            expected_sum = (max_number + 1) / 2 * k

            spread_score = (s_sorted[-1] - s_sorted[0]) / max_number if k > 1 else 0.0
            sum_score = 1 - abs(sum(s_sorted) - expected_sum) / expected_sum
            freq_score = sum(self.probabilities[n] for n in s_sorted) / k
            if recent_numbers:
                overlap = len(set(s_sorted) & recent_numbers) / k
                recent_score = 1 - overlap
            else:
                recent_score = 1.0

            final = (
                weights["spread"] * spread_score
                + weights["sum"] * sum_score
                + weights["frequency"] * freq_score
                + weights["recent"] * recent_score
            )

            scores.append(
                {
                    "set": s_sorted,
                    "score": final,
                    "components": {
                        "spread": spread_score,
                        "sum": sum_score,
                        "frequency": freq_score,
                        "recent": recent_score,
                    },
                }
            )
        return scores
