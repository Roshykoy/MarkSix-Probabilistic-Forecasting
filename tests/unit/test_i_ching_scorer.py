import pytest
from src.i_ching_scorer import IChingScorer


def test_probabilities_sum_to_one():
    config = {"num_lotto_numbers": 10}
    draws = [
        [1, 2, 3],
        [3, 4, 5],
        [5, 5, 5],
    ]
    scorer = IChingScorer(config, draws, smoothing=1.0)
    assert pytest.approx(sum(scorer.probabilities.values())) == 1.0


def test_element_modifiers_change_rankings():
    # Numbers 1 and 5 appear once each; without modifiers they'd tie
    config = {"num_lotto_numbers": 5}
    draws = [[1], [5]]
    scorer = IChingScorer(config, draws, smoothing=0)
    assert scorer.frequencies[1] == scorer.frequencies[5]
    assert scorer.probabilities[5] > scorer.probabilities[1]
