"""How the package reads and checks what it is given."""

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from scipy.stats import entropy

import incoherence as ic

RNG = np.random.default_rng(0)
TRIALS = [RNG.normal(i, 1, 200) for i in range(4)]


def test_a_list_of_trials_and_an_array_give_the_same_answer():
    assert ic.incoherence(np.stack(TRIALS)) == ic.incoherence(TRIALS)
    assert ic.incoherence([list(trial) for trial in TRIALS]) == ic.incoherence(TRIALS)


def test_trials_can_hold_different_numbers_of_observations():
    ragged = [TRIALS[0][:50], TRIALS[1], TRIALS[2][:120]]
    assert 0 < ic.incoherence(ragged) < 1


def test_shortcuts_match_the_result():
    result = ic.measure(TRIALS)
    assert isinstance(result, ic.Result)
    assert ic.incoherence(TRIALS) == result.incoherence
    assert ic.cohesion(TRIALS) == result.cohesion
    assert len(result.entropies) == 4
    assert repr(result) == f"Result(trials=4, incoherence={result.incoherence:.3f})"


def test_cohesion_is_only_worked_out_when_read():
    result = ic.measure(TRIALS)
    assert result.incoherence > 0
    assert "divergences" not in vars(result)
    assert result.cohesion > 0
    assert "divergences" in vars(result)


def test_bins_count_observations_then_treat_them_as_discrete():
    edges = np.linspace(-4, 8, 13)
    counts = [np.histogram(trial, edges)[0] for trial in TRIALS]
    binned, direct = ic.measure(TRIALS, bins=edges), ic.measure_counts(counts)
    assert binned.incoherence == direct.incoherence
    assert binned.cohesion == direct.cohesion
    assert binned.max_entropy == pytest.approx(np.log2(12))


def test_bins_accept_a_number_and_a_rule():
    assert ic.measure(TRIALS, bins=8).max_entropy == pytest.approx(3.0)
    assert 0 < ic.incoherence(TRIALS, bins="auto") < 1


def test_counts_and_probabilities_give_the_same_answer():
    counts = np.array([[5, 3, 2], [1, 1, 8], [4, 4, 2]])
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    assert ic.measure_counts(probabilities).incoherence == pytest.approx(
        ic.measure_counts(counts).incoherence
    )


def test_discrete_entropies_are_in_bits():
    counts = np.array([[5, 3, 2], [1, 1, 8]])
    result = ic.measure_counts(counts)
    assert result.entropies == pytest.approx(entropy(counts, base=2, axis=1))
    assert result.pooled_entropy == pytest.approx(entropy([0.3, 0.2, 0.5], base=2))


def test_divergences_cover_every_pair():
    result = ic.measure(TRIALS)
    assert result.divergences.shape == (6,)
    matrix = squareform(result.divergences)
    assert matrix[0, 3] == result.divergences[2]
    assert matrix[0, 3] > matrix[0, 1]


def test_weights_follow_the_trial_they_belong_to():
    counts = np.array([[9, 1], [1, 9], [5, 5]])
    weights = [1, 2, 3]
    reordered = ic.measure_counts(counts[::-1], weights=weights[::-1])
    assert reordered.incoherence == pytest.approx(
        ic.measure_counts(counts, weights=weights).incoherence
    )
    assert ic.measure_counts(counts, weights=[7, 7, 7]).incoherence == pytest.approx(
        ic.measure_counts(counts).incoherence
    )
    assert ic.measure_counts(counts, weights=weights).incoherence != pytest.approx(
        ic.measure_counts(counts).incoherence
    )


def test_bounds_fix_the_scale():
    narrow = [RNG.uniform(0.4, 0.5, 300), RNG.uniform(0.5, 0.6, 300)]
    assert ic.incoherence(narrow, bounds=(0, 1)) < ic.incoherence(narrow)


def test_every_observation_the_same_is_coherent():
    assert ic.incoherence([[3.0, 3.0], [3.0, 3.0, 3.0]]) == 0.0
    assert ic.measure_counts([[4], [7]]).incoherence == 0.0


@pytest.mark.parametrize(
    ("trials", "kwargs", "message"),
    [
        ([[1.0, 2.0]], {}, "at least two trials"),
        ([[1.0, 2.0], []], {}, "at least one observation"),
        ([[1.0, np.nan], [1.0, 2.0]], {}, "finite"),
        ([[1.0, np.inf], [1.0, 2.0]], {}, "finite"),
        (np.arange(6.0), {}, "two dimensions"),
        ([[[1.0, 2.0]], [[3.0, 4.0]]], {}, "flat list"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (0, 3)}, "outside bounds"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (5, 1)}, "low below high"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1]}, "one value per trial"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1, 0]}, "positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": 0}, "k must be positive"),
    ],
)
def test_bad_observations_are_rejected(trials, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ic.measure(trials, **kwargs)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([[1, 2]], "at least two trials"),
        ([1, 2, 3], "one row per trial"),
        ([[1, -2], [3, 4]], "not negative"),
        ([[1, np.nan], [3, 4]], "finite"),
        ([[0, 0], [3, 4]], "at least one observation"),
    ],
)
def test_bad_counts_are_rejected(counts, message):
    with pytest.raises(ValueError, match=message):
        ic.measure_counts(counts)
