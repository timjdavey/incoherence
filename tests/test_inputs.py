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


def test_weights_reach_binned_trials():
    edges = np.linspace(-4, 8, 13)
    counts = [np.histogram(trial, edges)[0] for trial in TRIALS]
    weights = [1, 2, 3, 4]
    assert (
        ic.measure(TRIALS, bins=edges, weights=weights).incoherence
        == ic.measure_counts(counts, weights=weights).incoherence
    )


def test_bounds_set_the_range_of_the_bins():
    trials = [[0.1, 0.2], [0.2, 0.9]]
    assert ic.incoherence(trials, bins=2) > 0
    assert ic.incoherence(trials, bins=2, bounds=(0, 2)) == 0.0


@pytest.mark.parametrize("k", [50, 100, 250, 1000])
def test_high_k_does_not_depend_on_where_values_fall_on_the_grid(k):
    on_grid = ic.incoherence([np.full(20, 0.2), np.full(20, 0.6)], bounds=(0, 1), k=k)
    off_grid = ic.incoherence([np.full(20, 0.2011), np.full(20, 0.6013)], bounds=(0, 1), k=k)
    assert off_grid == pytest.approx(on_grid, abs=0.01)


def test_values_on_the_outer_bin_edges_are_counted():
    result = ic.measure([[0.0, 1.0, 2.0], [2.0, 2.0, 0.0]], bins=[0, 1, 2])
    assert result.incoherence == ic.measure_counts([[1, 2], [1, 2]]).incoherence


def test_k_is_ignored_when_binning():
    assert ic.incoherence(TRIALS, bins=8, k=-1) == ic.incoherence(TRIALS, bins=8)


def test_trials_can_come_from_a_generator():
    assert ic.incoherence(trial for trial in TRIALS) == ic.incoherence(TRIALS)


def test_inputs_are_not_changed():
    table = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([1.0, 3.0])
    table_before, weights_before = table.copy(), weights.copy()

    assert ic.measure(table, weights=weights).cohesion >= 0
    assert ic.measure_counts(table, weights=weights).cohesion >= 0

    assert np.array_equal(table, table_before)
    assert np.array_equal(weights, weights_before)


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
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (2, 5)}, "outside bounds"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (5, 1)}, "low below high"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (1, 1)}, "low below high"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1]}, "one value per trial"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [[1, 2]]}, "one value per trial"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1, 0]}, "positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1, -1]}, "positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1, np.nan]}, "finite"),
        ([[1.0, 2.0], [3.0, 4.0]], {"weights": [1, np.inf]}, "finite"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": 0}, "k must be positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": -1}, "k must be positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": np.nan}, "k must be positive"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": np.inf}, "at most 1000"),
        ([[1.0, 2.0], [3.0, 4.0]], {"k": 1000.5}, "at most 1000"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (0, np.inf)}, "bounds must be finite"),
        ([[1.0, 2.0], [3.0, 4.0]], {"bounds": (-np.inf, np.inf)}, "bounds must be finite"),
        ([[0.5, 0.5, 100.0], [0.5, 0.5, 0.5]], {"bins": [0, 1, 2]}, "outside the bin edges"),
        ([[0.5, 0.5, -0.1], [0.5, 0.5, 0.5]], {"bins": [0, 1, 2]}, "outside the bin edges"),
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
        ([[1, np.inf], [3, 4]], "finite"),
        ([[0, 0], [3, 4]], "at least one observation"),
    ],
)
def test_bad_counts_are_rejected(counts, message):
    with pytest.raises(ValueError, match=message):
        ic.measure_counts(counts)


def known_bug(reason):
    """Mark a confirmed bug. The test fails once the bug is fixed, so the mark gets removed."""
    return pytest.mark.xfail(
        strict=True, reason=reason, raises=(AssertionError, pytest.fail.Exception)
    )


SIMPLE = [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]


@known_bug("the entropy limits collapse for a vanishingly small k")
def test_a_tiny_k_gives_a_value_in_range():
    assert 0 <= ic.incoherence(SIMPLE, k=1e-300) <= 1


@known_bug("the sum of the weights overflows")
def test_huge_discrete_weights_match_equal_weights():
    counts = [[9, 1], [1, 9]]
    assert ic.measure_counts(counts, weights=[1e308, 1e308]).incoherence == pytest.approx(
        ic.measure_counts(counts).incoherence
    )


@known_bug("the sum of the weights overflows")
def test_huge_continuous_weights_match_equal_weights():
    assert ic.measure(SIMPLE, weights=[1e308, 1e308]).incoherence == pytest.approx(
        ic.measure(SIMPLE).incoherence
    )


@known_bug("the row totals overflow")
def test_huge_counts_match_their_proportions():
    assert ic.measure_counts([[1e308, 1e308], [1, 9]]).incoherence == pytest.approx(
        ic.measure_counts([[1, 1], [1, 9]]).incoherence
    )


@known_bug("the width of the range overflows")
def test_a_huge_range_matches_the_same_data_scaled_down():
    assert ic.incoherence([[-1e308, 0.0], [1e308, 0.0]]) == pytest.approx(
        ic.incoherence([[-1.0, 0.0], [1.0, 0.0]])
    )


@known_bug("cached arrays are writable, so an edit changes later measures")
@pytest.mark.parametrize("name", ["entropies", "divergences"])
def test_cached_arrays_are_read_only(name):
    values = getattr(ic.measure_counts([[9, 1], [1, 9]]), name)
    with pytest.raises(ValueError):
        values[:] = 0


@known_bug("numpy.asarray strips the mask, so hidden values are counted")
def test_masked_observations_are_rejected():
    masked = np.ma.array([0.0, 1.0, 999.0], mask=[False, False, True])
    with pytest.raises(ValueError):
        ic.measure([masked, np.array([0.0, 1.0, 2.0])])


@known_bug("only the first two values of bounds are read")
def test_bounds_of_the_wrong_length_are_rejected():
    with pytest.raises(ValueError):
        ic.measure(SIMPLE, bounds=(0, 1, 99))
