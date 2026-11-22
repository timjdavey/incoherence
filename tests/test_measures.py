"""The equations, checked against calculations written out separately."""

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

import incoherence as ic
from incoherence import _entropy, _measures


def bits(distribution):
    return entropy(distribution, base=2, axis=-1)


def test_weighted_incoherence_matches_equation_9():
    rows = np.array([[0.5, 0.5], [1.0, 0.0]])
    weights = np.array([0.75, 0.25])
    pooled = bits(weights @ rows)
    expected = np.sqrt(weights @ (pooled - bits(rows)) ** 2 / (pooled * 1.0))

    result = ic.measure_counts(rows, weights=[3, 1])

    assert result.pooled_entropy == pytest.approx(pooled)
    assert result.incoherence == pytest.approx(expected)


def test_a_weight_of_two_counts_like_a_repeated_trial():
    counts = np.array([[9, 1, 0], [1, 5, 4], [2, 2, 6]])
    repeated = ic.measure_counts(counts[[0, 0, 1, 2]])
    assert ic.measure_counts(counts, weights=[2, 1, 1]).incoherence == pytest.approx(
        repeated.incoherence
    )


def test_a_weight_of_two_counts_like_a_repeated_continuous_trial():
    rng = np.random.default_rng(0)
    trials = [rng.normal(i, 1, 200) for i in range(3)]
    repeated = ic.measure([trials[0], *trials])
    assert ic.measure(trials, weights=[2, 1, 1]).incoherence == pytest.approx(repeated.incoherence)


def test_divergences_match_scipy():
    rows = np.random.default_rng(0).dirichlet(np.ones(5), 8)
    a, b = np.triu_indices(8, k=1)
    expected = [jensenshannon(rows[i], rows[j], base=2) ** 2 for i, j in zip(a, b, strict=True)]
    assert ic.measure_counts(rows).divergences == pytest.approx(expected, abs=1e-12)


def test_weighted_divergences_match_the_formula():
    rows = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1]])
    weights = np.array([1.0, 2.0, 5.0])
    expected = []
    for i, j in zip(*np.triu_indices(3, k=1), strict=True):
        wi = weights[i] / (weights[i] + weights[j])
        wj = 1 - wi
        expected.append(bits(wi * rows[i] + wj * rows[j]) - wi * bits(rows[i]) - wj * bits(rows[j]))
    assert ic.measure_counts(rows, weights=weights).divergences == pytest.approx(expected)


def test_divergences_are_right_past_the_first_chunk():
    trials = 100
    rows = np.random.default_rng(0).dirichlet(np.ones(5), trials)
    a, b = np.triu_indices(trials, k=1)
    assert len(a) > _measures._CHUNK
    expected = [jensenshannon(rows[i], rows[j], base=2) ** 2 for i, j in zip(a, b, strict=True)]
    assert ic.measure_counts(rows).divergences == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(("k", "points"), [(1.0, 501), (10.0, 501), (100.0, 2001)])
def test_entropy_limits_match_their_definitions(k, points):
    v0, v1 = _entropy._variance_limits(k, 1.0, points)
    even = _entropy.densities(np.linspace(0, 1, 200_001), k, points)
    single = _entropy.densities(np.zeros(1), k, points)
    assert v0 == pytest.approx(even.var(), rel=1e-4)
    assert v1 == pytest.approx(single.var())


def test_evenly_spread_divergences_have_no_cohesion():
    assert _measures.cohesion(np.linspace(0, 1, 2000)) == pytest.approx(0.0, abs=1e-6)
