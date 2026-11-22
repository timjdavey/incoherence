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


@pytest.mark.parametrize(
    ("k", "points"), [(1, 501), (10, 501), (25, 501), (25.01, 502), (100, 2001), (1000, 20001)]
)
def test_the_grid_grows_with_k(k, points):
    assert _entropy.points_for(k) == points


def test_cohesion_uses_the_grid_for_its_own_k():
    assert _measures.COHESION_POINTS == 2001


@pytest.mark.parametrize("k", [10, 25, 100, 1000])
def test_a_finer_grid_barely_moves_incoherence(k):
    """The grid is a tool for the sum, so the answer should rest on ``k`` and not on the grid."""
    rng = np.random.default_rng(1)
    trials = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
    low, high = np.min(trials), np.max(trials)
    weights = np.full(10, 0.1)

    def incoherence(points):
        rows = np.stack([_entropy.densities((t - low) / (high - low), k, points) for t in trials])
        entropies = _entropy.density_entropy(rows, k)
        pooled = float(_entropy.density_entropy(_measures.pooled(rows, weights), k))
        return _measures.incoherence(entropies, pooled, 1.0, weights)

    points = _entropy.points_for(k)
    assert incoherence(points) == pytest.approx(ic.incoherence(trials, k=k))
    assert incoherence(points) == pytest.approx(incoherence(4 * (points - 1) + 1), rel=0.01)


@pytest.mark.filterwarnings("error")
def test_the_highest_k_runs_without_a_warning():
    _entropy._variance_limits.cache_clear()  # or an earlier test may have used up the warning
    rng = np.random.default_rng(0)
    result = ic.measure([rng.normal(i, 1, 200) for i in range(4)], k=1000)
    assert 0 < result.incoherence < 1
    assert 0 <= result.cohesion <= 1


def test_divergences_are_worked_out_in_blocks_at_a_high_k():
    rows = np.random.default_rng(0).dirichlet(np.ones(20_001), 16)
    a, b = np.triu_indices(16, k=1)
    assert _entropy._BLOCK // rows.shape[1] < len(a)
    expected = [jensenshannon(rows[i], rows[j], base=2) ** 2 for i, j in zip(a, b, strict=True)]
    assert ic.measure_counts(rows).divergences == pytest.approx(expected, abs=1e-9)


def test_evenly_spread_divergences_have_no_cohesion():
    assert _measures.cohesion(np.linspace(0, 1, 2000)) == pytest.approx(0.0, abs=1e-6)
