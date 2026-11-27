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


def test_cohesion_k_is_8_for_each_trial():
    assert _measures.cohesion_k(10) == 80
    assert ic.measure_counts(np.eye(5)).cohesion_k == 40
    assert ic.measure(np.eye(3)).cohesion_k == 24


def test_cohesion_k_can_be_set():
    counts = np.array([[9, 1, 0], [1, 5, 4], [2, 2, 6], [5, 5, 0]])
    result = ic.measure_counts(counts, cohesion_k=100)
    assert result.cohesion_k == 100
    assert result.cohesion == _measures.cohesion(result.divergences, 100)
    assert result.cohesion != ic.measure_counts(counts).cohesion


def test_binned_data_passes_cohesion_k_on():
    trials = [[0, 1, 1], [1, 2, 2], [0, 0, 2]]
    binned = ic.measure(trials, bins=[0, 1, 2, 3], cohesion_k=100)
    assert binned.cohesion_k == 100
    assert ic.cohesion(trials, bins=[0, 1, 2, 3], cohesion_k=100) == binned.cohesion


def test_cohesion_k_uses_the_grid_it_needs():
    divergences = np.random.default_rng(1).random(45)
    for k in (24, 100, 3200):
        expected = _measures.cohesion(divergences, k, points=_entropy.points_for(k))
        assert _measures.cohesion(divergences, k) == expected


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


@pytest.mark.parametrize("trials", [3, 10, 100, 400])
def test_a_finer_grid_barely_moves_cohesion(trials):
    """Cohesion's ``k`` grows with trials, so its grid has to grow with it."""
    rows = np.random.default_rng(trials).dirichlet(np.ones(4), trials)
    result = ic.measure_counts(rows)
    k = result.cohesion_k
    finer = 4 * (_entropy.points_for(k) - 1) + 1
    fine = _measures.cohesion(result.divergences, k, points=finer)
    assert result.cohesion == pytest.approx(fine, abs=0.005)


def test_the_grid_cache_stays_small():
    """Every number of trials has its own grid, and only the last two are kept."""
    rng = np.random.default_rng(0)
    for trials in range(3, 41):
        assert 0 <= ic.measure_counts(rng.dirichlet(np.ones(4), trials)).cohesion <= 1
    assert _entropy.grid.cache_info().currsize <= 2


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
    assert _measures._BLOCK // rows.shape[1] < len(a)
    expected = [jensenshannon(rows[i], rows[j], base=2) ** 2 for i, j in zip(a, b, strict=True)]
    assert ic.measure_counts(rows).divergences == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("k", "points"),
    [(1e-4, 501), (1.0, 501), (10.0, 501), (25.0, 501), (100.0, 2001), (1000.0, 20001)],
)
def test_densities_match_the_direct_sum(k, points):
    r = _entropy.grid(points)
    on_the_grid = r[(r >= 0) & (r <= 1)][::37]
    values = np.concatenate([np.random.default_rng(0).random(300), [0.0, 1.0], on_the_grid])
    direct = np.exp(-k * np.abs(r[:, None] - values[None, :])).mean(axis=1)
    assert _entropy.densities(values, k, points) == pytest.approx(direct, rel=1e-9)


def test_densities_match_the_direct_sum_for_many_observations():
    r = _entropy.grid(501)
    values = np.random.default_rng(1).random(5000)
    direct = np.exp(-10.0 * np.abs(r[:, None] - values[None, :])).mean(axis=1)
    assert _entropy.densities(values, 10.0, 501) == pytest.approx(direct, rel=1e-9)


def test_a_trial_with_almost_no_weight_barely_moves_cohesion():
    counts = np.array([[9, 1, 0, 0], [8, 2, 0, 0], [0, 0, 1, 9], [0, 0, 2, 8], [3, 3, 2, 2]])
    weighted = ic.measure_counts(counts, weights=[1, 1, 1, 1, 1e-12])
    plain = ic.measure_counts(counts[:4])
    assert weighted.cohesion_k == pytest.approx(plain.cohesion_k, rel=1e-9)
    # A k a hair above 32 takes one more grid point, which moves cohesion a little.
    assert weighted.cohesion == pytest.approx(plain.cohesion, abs=1e-3)


def test_unequal_weights_count_fewer_trials():
    counts = np.array([[9, 1], [1, 9], [5, 5], [3, 7]])
    assert ic.measure_counts(counts, weights=[1, 1, 1, 1]).cohesion_k == 32
    assert ic.measure_counts(counts, weights=[3, 1, 1, 1]).cohesion_k == pytest.approx(8 * 3)


@pytest.mark.parametrize("tiny", [1e-320, 5e-324])
def test_two_trials_score_a_cohesion_of_1_with_extreme_weights(tiny):
    result = ic.measure_counts([[9, 1], [1, 9]], weights=[1, tiny])
    assert result.cohesion == pytest.approx(1.0, abs=0.01)


def test_equal_weights_leave_cohesion_exactly_as_it_was():
    counts = np.array([[9, 1, 0], [1, 5, 4], [2, 2, 6], [5, 5, 0]])
    plain = _measures.cohesion(ic.measure_counts(counts).divergences, 32)
    assert ic.measure_counts(counts).cohesion == plain
    assert ic.measure_counts(counts, weights=[3, 3, 3, 3]).cohesion == plain


def test_evenly_spread_divergences_have_no_cohesion():
    for k in (100, _measures.cohesion_k(64)):
        assert _measures.cohesion(np.linspace(0, 1, 2000), k) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("shares", [[0.5, 0.5], [0.82, 0.18], [0.99, 0.01], [0.4, 0.3, 0.2, 0.1]])
def test_each_power_reads_separate_values_by_its_own_formula(shares):
    # Power 1 reads 1 - H as the chance that two values match, and power 0.5 reads H
    # as one equal step for each value. See experiments/power/results/laws.txt.
    k = 400.0
    shares = np.array(shares)
    values = np.repeat(np.linspace(0.1, 0.9, len(shares)), np.round(1000 * shares).astype(int))
    rows = _entropy.densities(values, k, _entropy.points_for(k))[None]
    one = _entropy.density_entropy(rows, k, 1.0)[0]
    half = _entropy.density_entropy(rows, k, _measures.COHESION_POWER)[0]
    matched, places = shares @ shares, np.sqrt(shares).sum() ** 2
    assert 1 - one == pytest.approx((k * matched - 4) / (k - 4), abs=1e-3)
    assert half == pytest.approx(8 * (places - 1) / (k - 8), abs=1e-3)
