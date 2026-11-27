"""Worked values and stated properties from the two papers.

"Incoherence" is Davey (2024), Entropy 26(8), 683.
"Cohesion" is Davey (2023), Entropy 25(12), 1605.
"""

import numpy as np
import pytest

import incoherence as ic
from incoherence import _measures


def discrete(counts):
    return ic.measure_counts(counts)


class TestIncoherenceDiscrete:
    """Incoherence paper, sections 3.8 and 5.1.1."""

    @pytest.mark.parametrize(
        ("counts", "pooled_entropy", "expected"),
        [
            ([[0.9, 0.1, 0, 0, 0], [0.8, 0.2, 0, 0, 0]], 0.6, 0.107),
            ([[0.2, 0.2, 0.2, 0.2, 0.2], [0.4, 0.0, 0.2, 0.2, 0.2]], 2.2, 0.103),
        ],
    )
    def test_condition_vi_table(self, counts, pooled_entropy, expected):
        result = discrete(counts)
        assert result.pooled_entropy == pytest.approx(pooled_entropy, abs=0.05)
        assert result.incoherence == pytest.approx(expected, abs=0.0005)

    @pytest.mark.parametrize("size", [2, 3, 4, 10, 50])
    def test_identity_matrix_is_one(self, size):
        assert discrete(np.eye(size)).incoherence == pytest.approx(1.0)

    def test_identical_trials_are_zero(self):
        assert discrete([[0.3, 0.7]] * 5).incoherence == pytest.approx(0.0)

    def test_highest_against_lowest_entropy(self):
        assert discrete([[0.5, 0.5], [1, 0]]).incoherence == pytest.approx(0.654, abs=0.0005)

    @pytest.mark.parametrize("states", [3, 5, 10, 50, 200])
    def test_figure_5_steady_as_states_are_added(self, states):
        uniform, certain = np.ones(states), np.eye(states)[0]
        assert discrete([uniform, certain]).incoherence == pytest.approx(0.65, abs=0.01)

    @pytest.mark.parametrize("copies", [2, 5, 20])
    def test_figure_4_steady_as_trials_are_duplicated(self, copies):
        pair = [[0.5, 0.5], [1, 0]]
        assert discrete(pair * copies).incoherence == pytest.approx(discrete(pair).incoherence)

    def test_figure_3_falls_as_the_outlier_is_outnumbered(self):
        values = [discrete([[0.5, 0.5]] + [[1, 0]] * n).incoherence for n in (3, 10, 100, 300)]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx(0.71, abs=0.005)

    def test_figure_6_falls_as_shared_empty_states_are_added(self):
        values = [discrete(np.eye(states)[:2]).incoherence for states in (2, 3, 4, 8)]
        assert values == sorted(values, reverse=True)


class TestIncoherenceContinuous:
    """Incoherence paper, section 5.1.2."""

    def test_highest_against_lowest_entropy_matches_discrete(self):
        rng = np.random.default_rng(0)
        result = ic.measure([rng.uniform(0, 1, 5000), np.ones(5000)])
        assert result.incoherence == pytest.approx(0.65, abs=0.02)

    def test_figure_8_spikes_approach_one(self):
        def spikes(count):
            return [np.full(50, i / count) for i in range(count)]

        values = [ic.incoherence(spikes(count), bounds=(0, 1)) for count in (2, 3, 6, 12, 50)]
        assert values == sorted(values)
        assert values[0] < 0.9
        assert values[-1] == pytest.approx(1.0, abs=0.01)

    def test_identical_trials_are_zero(self):
        trial = np.random.default_rng(0).normal(size=300)
        assert ic.incoherence([trial] * 4) == pytest.approx(0.0, abs=1e-9)

    def test_detects_a_change_in_spread_alone(self):
        rng = np.random.default_rng(0)
        steady = [rng.normal(1, 0.1, 2000) for _ in range(13)]
        one_wide = steady[:12] + [rng.normal(1, 0.5, 2000)]
        assert ic.incoherence(one_wide) > 3 * ic.incoherence(steady)


class TestCohesion:
    """Cohesion paper, section 4 and figure 2."""

    def test_identical_trials_are_one(self):
        assert discrete([[1, 2, 3]] * 6).cohesion == pytest.approx(1.0)
        trial = np.random.default_rng(0).normal(size=300)
        assert ic.cohesion([trial] * 6) == pytest.approx(1.0, abs=1e-6)

    def test_two_clear_scenarios(self):
        """The paper's 0.8 was at its fixed ``k`` of 100. Six trials give 48 by default."""
        counts = [[9, 1, 0, 0]] * 3 + [[0, 0, 1, 9]] * 3
        paper = ic.measure_counts(counts, cohesion_k=_measures.PAPER_COHESION_K)
        assert paper.cohesion == pytest.approx(0.8, abs=0.05)

    def test_scenarios_score_above_scatter(self):
        camps, scatter = reference_trials()[1:]
        assert ic.cohesion(camps) > 0.75
        assert ic.cohesion(scatter) < 0.4


def reference_trials():
    rng = np.random.default_rng(1)
    same = [rng.normal(0, 1, 500) for _ in range(10)]
    camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
    scatter = [rng.normal(rng.uniform(-4, 4), rng.uniform(0.3, 2), 500) for _ in range(10)]
    return same, camps, scatter


@pytest.mark.parametrize(
    ("index", "incoherence", "cohesion"),
    [(0, 0.0300, 0.9510), (1, 0.5712, 0.8017), (2, 0.4543, 0.3019)],
    ids=["same", "camps", "scatter"],
)
def test_matches_the_code_behind_the_papers(index, incoherence, cohesion):
    """Values from ``ensemblepy`` in timjdavey/incoherence-papers at 371249c.

    That code measured cohesion with ``k = 100`` on a 501-point grid. The
    package grows ``k`` with trials and uses a finer grid, so cohesion is
    checked at the old settings here.
    """
    result = ic.measure(reference_trials()[index])
    assert result.incoherence == pytest.approx(incoherence, abs=1e-3)
    old = _measures.cohesion(result.divergences, _measures.PAPER_COHESION_K, points=501)
    assert old == pytest.approx(cohesion, abs=1e-3)


@pytest.mark.parametrize(
    ("index", "cohesion"),
    [(0, 0.996), (1, 0.831), (2, 0.318)],
    ids=["same", "camps", "scatter"],
)
def test_cohesion_on_the_finer_grid(index, cohesion):
    """What ``cohesion_k=100`` returns, which differs from the old code by up to 0.06."""
    trials = reference_trials()[index]
    assert ic.cohesion(trials, cohesion_k=100) == pytest.approx(cohesion, abs=1e-3)


@pytest.mark.parametrize(
    ("index", "cohesion"),
    [(0, 0.996), (1, 0.794), (2, 0.313)],
    ids=["same", "camps", "scatter"],
)
def test_cohesion_by_default(index, cohesion):
    """What the package returns by default, with ``k = 8n``, here 80."""
    assert ic.cohesion(reference_trials()[index]) == pytest.approx(cohesion, abs=1e-3)


@pytest.mark.parametrize(
    ("value", "old_grid", "new_grid"),
    [(0.004, 1.000, 1.000), (0.0025, 0.927, 0.995), (0.3333, 0.932, 0.996)],
)
def test_grid_table_in_the_improvements_page(value, old_grid, new_grid):
    divergences = np.full(45, value)
    k = _measures.PAPER_COHESION_K
    assert _measures.cohesion(divergences, k, points=501) == pytest.approx(old_grid, abs=0.0005)
    assert _measures.cohesion(divergences, k) == pytest.approx(new_grid, abs=0.0005)


@pytest.mark.parametrize("value", np.linspace(0, 1, 41) + 0.0003)
def test_equal_divergences_are_fully_cohesive_wherever_they_fall(value):
    """The 501-point grid gave as little as 0.92 here, depending on the value."""
    divergences = np.full(45, min(value, 1.0))
    for k in (_measures.PAPER_COHESION_K, _measures.cohesion_k(10)):
        assert _measures.cohesion(divergences, k) == pytest.approx(1.0, abs=0.006)
