"""The conditions a measure must meet (Incoherence paper, section 3.5), on random data."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import incoherence as ic
from incoherence._entropy import densities, density_entropy

counts_tables = st.tuples(st.integers(2, 6), st.integers(2, 6)).flatmap(
    lambda shape: hnp.arrays(np.int64, shape, elements=st.integers(0, 50)).filter(
        lambda table: (table.sum(axis=1) > 0).all()
    )
)

values = st.floats(-100, 100, allow_nan=False, width=32)
continuous_trials = st.lists(st.lists(values, min_size=1, max_size=30), min_size=2, max_size=6)

ABS = 1e-9


@given(counts_tables)
def test_discrete_measures_are_bounded(counts):
    result = ic.measure_counts(counts)
    assert 0 <= result.incoherence <= 1
    assert 0 <= result.cohesion <= 1
    assert ((result.divergences >= 0) & (result.divergences <= 1)).all()


@settings(deadline=None)
@given(continuous_trials)
def test_continuous_measures_are_bounded(trials):
    result = ic.measure(trials)
    assert 0 <= result.incoherence <= 1
    assert 0 <= result.cohesion <= 1
    assert ((result.entropies >= 0) & (result.entropies <= 1)).all()


@given(counts_tables, st.randoms(use_true_random=False))
def test_order_of_trials_and_states_does_not_matter(counts, random):
    rows = random.sample(range(counts.shape[0]), counts.shape[0])
    columns = random.sample(range(counts.shape[1]), counts.shape[1])
    shuffled = ic.measure_counts(counts[rows][:, columns])
    original = ic.measure_counts(counts)
    assert shuffled.incoherence == pytest.approx(original.incoherence, abs=ABS)
    assert np.sort(shuffled.divergences) == pytest.approx(np.sort(original.divergences), abs=ABS)


@given(counts_tables, st.integers(2, 4))
def test_repeating_every_trial_does_not_change_incoherence(counts, copies):
    repeated = ic.measure_counts(np.tile(counts, (copies, 1)))
    assert repeated.incoherence == pytest.approx(ic.measure_counts(counts).incoherence, abs=ABS)


@given(counts_tables, st.integers(2, 9))
def test_more_observations_in_one_trial_do_not_change_incoherence(counts, factor):
    scaled = counts.copy()
    scaled[0] *= factor
    assert ic.measure_counts(scaled).incoherence == pytest.approx(
        ic.measure_counts(counts).incoherence, abs=ABS
    )


@settings(deadline=None)
@given(continuous_trials, st.integers(2, 4))
def test_repeating_observations_within_a_trial_changes_nothing(trials, copies):
    repeated = [trials[0] * copies] + trials[1:]
    assert ic.incoherence(repeated) == pytest.approx(ic.incoherence(trials), abs=1e-6)


@settings(deadline=None)
@given(continuous_trials, st.floats(0.01, 100), st.floats(-100, 100))
def test_continuous_scale_and_shift_do_not_matter(trials, scale, shift):
    moved = [np.asarray(trial, dtype=np.float64) * scale + shift for trial in trials]
    assert ic.incoherence(moved) == pytest.approx(ic.incoherence(trials), abs=1e-4)


@given(counts_tables)
def test_identical_trials_are_coherent_and_cohesive(counts):
    result = ic.measure_counts(np.tile(counts[:1], (4, 1)))
    assert result.incoherence == pytest.approx(0.0, abs=1e-7)
    assert result.cohesion == pytest.approx(1.0, abs=1e-7)


unit_trials = st.lists(
    st.lists(st.floats(0, 1, width=32), min_size=1, max_size=30), min_size=2, max_size=6
)


@pytest.mark.parametrize(("k", "power"), [(10, 1.0), (10, 0.5), (100, 1.0), (100, 0.5)])
@settings(deadline=None)
@given(unit_trials, st.randoms(use_true_random=False))
def test_pooling_never_lowers_continuous_entropy(k, power, trials, random):
    """Incoherence needs the pooled entropy to be at least the average trial entropy.

    It holds with and without the square root of the densities (``power=0.5``).
    """
    rows = np.array([densities(np.asarray(trial, dtype=np.float64), k) for trial in trials])
    weights = np.array([random.uniform(0.01, 1) for _ in trials])
    weights /= weights.sum()
    pooled = density_entropy((weights[:, None] * rows).sum(axis=0)[None], k, power)[0]
    assert pooled >= weights @ density_entropy(rows, k, power) - ABS
