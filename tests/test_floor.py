"""The floor: the incoherence left when the observations are dealt out to the trials at random."""

import numpy as np
import pytest

import incoherence as ic
from incoherence._result import _deal_counts


def by_hand(trials, rng, repeats, **kwargs):
    """Shuffle the observations between the trials and measure again, as the FAQ once did."""
    pooled = np.concatenate(trials)
    cuts = np.cumsum([len(trial) for trial in trials])[:-1]
    shuffles = [np.split(rng.permutation(pooled), cuts) for _ in range(repeats)]
    return float(np.mean([ic.incoherence(shuffle, **kwargs) for shuffle in shuffles]))


def draw(seed, centres, sizes):
    rng = np.random.default_rng(seed)
    return [rng.normal(centre, 1, size) for centre, size in zip(centres, sizes, strict=True)]


SAME = draw(1, [0] * 10, [30] * 10)
CAMPS = draw(2, [-3, 3] * 5, [30] * 10)
UNEVEN = draw(3, [-3, 3, 0, 1, -1, 2], [12, 40, 25, 8, 60, 33])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"bounds": (-20, 20)}, {"k": 40}, {"weights": [1, 2, 3, 4, 5, 6]}],
    ids=["defaults", "bounds", "k", "weights"],
)
def test_continuous_floor_matches_shuffling_by_hand(kwargs):
    floor = ic.measure(UNEVEN, **kwargs).floor(repeats=5, rng=np.random.default_rng(7))
    assert floor == pytest.approx(by_hand(UNEVEN, np.random.default_rng(7), 5, **kwargs))


def test_binned_floor_matches_shuffling_by_hand_on_average():
    edges = np.linspace(-8, 8, 9)
    floor = ic.measure(UNEVEN, bins=edges).floor(repeats=2000, rng=1)
    expected = by_hand(UNEVEN, np.random.default_rng(2), 2000, bins=edges)
    assert floor == pytest.approx(expected, rel=0.05)


def test_a_table_gives_the_same_floor_as_the_observations_behind_it():
    edges = np.linspace(-8, 8, 9)
    counts = [np.histogram(trial, edges)[0] for trial in UNEVEN]
    assert ic.measure_counts(counts).floor(rng=5) == ic.measure(UNEVEN, bins=edges).floor(rng=5)


def test_dealt_counts_keep_every_total():
    table = np.array([[5, 0, 9], [1, 1, 1], [0, 40, 2], [7, 7, 7]])
    sizes = table.sum(axis=1)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        dealt = _deal_counts(table, rng) * sizes[:, None]
        assert dealt == pytest.approx(np.round(dealt))
        assert dealt.sum(axis=1) == pytest.approx(sizes)
        assert dealt.sum(axis=0) == pytest.approx(table.sum(axis=0))
        assert (dealt >= 0).all()
        seen.add(tuple(np.round(dealt).astype(int).ravel()))
    assert len(seen) > 1


def test_trials_from_one_source_sit_on_their_floor():
    result = ic.measure(SAME)
    assert result.incoherence == pytest.approx(result.floor(repeats=50, rng=0), abs=0.03)


def test_trials_that_disagree_sit_far_above_their_floor():
    result = ic.measure(CAMPS)
    assert result.incoherence > 5 * result.floor(rng=0)


def test_floor_falls_as_trials_hold_more_observations():
    floors = [
        ic.measure(draw(4, [0] * 10, [size] * 10)).floor(rng=0) for size in [10, 30, 100, 500]
    ]
    assert floors == sorted(floors, reverse=True)
    assert floors[0] > 3 * floors[-1]


def test_a_seed_fixes_the_floor():
    result = ic.measure(CAMPS)
    assert result.floor(rng=3) == result.floor(rng=3)
    assert result.floor(rng=3) == result.floor(rng=np.random.default_rng(3))
    assert result.floor(rng=3) != result.floor(rng=4)


def test_a_generator_moves_on_between_calls():
    result = ic.measure(CAMPS)
    rng = np.random.default_rng(3)
    assert result.floor(rng=rng) != result.floor(rng=rng)


def test_more_repeats_use_more_shuffles():
    result = ic.measure(CAMPS)
    assert result.floor(repeats=1, rng=3) != result.floor(repeats=2, rng=3)


def test_floor_does_not_change_the_result():
    result = ic.measure(CAMPS)
    before = (result.incoherence, result.cohesion)
    result.floor(rng=0)
    assert (result.incoherence, result.cohesion) == before
    assert ic.measure(CAMPS).incoherence == before[0]


def test_identical_observations_have_no_floor():
    assert ic.measure([[2.0] * 5, [2.0] * 8]).floor(rng=0) == pytest.approx(0.0, abs=1e-6)
    assert ic.measure_counts([[4, 0], [9, 0]]).floor(rng=0) == 0.0


def test_one_observation_per_trial_cannot_be_shuffled_apart():
    result = ic.measure_counts([[1, 0], [0, 1]])
    assert result.floor(rng=0) == result.incoherence == 1.0


@pytest.mark.parametrize(
    "counts",
    [
        [[0.5, 0.5], [0.9, 0.1]],
        [[3, 1.5], [2, 2]],
        [[1e9, 1], [1, 1]],
        [[6e8, 1], [6e8, 1]],
        [[1e308, 1e308], [1, 9]],
    ],
    ids=["probabilities", "part counts", "a billion", "a billion in total", "overflowing"],
)
@pytest.mark.filterwarnings("error")
def test_a_table_that_cannot_be_shuffled_has_no_floor(counts):
    result = ic.measure_counts(counts)
    assert 0 <= result.incoherence <= 1
    with pytest.raises(ValueError, match="whole-number counts"):
        result.floor()


@pytest.mark.parametrize("repeats", [0, -1])
def test_repeats_below_one_are_rejected(repeats):
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        ic.measure(CAMPS).floor(repeats=repeats)
