"""The public interface: measure an ensemble of trials and read the results."""

from collections.abc import Callable, Sequence
from functools import cached_property, partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _measures
from ._entropy import Array, densities, density_entropy, points_for, shannon
from ._measures import Entropy

#: Default ``k`` for the entropy of continuous observations.
DEFAULT_K = 10.0

#: Largest ``k``. The grid grows with ``k``, so this bounds the time and memory.
MAX_K = 1000.0

#: Smallest ``k`` used. Results have settled by here, and much lower loses precision.
MIN_K = 1e-4

#: Default number of shuffles behind :meth:`Result.floor`.
DEFAULT_REPEATS = 20

#: A table of counts must hold fewer observations than this to be shuffled, a limit set by numpy.
MAX_DEALT = 10**9

#: Shuffles the observations between the trials and returns the new rows.
Deal = Callable[[np.random.Generator], Array]


class Result:
    """The measures for one ensemble of trials.

    Returned by :func:`measure` and :func:`measure_counts`. You do not build
    one yourself. Each measure is worked out the first time it is read and
    then kept, so reading only ``incoherence`` never pays for ``cohesion``.
    """

    def __init__(
        self,
        rows: Array,
        weights: Array,
        entropy: Entropy,
        max_entropy: float,
        deal: Deal | None = None,
        cohesion_k: float | None = None,
    ) -> None:
        self._rows = rows
        self._weights = weights
        self._entropy = entropy
        self._max_entropy = max_entropy
        self._deal = deal
        self._cohesion_k = cohesion_k

    def __repr__(self) -> str:
        """Return a short description."""
        return (
            f"{type(self).__name__}(trials={len(self._rows)}, incoherence={self.incoherence:.3f})"
        )

    @cached_property
    def entropies(self) -> Array:
        """Entropy of each trial. Bits for discrete data, a value in [0, 1] for continuous."""
        return _read_only(self._entropy(self._rows))

    @cached_property
    def pooled_entropy(self) -> float:
        """Entropy of all the trials pooled into one distribution."""
        return float(self._entropy(_measures.pooled(self._rows, self._weights)))

    @property
    def max_entropy(self) -> float:
        """The largest entropy a trial could have."""
        return self._max_entropy

    @cached_property
    def incoherence(self) -> float:
        """How much the trials disagree with each other, from 0 to 1.

        It is 0 when every trial has the same distribution. For a table of
        counts, it reaches 1 when each trial is certain of one state and the
        pooled trials cover every state evenly. Unused states lower it, so two
        trials certain of different states out of four score 0.71.
        """
        return _measures.incoherence(
            self.entropies, self.pooled_entropy, self.max_entropy, self._weights
        )

    @cached_property
    def divergences(self) -> Array:
        """Jensen-Shannon divergence between every pair of trials, each from 0 to 1.

        The order matches ``numpy.triu_indices(trials, 1)``, so
        ``scipy.spatial.distance.squareform`` turns it into a square matrix.
        """
        return _read_only(_measures.pairwise_divergences(self._rows, self._entropy, self._weights))

    @property
    def cohesion_k(self) -> float:
        """The ``k`` behind ``cohesion``.

        Unless you set ``cohesion_k``, it is 8 for each trial. With unequal
        weights, trials are counted as ``1 / sum(weights ** 2)`` of the weights
        scaled to sum to 1, so a trial with almost no weight barely counts.
        """
        if self._cohesion_k is not None:
            return self._cohesion_k
        return _measures.cohesion_k(_measures.effective_trials(self._weights))

    @cached_property
    def cohesion(self) -> float:
        """How strongly the trials gather into a few shared patterns, from 0 to 1.

        Read it next to ``incoherence``. Trials that all disagree by the same
        amount also score 1, because their pairwise divergences bunch around
        one value. Two trials always score near 1, because they make one pair.
        With unequal weights, a pair counts in proportion to both its weights.
        """
        return _measures.cohesion(
            self.divergences, self.cohesion_k, weights=_measures.pair_weights(self._weights)
        )

    def floor(
        self, repeats: int = DEFAULT_REPEATS, rng: np.random.Generator | int | None = None
    ) -> float:
        """Incoherence these observations would give if every trial came from one source.

        Every observation is pooled, then dealt back out at random into trials
        of the same sizes. The shuffled trials share one source, so whatever
        incoherence is left comes from the sample size alone. An ``incoherence``
        close to its floor may not be a real disagreement.

        The shuffles use the same settings as the measurement. Each one costs
        about as much as measuring ``incoherence`` once.

        Parameters
        ----------
        repeats
            How many times to shuffle. The result is the mean over them.
        rng
            A seed or a ``numpy.random.Generator``. Leave as ``None`` for a
            fresh shuffle on every call, which moves the result a little.

        Raises
        ------
        ValueError
            If the result came from a table of probabilities, because the
            floor depends on how many observations each trial holds.
        """
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        if self._deal is None:
            raise ValueError(
                "floor needs whole-number counts that total less than a billion, "
                "because it deals the observations out again at random"
            )
        generator = np.random.default_rng(rng)
        scores = [
            Result(
                self._deal(generator),
                self._weights,
                self._entropy,
                self._max_entropy,
                cohesion_k=self._cohesion_k,
            ).incoherence
            for _ in range(repeats)
        ]
        return float(np.mean(scores))


def measure(
    trials: Sequence[ArrayLike] | ArrayLike,
    *,
    bins: int | str | ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    weights: ArrayLike | None = None,
    k: float = DEFAULT_K,
    cohesion_k: float | None = None,
) -> Result:
    """Measure an ensemble of trials from the observations in each trial.

    A trial is one run of an experiment or simulation. It holds the observations
    from that run. The ensemble is all of the trials together.

    Parameters
    ----------
    trials
        The observations from each trial. Pass a list with one array per trial,
        or a two-dimensional array with one row per trial. Trials may hold
        different numbers of observations.
    bins
        Leave as ``None`` to treat the observations as continuous. Otherwise
        the observations are counted into bins first and treated as discrete.
        Takes anything ``numpy.histogram_bin_edges`` accepts: a number of bins,
        a list of edges, or a rule such as ``"auto"``. If you give a list of
        edges, every observation must fall inside them. The last bin includes
        its upper edge, so ``n`` categories need ``n + 1`` edges. Use
        ``[0, 1, 2, 3]`` for the values 0, 1 and 2.
    bounds
        The lowest and highest values an observation could take. Defaults to
        the lowest and highest values seen across all trials. Cannot be
        combined with a list of bin edges.
    weights
        One positive weight per trial. Defaults to equal weights, so each trial
        counts the same however many observations it holds. A weight sets the
        trial's share of the pooled distribution, and each pair of trials
        counts towards cohesion in proportion to both its weights.
    k
        How quickly an observation's influence falls away with distance when
        estimating continuous entropy, up to 1000. A higher ``k`` picks out
        finer detail and takes longer. Below about 5 the scale is unreliable,
        because one repeated value no longer scores near 0. Anything below
        0.0001 is treated as 0.0001, because results have settled by then.
        Ignored when ``bins`` is given.
    cohesion_k
        How quickly each pairwise divergence's influence falls away when
        estimating cohesion. Leave as ``None`` for 8 times the number of
        trials, see :attr:`Result.cohesion_k`. Pass ``100`` to match the papers,
        which fixed it there. A higher value is slower, as the grid grows with
        it. This is separate from ``k``, and applies to continuous and binned
        data alike.

    Returns
    -------
    Result
        Holds ``incoherence``, ``cohesion`` and the values they are built from.

    See Also
    --------
    measure_counts : Start from a table of counts or probabilities.
    """
    observations = _as_trials(trials)
    low, high = _as_bounds(observations, bounds)
    if bins is not None:
        if bounds is not None and np.ndim(bins) == 1:
            raise ValueError("pass either bounds or a list of bin edges, not both")
        edges = np.histogram_bin_edges(np.concatenate(observations), bins, (low, high))
        seen = _span(observations)
        if seen[0] < edges[0] or seen[1] > edges[-1]:
            raise ValueError(
                f"observations span {seen}, which falls outside "
                f"the bin edges ({edges[0]}, {edges[-1]})"
            )
        counts = np.stack([np.histogram(trial, edges)[0] for trial in observations])
        return measure_counts(counts, weights=weights, cohesion_k=cohesion_k)
    k = _as_k(k, "k", MAX_K)
    cohesion_k = None if cohesion_k is None else _as_k(cohesion_k, "cohesion_k")
    points = points_for(k)
    scaled = [_to_unit(trial, low, high) for trial in observations]
    to_rows = partial(_density_rows, k=k, points=points)
    return Result(
        to_rows(scaled),
        _as_weights(weights, len(scaled)),
        partial(density_entropy, k=k),
        1.0,
        partial(_deal_observations, scaled, to_rows),
        cohesion_k,
    )


def measure_counts(
    counts: ArrayLike, *, weights: ArrayLike | None = None, cohesion_k: float | None = None
) -> Result:
    """Measure an ensemble of trials from a table of counts or probabilities.

    Parameters
    ----------
    counts
        A two-dimensional array with one row per trial and one column per
        state. Each row is scaled to sum to 1, so counts and probabilities
        both work.
    weights
        One positive weight per trial. Defaults to equal weights.
    cohesion_k
        The ``k`` behind cohesion. Leave as ``None`` for 8 times the number of
        trials, or pass ``100`` to match the papers. See :func:`measure`.

    Returns
    -------
    Result
        Holds ``incoherence``, ``cohesion`` and the values they are built from.
    """
    table = _as_floats(counts)
    if table.ndim != 2 or len(table) < 2:
        raise ValueError("counts needs one row per trial and at least two trials")
    if not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("counts must be finite and not negative")
    if not table.any(axis=1).all():
        raise ValueError("every trial needs at least one observation")
    cohesion_k = None if cohesion_k is None else _as_k(cohesion_k, "cohesion_k")
    max_entropy = float(np.log2(table.shape[1]))
    whole = _as_whole(table)
    deal = None if whole is None else partial(_deal_counts, whole)
    return Result(
        _sum_to_one(table),
        _as_weights(weights, len(table)),
        shannon,
        max_entropy,
        deal,
        cohesion_k,
    )


def incoherence(
    trials: Sequence[ArrayLike] | ArrayLike,
    *,
    bins: int | str | ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    weights: ArrayLike | None = None,
    k: float = DEFAULT_K,
) -> float:
    """Incoherence of an ensemble of trials.

    Takes the same arguments as :func:`measure`, bar ``cohesion_k``, which has
    no effect on incoherence.
    """
    return measure(trials, bins=bins, bounds=bounds, weights=weights, k=k).incoherence


def cohesion(
    trials: Sequence[ArrayLike] | ArrayLike,
    *,
    bins: int | str | ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    weights: ArrayLike | None = None,
    k: float = DEFAULT_K,
    cohesion_k: float | None = None,
) -> float:
    """Cohesion of an ensemble of trials. Takes the same arguments as :func:`measure`."""
    return measure(
        trials, bins=bins, bounds=bounds, weights=weights, k=k, cohesion_k=cohesion_k
    ).cohesion


def _as_k(value: float, name: str, maximum: float = np.inf) -> float:
    """Check a ``k`` and return it as a float, raised to ``MIN_K`` if it is below."""
    k = float(value)
    if not (0 < k <= maximum and np.isfinite(k)):
        limit = "finite" if np.isinf(maximum) else f"at most {maximum:g}"
        raise ValueError(f"{name} must be positive and {limit}, got {k}")
    return max(k, MIN_K)


def _read_only(values: Array) -> Array:
    """Lock an array, so editing a kept result cannot change later measures."""
    values.setflags(write=False)
    return values


def _as_floats(values: ArrayLike) -> Array:
    """Return values as floats, refusing any that are masked."""
    out = np.ma.asarray(values, dtype=np.float64)
    if np.ma.is_masked(out):
        raise ValueError("masked values are not supported, so remove them first")
    return np.asarray(out)


def _sum_to_one(values: Array, axis: int = -1) -> Array:
    """Scale values to sum to 1, dividing by the largest first so the sum cannot overflow."""
    scaled = values / values.max(axis=axis, keepdims=True)
    return scaled / scaled.sum(axis=axis, keepdims=True)


def _to_unit(trial: Array, low: float, high: float) -> Array:
    """Scale observations to [0, 1], halving them first if the range would overflow."""
    half = 1.0 if np.isfinite(high - low) else 0.5
    width = half * high - half * low or 1.0
    return (half * trial - half * low) / width


def _as_whole(table: Array) -> NDArray[np.int64] | None:
    """Return a table as whole-number counts, or ``None`` if it cannot be shuffled.

    Probabilities cannot, because they do not say how many observations a trial holds.
    """
    if (table != np.round(table)).any() or table.max() >= MAX_DEALT or table.sum() >= MAX_DEALT:
        return None
    return table.astype(np.int64)


def _density_rows(groups: list[Array], k: float, points: int) -> Array:
    """Return one row of densities for each group of observations."""
    return np.stack([densities(group, k, points) for group in groups])


def _deal_observations(
    trials: list[Array], to_rows: Callable[[list[Array]], Array], rng: np.random.Generator
) -> Array:
    """Deal the pooled observations out at random into trials of the same sizes."""
    cuts = np.cumsum([len(trial) for trial in trials])[:-1]
    return to_rows(np.split(rng.permutation(np.concatenate(trials)), cuts))


def _deal_counts(table: NDArray[np.int64], rng: np.random.Generator) -> Array:
    """Deal the pooled counts out at random into trials of the same sizes.

    Each trial draws its observations from what is left of the pool, without
    putting any back. That gives what shuffling the observations one by one
    would give, without having to list them.
    """
    sizes = table.sum(axis=1)
    left = table.sum(axis=0)
    dealt = np.empty_like(table)
    for i, size in enumerate(sizes[:-1]):
        dealt[i] = rng.multivariate_hypergeometric(left, size)
        left = left - dealt[i]
    dealt[-1] = left
    return dealt / sizes[:, None]


def _as_trials(trials: Sequence[ArrayLike] | ArrayLike) -> list[Array]:
    """Check the observations and return one flat array per trial."""
    if isinstance(trials, np.ndarray) and trials.ndim != 2:
        # A flat array of objects is how numpy holds trials of different lengths.
        if not (trials.ndim == 1 and trials.dtype == object):
            raise ValueError("an array of trials needs two dimensions, one row per trial")
    observations = [_as_floats(trial) for trial in trials]  # type: ignore[union-attr]
    if len(observations) < 2:
        raise ValueError("an ensemble needs at least two trials")
    for trial in observations:
        if trial.ndim != 1 or trial.size == 0:
            raise ValueError("every trial needs a flat list of at least one observation")
        if not np.isfinite(trial).all():
            raise ValueError("observations must be finite, so remove any NaN or infinite values")
    return observations


def _span(observations: list[Array]) -> tuple[float, float]:
    """Return the lowest and highest observation across all trials."""
    low = min(float(trial.min()) for trial in observations)
    high = max(float(trial.max()) for trial in observations)
    return low, high


def _as_bounds(
    observations: list[Array], bounds: tuple[float, float] | None
) -> tuple[float, float]:
    """Return the range the observations are scaled by."""
    low, high = _span(observations)
    if bounds is None:
        return low, high
    limits = _as_floats(bounds)
    if limits.shape != (2,):
        raise ValueError(f"bounds must be two values, (low, high), got {bounds}")
    if not np.isfinite(limits).all():
        raise ValueError(f"bounds must be finite, got {bounds}")
    if not limits[0] < limits[1]:
        raise ValueError(f"bounds must be (low, high) with low below high, got {bounds}")
    if low < limits[0] or high > limits[1]:
        raise ValueError(f"observations span ({low}, {high}), which falls outside bounds {bounds}")
    return float(limits[0]), float(limits[1])


def _as_weights(weights: ArrayLike | None, count: int) -> Array:
    """Return one weight per trial, scaled to sum to 1."""
    if weights is None:
        return np.full(count, 1 / count)
    out = _as_floats(weights)
    if out.shape != (count,):
        raise ValueError(f"weights needs one value per trial, expected {count}")
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("weights must be finite and positive")
    scaled = _sum_to_one(out)
    if (scaled == 0).any():
        raise ValueError("weights are too far apart to represent")
    return scaled
