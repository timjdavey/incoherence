"""The public interface: measure an ensemble of trials and read the results."""

from collections.abc import Sequence
from functools import cached_property, partial

import numpy as np
from numpy.typing import ArrayLike

from . import _measures
from ._entropy import Array, densities, density_entropy, shannon
from ._measures import Entropy

#: Default ``k`` for the entropy of continuous observations.
DEFAULT_K = 10.0


class Result:
    """The measures for one ensemble of trials.

    Returned by :func:`measure` and :func:`measure_counts`. You do not build
    one yourself. Each measure is worked out the first time it is read and
    then kept, so reading only ``incoherence`` never pays for ``cohesion``.
    """

    def __init__(self, rows: Array, weights: Array, entropy: Entropy, max_entropy: float) -> None:
        self._rows = rows
        self._weights = weights
        self._entropy = entropy
        self._max_entropy = max_entropy

    def __repr__(self) -> str:
        """Return a short description."""
        return (
            f"{type(self).__name__}(trials={len(self._rows)}, incoherence={self.incoherence:.3f})"
        )

    @cached_property
    def entropies(self) -> Array:
        """Entropy of each trial. Bits for discrete data, a value in [0, 1] for continuous."""
        return self._entropy(self._rows)

    @cached_property
    def pooled_entropy(self) -> float:
        """Entropy of all the trials pooled into one distribution."""
        return float(self._entropy(_measures.pooled(self._rows, self._weights)))

    @cached_property
    def max_entropy(self) -> float:
        """The largest entropy a trial could have."""
        return self._max_entropy

    @cached_property
    def incoherence(self) -> float:
        """How much the trials disagree with each other, from 0 to 1.

        It is 0 when every trial has the same distribution. It is 1 when each
        trial is certain of a different outcome.
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
        return _measures.pairwise_divergences(self._rows, self._entropy, self._weights)

    @cached_property
    def cohesion(self) -> float:
        """How strongly the trials gather into a few shared patterns, from 0 to 1."""
        return _measures.cohesion(self.divergences)


def measure(
    trials: Sequence[ArrayLike] | ArrayLike,
    *,
    bins: int | str | ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    weights: ArrayLike | None = None,
    k: float = DEFAULT_K,
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
        a list of edges, or a rule such as ``"auto"``.
    bounds
        The lowest and highest values an observation could take. Defaults to
        the lowest and highest values seen across all trials.
    weights
        One positive weight per trial. Defaults to equal weights, so each trial
        counts the same however many observations it holds.
    k
        How quickly an observation's influence falls away with distance when
        estimating continuous entropy. Ignored when ``bins`` is given.

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
        edges = np.histogram_bin_edges(np.concatenate(observations), bins, (low, high))
        counts = np.stack([np.histogram(trial, edges)[0] for trial in observations])
        return measure_counts(counts, weights=weights)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    width = high - low or 1.0
    rows = np.stack([densities((trial - low) / width, k) for trial in observations])
    return Result(rows, _as_weights(weights, len(rows)), partial(density_entropy, k=k), 1.0)


def measure_counts(counts: ArrayLike, *, weights: ArrayLike | None = None) -> Result:
    """Measure an ensemble of trials from a table of counts or probabilities.

    Parameters
    ----------
    counts
        A two-dimensional array with one row per trial and one column per
        state. Each row is scaled to sum to 1, so counts and probabilities
        both work.
    weights
        One positive weight per trial. Defaults to equal weights.

    Returns
    -------
    Result
        Holds ``incoherence``, ``cohesion`` and the values they are built from.
    """
    table = np.asarray(counts, dtype=np.float64)
    if table.ndim != 2 or len(table) < 2:
        raise ValueError("counts needs one row per trial and at least two trials")
    if not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("counts must be finite and not negative")
    totals = table.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        raise ValueError("every trial needs at least one observation")
    max_entropy = float(np.log2(max(table.shape[1], 2)))
    return Result(table / totals, _as_weights(weights, len(table)), shannon, max_entropy)


def incoherence(trials: Sequence[ArrayLike] | ArrayLike, **kwargs) -> float:
    """Incoherence of an ensemble of trials. Takes the same arguments as :func:`measure`."""
    return measure(trials, **kwargs).incoherence


def cohesion(trials: Sequence[ArrayLike] | ArrayLike, **kwargs) -> float:
    """Cohesion of an ensemble of trials. Takes the same arguments as :func:`measure`."""
    return measure(trials, **kwargs).cohesion


def _as_trials(trials: Sequence[ArrayLike] | ArrayLike) -> list[Array]:
    """Check the observations and return one flat array per trial."""
    if isinstance(trials, np.ndarray) and trials.ndim != 2:
        raise ValueError("an array of trials needs two dimensions, one row per trial")
    observations = [np.asarray(trial, dtype=np.float64) for trial in trials]  # type: ignore[union-attr]
    if len(observations) < 2:
        raise ValueError("an ensemble needs at least two trials")
    for trial in observations:
        if trial.ndim != 1 or trial.size == 0:
            raise ValueError("every trial needs a flat list of at least one observation")
        if not np.isfinite(trial).all():
            raise ValueError("observations must be finite, so remove any NaN or infinite values")
    return observations


def _as_bounds(
    observations: list[Array], bounds: tuple[float, float] | None
) -> tuple[float, float]:
    """Return the range the observations are scaled by."""
    low = min(float(trial.min()) for trial in observations)
    high = max(float(trial.max()) for trial in observations)
    if bounds is None:
        return low, high
    if not bounds[0] < bounds[1]:
        raise ValueError(f"bounds must be (low, high) with low below high, got {bounds}")
    if low < bounds[0] or high > bounds[1]:
        raise ValueError(f"observations span ({low}, {high}), which falls outside bounds {bounds}")
    return float(bounds[0]), float(bounds[1])


def _as_weights(weights: ArrayLike | None, count: int) -> Array:
    """Return one weight per trial, scaled to sum to 1."""
    if weights is None:
        return np.full(count, 1 / count)
    out = np.asarray(weights, dtype=np.float64)
    if out.shape != (count,):
        raise ValueError(f"weights needs one value per trial, expected {count}")
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("weights must be finite and positive")
    return out / out.sum()
