"""The public interface: measure an ensemble of trials."""

from collections.abc import Sequence
from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from . import _inputs, _shuffle
from ._entropy import density_entropy, points_for, shannon
from ._result import Result

#: Default ``k`` for the entropy of continuous observations.
DEFAULT_K = 10.0

#: Largest ``k``. The grid grows with ``k``, so this bounds the time and memory.
MAX_K = 1000.0


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
    observations = _inputs.as_trials(trials)
    low, high = _inputs.as_bounds(observations, bounds)
    if bins is not None:
        if bounds is not None and np.ndim(bins) == 1:
            raise ValueError("pass either bounds or a list of bin edges, not both")
        edges = np.histogram_bin_edges(np.concatenate(observations), bins, (low, high))
        seen = _inputs.span(observations)
        if seen[0] < edges[0] or seen[1] > edges[-1]:
            raise ValueError(
                f"observations span {seen}, which falls outside "
                f"the bin edges ({edges[0]}, {edges[-1]})"
            )
        counts = np.stack([np.histogram(trial, edges)[0] for trial in observations])
        return measure_counts(counts, weights=weights, cohesion_k=cohesion_k)
    k = _inputs.as_k(k, "k", MAX_K)
    cohesion_k = None if cohesion_k is None else _inputs.as_k(cohesion_k, "cohesion_k")
    points = points_for(k)
    scaled = [_inputs.to_unit(trial, low, high) for trial in observations]
    to_rows = partial(_shuffle.density_rows, k=k, points=points)
    return Result(
        to_rows(scaled),
        _inputs.as_weights(weights, len(scaled)),
        partial(density_entropy, k=k),
        1.0,
        partial(_shuffle.deal_observations, scaled, to_rows),
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
    table = _inputs.as_floats(counts)
    if table.ndim != 2 or len(table) < 2:
        raise ValueError("counts needs one row per trial and at least two trials")
    if not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("counts must be finite and not negative")
    if not table.any(axis=1).all():
        raise ValueError("every trial needs at least one observation")
    cohesion_k = None if cohesion_k is None else _inputs.as_k(cohesion_k, "cohesion_k")
    max_entropy = float(np.log2(table.shape[1]))
    whole = _shuffle.as_whole(table)
    deal = None if whole is None else partial(_shuffle.deal_counts, whole)
    return Result(
        _inputs.sum_to_one(table),
        _inputs.as_weights(weights, len(table)),
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
