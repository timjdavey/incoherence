"""Check what the user passes in and put it in the form the measures expect."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._entropy import Array

#: Smallest ``k`` used. Results have settled by here, and much lower loses precision.
MIN_K = 1e-4


def as_k(value: float, name: str, maximum: float = np.inf) -> float:
    """Check a ``k`` and return it as a float, raised to ``MIN_K`` if it is below."""
    k = float(value)
    if not (0 < k <= maximum and np.isfinite(k)):
        limit = "finite" if np.isinf(maximum) else f"at most {maximum:g}"
        raise ValueError(f"{name} must be positive and {limit}, got {k}")
    return max(k, MIN_K)


def as_floats(values: ArrayLike) -> Array:
    """Return values as floats, refusing any that are masked."""
    out = np.ma.asarray(values, dtype=np.float64)
    if np.ma.is_masked(out):
        raise ValueError("masked values are not supported, so remove them first")
    return np.asarray(out)


def sum_to_one(values: Array, axis: int = -1) -> Array:
    """Scale values to sum to 1, dividing by the largest first so the sum cannot overflow."""
    scaled = values / values.max(axis=axis, keepdims=True)
    return scaled / scaled.sum(axis=axis, keepdims=True)


def to_unit(trial: Array, low: float, high: float) -> Array:
    """Scale observations to [0, 1], halving them first if the range would overflow."""
    half = 1.0 if np.isfinite(high - low) else 0.5
    width = half * high - half * low or 1.0
    return (half * trial - half * low) / width


def as_trials(trials: Sequence[ArrayLike] | ArrayLike) -> list[Array]:
    """Check the observations and return one flat array per trial."""
    if isinstance(trials, np.ndarray) and trials.ndim != 2:
        # A flat array of objects is how numpy holds trials of different lengths.
        if not (trials.ndim == 1 and trials.dtype == object):
            raise ValueError("an array of trials needs two dimensions, one row per trial")
    observations = [as_floats(trial) for trial in trials]  # type: ignore[union-attr]
    if len(observations) < 2:
        raise ValueError("an ensemble needs at least two trials")
    for trial in observations:
        if trial.ndim != 1 or trial.size == 0:
            raise ValueError("every trial needs a flat list of at least one observation")
        if not np.isfinite(trial).all():
            raise ValueError("observations must be finite, so remove any NaN or infinite values")
    return observations


def span(observations: list[Array]) -> tuple[float, float]:
    """Return the lowest and highest observation across all trials."""
    low = min(float(trial.min()) for trial in observations)
    high = max(float(trial.max()) for trial in observations)
    return low, high


def as_bounds(observations: list[Array], bounds: tuple[float, float] | None) -> tuple[float, float]:
    """Return the range the observations are scaled by."""
    low, high = span(observations)
    if bounds is None:
        return low, high
    limits = as_floats(bounds)
    if limits.shape != (2,):
        raise ValueError(f"bounds must be two values, (low, high), got {bounds}")
    if not np.isfinite(limits).all():
        raise ValueError(f"bounds must be finite, got {bounds}")
    if not limits[0] < limits[1]:
        raise ValueError(f"bounds must be (low, high) with low below high, got {bounds}")
    if low < limits[0] or high > limits[1]:
        raise ValueError(f"observations span ({low}, {high}), which falls outside bounds {bounds}")
    return float(limits[0]), float(limits[1])


def as_weights(weights: ArrayLike | None, count: int) -> Array:
    """Return one weight per trial, scaled to sum to 1."""
    if weights is None:
        return np.full(count, 1 / count)
    out = as_floats(weights)
    if out.shape != (count,):
        raise ValueError(f"weights needs one value per trial, expected {count}")
    if not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("weights must be finite and positive")
    scaled = sum_to_one(out)
    if (scaled == 0).any():
        raise ValueError("weights are too far apart to represent")
    return scaled
