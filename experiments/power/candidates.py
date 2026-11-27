"""Every candidate last step for Cohesion, read off three raw numbers per draw.

One pass over each draw keeps three numbers at every k: the variance of the
square root of the densities, the variance of the densities and the mean of
their square. Every candidate is then read off those, so all of them score the
same draws, and a new candidate costs nothing to add.

``run.py`` puts ``experiments`` and ``experiments/k`` on the import path, so
this module can share the k experiments' populations and statistics.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from common import SEED, Population, RealPopulation, fmt_k, run_cells

from incoherence._entropy import Array, densities, grid, points_for

#: Draws per cell.
DRAWS = 200

#: A raw entropy below this is rounding error, as in the package.
_ROUNDING = 1e-12


@cache
def limits(k: float) -> Array:
    """Return the raw numbers for an even spread (row 0) and one repeated value (row 1).

    These are V0 and V1 of the papers, computed as the package computes them.
    """
    r = grid(points_for(k))
    within, under, over = np.clip(r, 0, 1), np.minimum(r, 0), np.maximum(r, 1)
    inside = (2 - np.exp(-k * within) - np.exp(-k * (1 - within))) / k
    below = np.exp(k * under) * (1 - np.exp(-k)) / k
    above = np.exp(-k * (over - 1)) * (1 - np.exp(-k)) / k
    uniform = np.where(r < 0, below, np.where(r > 1, above, inside))
    point = np.exp(-k * np.abs(r))
    return np.array([raw(uniform), raw(point)])


def raw(density: Array) -> tuple[float, float, float]:
    """Return ``Var(√g)``, ``Var(g)`` and ``mean(g²)`` for one row of densities."""
    return float(np.sqrt(density).var()), float(density.var()), float((density**2).mean())


def raw_at(values: Array, k: float) -> tuple[float, float, float]:
    """Return the raw numbers for a set of divergences at this k."""
    return raw(densities(np.asarray(values, dtype=float), k, points_for(k)))


@dataclass(frozen=True)
class Cell:
    """One population at one number of trials, with raw numbers kept at a set of k."""

    population: Population | RealPopulation
    trials: int
    ks: tuple[float, ...]
    draws: int
    seed: tuple[int, ...]

    def run(self) -> Array:
        """Return the raw numbers, shaped ``(draws, ks, 3)``."""
        out = np.empty((self.draws, len(self.ks), 3))
        for draw in range(self.draws):
            rng = np.random.default_rng([SEED, *self.seed, self.trials, draw])
            values = self.population.divergences(rng, self.trials)
            out[draw] = [raw_at(values, k) for k in self.ks]
        return out


def _unit(entropy: Array) -> Array:
    """Clip an entropy to [0, 1] and drop rounding error, as the package does."""
    entropy = np.clip(entropy, 0.0, 1.0)
    return np.where(entropy < _ROUNDING, 0.0, entropy)


def entropies(scores: Array, ks: Sequence[float]) -> dict[str, Array]:
    """Return each entropy of the divergences, shaped ``(draws, ks)``, from raw numbers.

    ``power 0.5`` and ``power 1`` are Equation 4 of the Cohesion paper at each
    power. ``log count`` reads power 1 as a number of separate values: one
    repeated value gives 1 and an even spread gives the most, and the entropy
    is the log of that count over the log of the most.
    """
    both = np.stack([limits(float(k)) for k in ks])
    even, point = both[:, 0], both[:, 1]
    count = point[:, 2] / scores[..., 2]
    most = point[:, 2] / even[:, 2]
    return {
        "power 0.5": _unit(1 - (scores[..., 0] - even[:, 0]) / (point[:, 0] - even[:, 0])),
        "power 1": _unit(1 - (scores[..., 1] - even[:, 1]) / (point[:, 1] - even[:, 1])),
        "log count": _unit(np.log(count) / np.log(most)),
    }


#: Each candidate readout: the entropy it reads, and how it turns that into Cohesion.
READOUTS: dict[str, tuple[str, Callable[[Array], Array]]] = {
    "0.5 squared": ("power 0.5", lambda h: (1 - h) ** 2),
    "1 squared": ("power 1", lambda h: (1 - h) ** 2),
    "1 plain": ("power 1", lambda h: 1 - h),
    "1 doubled, capped": ("power 1", lambda h: np.minimum(1.0, 2 * (1 - h))),
    "1 - H squared": ("power 1", lambda h: 1 - h**2),
    "1 log squared": ("log count", lambda h: (1 - h) ** 2),
}


def readouts(scores: Array, ks: Sequence[float]) -> dict[str, Array]:
    """Return every candidate's Cohesion, shaped ``(draws, ks)``, from raw numbers."""
    read = entropies(scores, ks)
    return {name: turn(read[entropy]) for name, (entropy, turn) in READOUTS.items()}


Rule = Callable[[int], float]


@dataclass(frozen=True)
class Candidate:
    """A readout with a rule for its k."""

    readout: str
    rule: Rule
    k: str

    @property
    def name(self) -> str:
        """Return the short name tables use, such as ``0.5 squared, 8n``."""
        return f"{self.readout}, {self.k}"


def fixed(readout: str, k: float) -> Candidate:
    """Return a candidate at a fixed k."""
    return Candidate(readout, lambda n: k, fmt_k(k))


def growing(readout: str, rate: float) -> Candidate:
    """Return a candidate whose k is ``rate`` for each trial."""
    return Candidate(readout, lambda n: rate * n, f"{rate:g}n")


#: The package, first, then every rival on the same draws.
PACKAGE = growing("0.5 squared", 8.0)
CANDIDATES: tuple[Candidate, ...] = (
    PACKAGE,
    fixed("0.5 squared", 100.0),
    fixed("1 squared", 56.0),
    fixed("1 plain", 100.0),
    fixed("1 plain", 160.0),
    fixed("1 doubled, capped", 100.0),
    fixed("1 - H squared", 100.0),
    fixed("1 log squared", 200.0),
    growing("1 log squared", 20.0),
)
NAMES = [candidate.name for candidate in CANDIDATES]


def ks_for(trials: int, candidates: Iterable[Candidate] = CANDIDATES) -> tuple[float, ...]:
    """Return every k the candidates need at this number of trials."""
    return tuple(sorted({float(candidate.rule(trials)) for candidate in candidates}))


Scores = dict[str, dict[int, dict[str, Array]]]


def score(
    populations: Sequence[Population | RealPopulation],
    trials: Sequence[int],
    seed: int,
    cache: Path | None,
    tag: str,
) -> Scores:
    """Score every population at every number of trials under every candidate.

    Returns ``{population: {trials: {candidate: Cohesion per draw}}}``.
    """
    cells = [
        Cell(population, n, ks_for(n), DRAWS, (seed, index))
        for n in trials
        for index, population in enumerate(populations)
    ]
    out: Scores = {population.name: {} for population in populations}
    for cell, values in zip(cells, run_cells(cells, cache, tag), strict=True):
        read = readouts(values, cell.ks)
        out[cell.population.name][cell.trials] = {
            candidate.name: read[candidate.readout][:, cell.ks.index(candidate.rule(cell.trials))]
            for candidate in CANDIDATES
        }
    return out


def read(values: Array, trials: int) -> dict[str, float]:
    """Return every candidate's Cohesion for one set of divergences."""
    ks = ks_for(trials)
    got = readouts(np.array([[raw_at(values, k) for k in ks]]), ks)
    return {c.name: float(got[c.readout][0, ks.index(c.rule(trials))]) for c in CANDIDATES}


def trials_of(values: Array) -> int:
    """Return the number of trials behind a set of pairwise divergences."""
    return round((1 + (1 + 8 * len(values)) ** 0.5) / 2)
