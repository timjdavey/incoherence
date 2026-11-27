"""Shared setup for the k experiments: populations, scoring, rules and statistics."""

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from incoherence import measure
from incoherence._entropy import Array, densities, density_entropy, points_for, shannon
from incoherence._measures import COHESION_POWER, pairwise_divergences

#: Root seed. Every draw is seeded from this, the phase, the population and the trial count.
SEED = 20261127

#: Draws per cell, and in phase 2 where the tails decide.
DRAWS = 200
DRAWS_TAILS = 2000

#: Resamples behind every interval, and the share of them an interval covers.
RESAMPLES = 1000
INTERVAL = 95.0

#: How far a trial in a tight camp sits from its camp's centre. See ``Population``.
TIGHT = 0.02

#: The map: 25 values of k from 10 to 40,000, evenly spaced on a log scale. They are whole
#: numbers, as every rule gives, so the grid has a point at 0 as it does for the rules.
MAP_KS = np.round(np.geomspace(10.0, 40000.0, 25))

#: The k the package uses today.
FIXED_K = 100.0

#: How many times a false pattern counts against a missed one. See :func:`error`.
WEIGHT = 2.0

#: The weights the choice is checked at, to show how far it leans on the weight.
WEIGHTS = (1.0, 1.5, 2.0, 3.0, 4.0)

#: Trial counts for the phase 1 grid.
TRIALS = (3, 4, 5, 6, 8, 10, 13, 16, 25, 50, 100, 200, 400)

Rule = Callable[[int], float]

RULES: dict[str, Rule] = {
    "fixed 100": lambda n: FIXED_K,
    "4n": lambda n: 4.0 * n,
    "8n": lambda n: 8.0 * n,
    "20n": lambda n: 20.0 * n,
    "max(100, 8n)": lambda n: max(FIXED_K, 8.0 * n),
    "max(100, 20n)": lambda n: max(FIXED_K, 20.0 * n),
    "100 + 8n": lambda n: FIXED_K + 8.0 * n,
    "root(100^2 + (8n)^2)": lambda n: float(np.hypot(FIXED_K, 8.0 * n)),
}


@dataclass(frozen=True)
class Population:
    """A recipe for making draws.

    Every trial is a row of probabilities over ``options``. A cloud scatters
    each trial with a Dirichlet of concentration ``spread``, so 1 is uniform
    over all mixes, below 1 pushes trials to the corners and above 1 pulls
    them to the middle. A camp has a centre drawn uniformly over all mixes,
    and each of its trials is ``1 - looseness`` of that centre plus
    ``looseness`` of its own uniform mix. So 0 is an exact camp and 1 is a
    cloud. ``observations`` then replaces each row with the shares of that
    many observations drawn from it.
    """

    name: str
    camps: int = 0
    looseness: float = TIGHT
    options: int = 4
    spread: float = 1.0
    sizes: str = "even"
    observations: int | None = None

    def draw(self, rng: np.random.Generator, trials: int) -> Array:
        """Return one ensemble, as one row of probabilities per trial."""
        if self.camps == 0:
            rows = rng.dirichlet(np.full(self.options, self.spread), trials)
        else:
            own = rng.dirichlet(np.ones(self.options), trials)
            centres = rng.dirichlet(np.ones(self.options), self.camps)
            member, loose = self._members(trials)
            mix = np.where(loose, 1.0, self.looseness)[:, None]
            rows = (1 - mix) * centres[member] + mix * own
        if self.observations is not None:
            rows = rng.multinomial(self.observations, rows) / self.observations
        return rows

    def _members(self, trials: int) -> tuple[Array, Array]:
        """Return each trial's camp, and whether it sits outside every camp."""
        index = np.arange(trials)
        loose = np.zeros(trials, dtype=bool)
        if self.sizes == "even":
            member = index % self.camps
        elif self.sizes == "9 to 1":
            member = (index >= max(1, round(0.9 * trials))).astype(int)
            member[-1] = 1
        elif self.sizes == "stray":
            member = index % self.camps
            loose[-1] = True
        elif self.sizes == "half":
            member = index % self.camps
            loose[trials - trials // 2 :] = True
        else:
            raise ValueError(f"unknown sizes: {self.sizes}")
        return member, loose

    def divergences(self, rng: np.random.Generator, trials: int) -> Array:
        """Return the pairwise divergences of one ensemble."""
        return divergences(self.draw(rng, trials))


@dataclass(frozen=True)
class RealPopulation:
    """A recipe for draws of real-valued observations, measured as the package measures them.

    Every trial is a normal sample. A cloud gives each trial its own mean,
    uniform from -4 to 4, and its own spread, uniform from 0.3 to 2. A camp
    has one such mean and spread, shared by all its trials, so its trials
    differ only by sampling.
    """

    name: str
    camps: int = 0
    observations: int = 500

    def divergences(self, rng: np.random.Generator, trials: int) -> Array:
        """Return the pairwise divergences of one ensemble, at the default observation k."""
        count = self.camps or trials
        means, spreads = rng.uniform(-4, 4, count), rng.uniform(0.3, 2, count)
        member = np.arange(trials) % count
        samples = [rng.normal(means[m], spreads[m], self.observations) for m in member]
        return np.asarray(measure(samples).divergences)


CLOUD = Population("cloud")
CAMPS_2 = Population("2 tight camps", camps=2)
CAMPS_4 = Population("4 tight camps", camps=4)


def divergences(rows: Array) -> Array:
    """Pairwise divergences between rows of probabilities, with equal weights."""
    return pairwise_divergences(rows, shannon, np.full(len(rows), 1 / len(rows)))


def entropy_read(values: Array, k: float, points: int | None = None) -> float:
    """Entropy of a set of divergences, as Cohesion's last step reads it at this k."""
    points = points_for(k) if points is None else points
    return float(density_entropy(densities(values, k, points), k, COHESION_POWER))


def cohesion_at(values: Array, k: float, points: int | None = None) -> float:
    """Cohesion from a set of divergences at an explicit k."""
    return (1 - entropy_read(values, k, points)) ** 2


def error(noise: Array, camps: Array, weight: float = WEIGHT) -> Array:
    """Return the weighted error, from what noise reads and what tight camps read.

    Noise should read 0, so its score is its error. Tight camps should read 1,
    so their error is what they fall short by. A false pattern counts
    ``weight`` times a missed one, and the result is scaled so that 0 is
    perfect and 1 is as wrong as a rule can be.
    """
    return (weight * noise + 1 - camps) / (weight + 1)


def rule_ks(trials: int, rules: Iterable[str]) -> dict[str, float]:
    """Return the k each named rule gives at this number of trials."""
    return {name: float(RULES[name](trials)) for name in rules}


@dataclass(frozen=True)
class Cell:
    """One population at one number of trials, scored at a set of k."""

    population: Population | RealPopulation
    trials: int
    ks: tuple[float, ...]
    draws: int
    seed: tuple[int, ...]

    def run(self) -> Array:
        """Return Cohesion for every draw (rows) at every k (columns)."""
        out = np.empty((self.draws, len(self.ks)))
        for draw in range(self.draws):
            rng = np.random.default_rng([SEED, *self.seed, self.trials, draw])
            values = self.population.divergences(rng, self.trials)
            out[draw] = [cohesion_at(values, k) for k in self.ks]
        return out


def _run(cell: Cell) -> Array:
    return cell.run()


def run_cells(cells: Sequence[Cell], cache: Path | None = None, tag: str = "") -> list[Array]:
    """Score every cell, in parallel. The result does not depend on the worker count.

    With ``cache`` set, scores are kept there under ``tag`` and reused when the
    cells are unchanged. It is only for working on the analysis. A ``--check``
    run never reads it.
    """
    path = None if cache is None else cache / f"{tag}.npz"
    key = repr(cells)
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as kept:
            if str(kept["key"]) == key:
                return [kept[f"cell{i}"] for i in range(len(cells))]
    order = sorted(range(len(cells)), key=lambda i: -cells[i].trials * cells[i].draws)
    with ProcessPoolExecutor() as pool:
        done = list(pool.map(_run, [cells[i] for i in order]))
    out: list[Array] = [np.empty(0)] * len(cells)
    for i, scores in zip(order, done, strict=True):
        out[i] = scores
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, key=key, **{f"cell{i}": scores for i, scores in enumerate(out)})
    return out


def resamples(draws: int, seed: Sequence[int]) -> Array:
    """Return the draws picked by each resample, one row per resample."""
    rng = np.random.default_rng([SEED, 999, *seed])
    return rng.integers(0, draws, (RESAMPLES, draws))


def outscore(high: Array, low: Array) -> Array:
    """How often a draw of ``high`` outscores a draw of ``low``, over every pairing.

    Rows are draws and columns stay separate. A dead heat counts as half.
    """
    count = len(high)
    ranks = rankdata(np.concatenate([high, low]), axis=0)
    wins = ranks[:count].sum(axis=0) - count * (count + 1) / 2
    return wins / (count * len(low))


#: The measures behind principles 1 to 4, and whether a higher value is the better one.
MEASURES = {
    "error": False,
    "camps mean": True,
    "cloud 95th": False,
    "2 camps 5th": True,
    "4 camps 5th": True,
    "order": True,
    "rate": True,
    "rate 2": True,
    "rate 4": True,
    "gap": True,
    "cloud mean": False,
    "2 camps mean": True,
    "4 camps mean": True,
}


def measures(cloud: Array, two: Array, four: Array) -> dict[str, Array]:
    """Return every measure for each column of k.

    Each argument holds one population's scores, with one row per draw. A camp
    draw is a draw from either camp population, so ``rate`` and ``gap`` pool
    the two.
    """
    both = np.concatenate([two, four])
    p95 = np.percentile(cloud, 95, axis=0)
    return {
        "error": error(cloud.mean(axis=0), both.mean(axis=0)),
        "camps mean": both.mean(axis=0),
        "cloud 95th": p95,
        "2 camps 5th": np.percentile(two, 5, axis=0),
        "4 camps 5th": np.percentile(four, 5, axis=0),
        "order": outscore(two, four),
        "rate": outscore(both, cloud),
        "rate 2": outscore(two, cloud),
        "rate 4": outscore(four, cloud),
        "gap": np.percentile(both, 5, axis=0) - p95,
        "cloud mean": cloud.mean(axis=0),
        "2 camps mean": two.mean(axis=0),
        "4 camps mean": four.mean(axis=0),
    }


@dataclass(frozen=True)
class Scored:
    """Every measure at one number of trials, with resampled values for calling ties."""

    point: dict[str, Array]
    resampled: dict[str, Array]

    def difference(self, name: str, column: int) -> tuple[Array, Array]:
        """Return the interval on each column's measure, less the same measure at ``column``."""
        values = self.resampled[name]
        return interval(values - values[:, [column]])

    def ties(self, name: str, column: int) -> Array:
        """Return whether each column ties with ``column`` on a measure."""
        low, high = self.difference(name, column)
        return (low <= 0) & (high >= 0)

    def worse(self, name: str, column: int) -> Array:
        """Return whether each column is worse than ``column`` by more than a tie."""
        low, high = self.difference(name, column)
        return high < 0 if MEASURES[name] else low > 0

    def better(self, name: str, column: int) -> Array:
        """Return whether each column is better than ``column`` by more than a tie."""
        low, high = self.difference(name, column)
        return low > 0 if MEASURES[name] else high < 0

    def mark(self, name: str, column: int) -> Array:
        """Return ``+``, ``=`` or ``-`` for each column against ``column``."""
        return np.where(
            self.better(name, column), "+", np.where(self.worse(name, column), "-", "=")
        )


def judge(scores: tuple[Array, Array, Array], seed: tuple[int, ...]) -> Scored:
    """Score cloud, 2 camps and 4 camps on every measure, with the resamples behind ties.

    Every column scores the same draws, and every resample picks the same
    draws for every column. So two columns are compared draw by draw and the
    luck of the draw cancels.
    """
    picks = [resamples(len(scores[0]), (*seed, i)) for i in range(3)]
    kept: dict[str, list[Array]] = {name: [] for name in MEASURES}
    for resample in range(RESAMPLES):
        again = measures(
            *(values[pick[resample]] for values, pick in zip(scores, picks, strict=True))
        )
        for name, values in again.items():
            kept[name].append(values)
    return Scored(measures(*scores), {name: np.stack(values) for name, values in kept.items()})


def _judge(args: tuple[tuple[Array, Array, Array], tuple[int, ...]]) -> Scored:
    return judge(*args)


def judge_all(jobs: Sequence[tuple[tuple[Array, Array, Array], tuple[int, ...]]]) -> list[Scored]:
    """Judge several numbers of trials at once, in parallel."""
    with ProcessPoolExecutor() as pool:
        return list(pool.map(_judge, jobs))


def interval(differences: Array) -> tuple[Array, Array]:
    """Return the middle 95% of each column of resampled differences."""
    edge = (100 - INTERVAL) / 2
    low, high = np.percentile(differences, [edge, 100 - edge], axis=0)
    return low, high


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Lay rows out as a plain text table, with every column right-aligned but the first."""
    cells = [[str(value) for value in row] for row in rows]
    widths = [max(len(row[i]) for row in [list(headers), *cells]) for i in range(len(headers))]
    lines = []
    for row in [list(headers), ["-" * width for width in widths], *cells]:
        rest = [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  ".join([row[0].ljust(widths[0]), *rest]).rstrip())
    return "\n".join(lines)


def fmt_k(k: float) -> str:
    """Write a k to three significant figures, without an exponent."""
    return f"{float(f'{k:.3g}'):,.10g}"
