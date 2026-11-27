"""Level 1: each trial's entropy, at power 1 as the package has it and at power 0.5.

Each test has a right answer that does not depend on the published figures.
Power 0.5 is run three ways, so it is not beaten on a technicality, and power 1
is run at ``k = 20`` too, so a lead for power 0.5 at ``k = 20`` can be put down
to the power or to the k.
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from common import table
from scipy.stats import beta, rankdata, spearmanr, truncnorm

import incoherence as ic
from incoherence._entropy import Array, densities, density_entropy, shannon
from incoherence._result import Result


@dataclass(frozen=True)
class Variant:
    """One way to turn observations in [0, 1] into an entropy.

    A grid that runs ``overhang`` past each end is the package's grid, which
    runs 0.5 past each end, with the data squeezed into its middle and k raised
    to match.
    """

    name: str
    power: float
    k: float = 10.0
    overhang: float = 0.5

    @property
    def squeeze(self) -> float:
        """Return how far the data is squeezed to fit the wider grid."""
        return 2.0 / (1.0 + 2.0 * self.overhang)

    def rows(self, trials: list[Array]) -> Array:
        """Return the densities of each trial."""
        k = self.k / self.squeeze
        return np.stack([densities(0.5 + (np.asarray(t) - 0.5) * self.squeeze, k) for t in trials])

    @cached_property
    def limits(self) -> tuple[float, float]:
        """Return V0 and V1, from an even spread and one repeated value on this grid."""
        even, point = self.rows([np.linspace(0, 1, 200_001), np.zeros(1)])
        return float((even**self.power).var()), float((point**self.power).var())

    def raw(self, rows: Array) -> Array:
        """Return the entropy of each row of densities, before it is clipped."""
        v0, v1 = self.limits
        return 1 - ((rows**self.power).var(axis=-1) - v0) / (v1 - v0)

    def entropy(self, rows: Array) -> Array:
        """Return the entropy of each row of densities."""
        return np.clip(self.raw(rows), 0.0, 1.0)

    def of(self, observations: Array) -> float:
        """Return the entropy of one trial."""
        return float(self.entropy(self.rows([observations]))[0])

    def incoherence(self, trials: list[Array]) -> float:
        """Return the incoherence of an ensemble, with equal weights."""
        rows = self.rows(trials)
        weights = np.full(len(rows), 1 / len(rows))
        return Result(rows, weights, self.entropy, 1.0).incoherence


PACKAGE = Variant("Power 1", 1.0)
VARIANTS = (
    PACKAGE,
    Variant("Power 1, k = 20", 1.0, k=20.0),
    Variant("Power 0.5", 0.5),
    Variant("Power 0.5, k = 20", 0.5, k=20.0),
    Variant("Power 0.5, wide grid", 0.5, overhang=1.5),
)
NAMES = [variant.name for variant in VARIANTS]

#: Evenly spaced quantiles, for shapes with no sampling noise.
QUANTILES = (np.arange(4000) + 0.5) / 4000


def _normal(centre: float, sd: float, q: Array = QUANTILES) -> Array:
    """Return a normal cut to [0, 1], as evenly spaced quantiles."""
    return truncnorm.ppf(q, -centre / sd, (1 - centre) / sd, loc=centre, scale=sd)


def _auc(signal: Array, noise: Array) -> float:
    """Return the chance that a score from ``signal`` beats one from ``noise``."""
    ranks = rankdata(np.concatenate([signal, noise]))
    n, m = len(signal), len(noise)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def _cells(values: list[float], places: int = 3) -> list[str]:
    return [f"{value:.{places}f}" for value in values]


def _check() -> str:
    """Check this file's power 1 is the package's."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        trials = [
            np.clip(rng.normal(rng.uniform(0.2, 0.8), rng.uniform(0.02, 0.2), 200), 0, 1)
            for _ in range(6)
        ]
        theirs = ic.measure(trials, bounds=(0, 1)).incoherence
        worst = max(worst, abs(PACKAGE.incoherence(trials) - theirs))
    half = VARIANTS[2]
    rows = half.rows([np.clip(rng.normal(0.5, 0.1, 500), 0, 1)])
    gap = abs(float(half.entropy(rows)[0]) - float(density_entropy(rows, 10.0, 0.5)[0]))
    assert worst < 1e-5 and gap < 1e-5, (worst, gap)
    return (
        "The check. Over 50 random ensembles, power 1 here gives the package's incoherence, and "
        "power 0.5\ngives the package's density_entropy at power 0.5, to within 1e-5. Its limits "
        "come from 200,001\nevenly spaced observations, where the package integrates exactly."
    )


def random_shape(rng: np.random.Generator, n: int) -> Array:
    """Return observations from a random mix of one to three humps, blocks and beta shapes."""
    parts = rng.integers(1, 4)
    out = []
    for share in rng.dirichlet(np.ones(parts)):
        m = max(1, int(round(n * share)))
        kind = rng.integers(3)
        if kind == 0:
            sd = np.exp(rng.uniform(np.log(0.01), np.log(0.3)))
            centre = rng.uniform(0, 1)
            out.append(
                truncnorm.rvs(
                    -centre / sd, (1 - centre) / sd, loc=centre, scale=sd, size=m, random_state=rng
                )
            )
        elif kind == 1:
            width = np.exp(rng.uniform(np.log(0.02), np.log(1.0)))
            low = rng.uniform(0, 1 - width)
            out.append(rng.uniform(low, low + width, m))
        else:
            a, b = np.exp(rng.uniform(np.log(0.5), np.log(8), 2))
            out.append(beta.rvs(a, b, size=m, random_state=rng))
    return np.concatenate(out)


BINS = (2, 3, 5, 10, 20, 50)


def _binned(trials: list[Array], bins: int) -> Array:
    """Return Shannon entropy of each trial in equal bins, as a share of its most."""
    counts = np.stack([np.histogram(t, np.linspace(0, 1, bins + 1))[0] for t in trials])
    return shannon(counts / counts.sum(axis=1, keepdims=True)) / np.log2(bins)


def _agreement(read: dict[str, Array], reference: dict[int, Array]) -> str:
    """Lay out rank agreement and the mean gap with the binned reference, at each number of bins."""
    rows = []
    for bins, values in reference.items():
        rows.append([f"{bins} bins, rank", *_cells([spearmanr(read[n], values)[0] for n in NAMES])])
    for bins, values in reference.items():
        rows.append(
            [f"{bins} bins, gap", *_cells([np.abs(read[n] - values).mean() for n in NAMES])]
        )
    return table(["Against", *NAMES], rows)


def _shannon() -> str:
    """Agreement with Shannon entropy on the same data, binned."""
    rng = np.random.default_rng(6)
    shapes = [random_shape(rng, 5000) for _ in range(500)]
    read = {v.name: v.entropy(v.rows(shapes)) for v in VARIANTS}
    each = _agreement(read, {bins: _binned(shapes, bins) for bins in BINS})
    wide = range(2, 61)
    gaps = np.array(
        [[np.abs(read[n] - _binned(shapes, bins)).mean() for n in NAMES] for bins in wide]
    )
    closest = table(
        ["", *NAMES],
        [
            ["Closest number of bins", *(str(wide[i]) for i in gaps.argmin(axis=0))],
            ["Mean gap there", *_cells(list(gaps.min(axis=0)))],
        ],
    )

    rng = np.random.default_rng(7)
    continuous: dict[str, list[float]] = {name: [] for name in NAMES}
    discrete: dict[int, list[float]] = {bins: [] for bins in BINS}
    for index in range(500):
        if index % 2:
            trials = [random_shape(rng, 2000) for _ in range(8)]
        else:
            base = random_shape(rng, 2000)
            move, stretch = rng.uniform(0, 0.15), rng.uniform(0, 0.4)
            trials = [
                np.clip(
                    0.5 + (base - 0.5) * np.exp(rng.normal(0, stretch)) + rng.normal(0, move), 0, 1
                )
                for _ in range(8)
            ]
        for v in VARIANTS:
            continuous[v.name].append(v.incoherence(trials))
        for bins in BINS:
            counts = np.stack([np.histogram(t, np.linspace(0, 1, bins + 1))[0] for t in trials])
            discrete[bins].append(ic.measure_counts(counts).incoherence)
    whole = _agreement(
        {name: np.array(values) for name, values in continuous.items()},
        {bins: np.array(values) for bins, values in discrete.items()},
    )
    return (
        "Agreement with Shannon entropy. The papers justify the estimator against Shannon "
        "entropy, binned,\nas a share of its most. Rank is the Spearman rank agreement, so 1 is "
        "the same order. Gap is the\nmean gap, so 0 is the same value.\n\n"
        "Each trial's entropy, over 500 random shapes of 5,000 observations: mixes of one to "
        "three humps,\nblocks and beta shapes.\n"
        + each
        + "\n\nThe one number of bins, from 2 to 60, that each sits closest to.\n"
        + closest
        + "\n\nIncoherence against the discrete measure on the same data binned, over 500 "
        "ensembles of 8 trials\nof 2,000 observations. In half, each trial is its own random "
        "shape. In the rest, the trials share\none shape, each moved and stretched a little.\n"
        + whole
    )


def _stray() -> str:
    """Move a share of a tight trial's observations far away, and see how far entropy rises."""
    n = 10_000
    base = _normal(0.3, 0.03, (np.arange(n) + 0.5) / n)
    rows = []
    for share in (0.001, 0.01, 0.05):
        moved = base.copy()
        moved[np.linspace(0, n - 1, int(n * share)).astype(int)] = 0.9
        rises = [v.of(moved) - v.of(base) for v in VARIANTS]
        binned = [float(_binned([moved], b)[0] - _binned([base], b)[0]) for b in (10, 50)]
        rows.append([f"{share:.1%}", *_cells(rises), *_cells(binned)])
    return (
        "Stray observations. A trial of Normal(0.3, 0.03), with a share of its observations "
        "moved to 0.9.\nEach cell is the rise in entropy. Shannon entropy on the same data in 10 "
        "and 50 bins gives the scale.\n"
        + table(["Share moved", *NAMES, "Shannon, 10 bins", "Shannon, 50 bins"], rows)
    )


SIZES = (5, 10, 20, 50, 200)


def _few() -> str:
    """Bias and spread when a trial holds few observations."""
    rng = np.random.default_rng(1)
    shapes: dict[str, tuple[Callable[[int], Array], Array]] = {
        "Normal(0.5, 0.1)": (
            lambda n: np.clip(rng.normal(0.5, 0.1, n), 0, 1),
            _normal(0.5, 0.1),
        ),
        "Even spread, 0.2 to 0.8": (lambda n: rng.uniform(0.2, 0.8, n), 0.2 + 0.6 * QUANTILES),
        "Two humps, 0.3 and 0.7, sd 0.05": (
            lambda n: np.clip(rng.normal(rng.choice([0.3, 0.7], n), 0.05), 0, 1),
            np.concatenate([_normal(0.3, 0.05), _normal(0.7, 0.05)]),
        ),
        "Even spread, 0 to 1": (lambda n: rng.uniform(0, 1, n), QUANTILES),
    }
    kept: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for draw, limit in shapes.values():
        for n in SIZES:
            samples = [draw(n) for _ in range(2000)]
            for v in VARIANTS:
                values = v.entropy(v.rows(samples))
                kept.setdefault((v.name, n), []).append(
                    (abs(values.mean() - v.of(limit)), values.std())
                )
    rows = []
    for n in SIZES:
        cells = [np.mean(kept[(name, n)], axis=0) for name in NAMES]
        rows.append([n, *(f"{bias:.3f} ({spread:.3f})" for bias, spread in cells)])
    return (
        "Few observations. Right answer: the same entropy however many observations a trial "
        "holds. Each cell\nis the size of the bias, then the spread over draws in brackets, as "
        "the mean over four shapes.\nThe bias is the mean over 2,000 draws less the value from "
        "4,000 evenly spaced quantiles.\n" + table(["Observations", *NAMES], rows)
    )


TRIALS = 10
REPS = 500
SD = 0.08

#: Each kind of difference, at three sizes.
KINDS = {
    "One trial, mean moved by 0.04, 0.08, 0.12": ("move", (0.04, 0.08, 0.12)),
    "One trial, sd times 1.2, 1.35, 1.5": ("sd", (1.2, 1.35, 1.5)),
    "One trial, sd times 0.8, 0.7, 0.6": ("sd", (0.8, 0.7, 0.6)),
    "Every trial, mean jitter sd 0.02, 0.03, 0.04": ("jitter", (0.02, 0.03, 0.04)),
    "Every trial, sd jitter 5%, 10%, 15%": ("spread", (0.05, 0.1, 0.15)),
    "Two camps, means 0.03, 0.05, 0.08 apart": ("camps", (0.03, 0.05, 0.08)),
    "One trial, 2%, 5%, 10% scattered over the range": ("scatter", (0.02, 0.05, 0.1)),
    "One trial, 50%, 75%, 100% an even spread, same mean and sd": ("flat", (0.5, 0.75, 1.0)),
}


def _ensemble(kind: str, size: float, rng: np.random.Generator, n: int) -> list[Array]:
    """Return one ensemble with a difference of this kind and size, or none."""
    if kind == "jitter":
        return [rng.normal(0.5 + rng.normal(0, size), SD, n) for _ in range(TRIALS)]
    if kind == "spread":
        return [rng.normal(0.5, SD * np.exp(rng.normal(0, size)), n) for _ in range(TRIALS)]
    if kind == "camps":
        return [rng.normal(0.5 + (size / 2 if i % 2 else -size / 2), SD, n) for i in range(TRIALS)]
    trials = [rng.normal(0.5, SD, n) for _ in range(TRIALS)]
    first = trials[0]
    if kind == "move":
        trials[0] = first + size
    elif kind == "sd":
        trials[0] = rng.normal(0.5, SD * size, n)
    elif kind == "scatter":
        trials[0] = np.where(rng.random(n) < size, rng.uniform(0.1, 0.9, n), first)
    elif kind == "flat":
        half = SD * np.sqrt(3)
        trials[0] = np.where(rng.random(n) < size, rng.uniform(0.5 - half, 0.5 + half, n), first)
    return trials


def _scores(kind: str, size: float, n: int, seed: int, fixed: bool) -> dict[str, Array]:
    """Return every variant's incoherence over ``REPS`` ensembles."""
    rng = np.random.default_rng(seed)
    out = {name: np.empty(REPS) for name in NAMES}
    for rep in range(REPS):
        trials = _ensemble(kind, size, rng, n)
        if fixed:
            trials = [np.clip(t, 0, 1) for t in trials]
        else:
            low, high = min(t.min() for t in trials), max(t.max() for t in trials)
            trials = [(t - low) / (high - low) for t in trials]
        for v in VARIANTS:
            out[v.name][rep] = v.incoherence(trials)
    return out


def _detection_table(job: tuple[bool, int]) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Return each variant's chance of telling each kind from none, and its reading of none."""
    fixed, n = job
    null = _scores("none", 0.0, n, 100 + n, fixed)
    found: dict[str, list[float]] = {name: [] for name in NAMES}
    for s, (kind, sizes) in enumerate(KINDS.values()):
        chances = {name: [] for name in NAMES}
        for j, size in enumerate(sizes):
            got = _scores(kind, size, n, 1000 * n + 10 * s + j, fixed)
            for name in NAMES:
                chances[name].append(_auc(got[name], null[name]))
        for name in NAMES:
            found[name].append(float(np.mean(chances[name])))
    return found, {name: float(null[name].mean()) for name in NAMES}


def _detection() -> str:
    """Telling a real difference from none."""
    jobs = [(fixed, n) for fixed in (True, False) for n in (50, 200)]
    with ProcessPoolExecutor() as pool:
        done = list(pool.map(_detection_table, jobs))
    rows = []
    for index, label in enumerate(KINDS):
        rows.append([label, *_cells([np.mean([d[0][n][index] for d in done]) for n in NAMES])])
    rows.append(["Mean of the eight", *_cells([np.mean([d[0][n] for d in done]) for n in NAMES])])
    floors = [
        [
            f"{n} observations, {'bounds 0 to 1' if fixed else 'bounds from the data'}",
            *_cells([null[name] for name in NAMES]),
        ]
        for (fixed, n), (_, null) in zip(jobs, done, strict=True)
    ]
    return (
        f"Detection. Each ensemble is {TRIALS} trials. With no difference, every trial is "
        f"Normal(0.5, {SD}). Each cell\nis the chance that an ensemble with the difference "
        f"scores a higher incoherence than one without,\nover {REPS} of each. 0.5 is a coin toss "
        "and 1 is perfect. Each is the mean over three sizes of the\ndifference and four "
        "tables: 50 and 200 observations a trial, with bounds at 0 and 1 or from the\ndata. "
        "Every variant scores the same draws, so a gap of 0.01 is about the noise.\n"
        + table(["Difference", *NAMES], rows)
        + "\n\nIncoherence with no difference, which should be low.\n"
        + table(["Table", *NAMES], floors)
    )


def report() -> str:
    """Return level 1 as text."""
    parts = [
        "Level 1. Each trial's entropy at power 1 and at power 0.5",
        "Power 1 is the package's estimator, at k = 10 on its grid. Power 0.5 is the same with "
        "the root of each\ndensity taken first. Its kernel then reaches twice as far, so the "
        "package's grid cuts its tail off.\n'k = 20' and 'wide grid', which runs 1.5 past each "
        "end, are the two repairs for that. Power 1 at\nk = 20 is the control for the first.",
        _check(),
        _shannon(),
        _stray(),
        _few(),
        _detection(),
    ]
    return "\n\n".join(parts) + "\n"
