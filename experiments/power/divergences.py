"""Level 2: every candidate last step on the pairwise divergences, against the package."""

from pathlib import Path

import numpy as np
from candidates import (
    DRAWS,
    NAMES,
    PACKAGE,
    READOUTS,
    Cell,
    Scores,
    readouts,
    score,
)
from common import (
    CAMPS_2,
    CAMPS_4,
    CLOUD,
    MAP_KS,
    TRIALS,
    Population,
    RealPopulation,
    error,
    fmt_k,
    interval,
    outscore,
    resamples,
    run_cells,
    table,
)

KS = tuple(float(k) for k in MAP_KS)
BASE = (CLOUD, CAMPS_2, CAMPS_4)

#: The trials the best k's growth is fitted over. Below 6, every readout's best k sits low.
GROWTH_FROM = 6


def _errors(data: list[list[dict]], readout: str) -> np.ndarray:
    """Return the weighted error at each number of trials (rows) and k (columns)."""
    return np.array(
        [
            error(cloud[readout].mean(0), (two[readout].mean(0) + four[readout].mean(0)) / 2)
            for cloud, two, four in data
        ]
    )


def chart(cache: Path | None = None) -> str:
    """Score every readout at 25 values of k, and find each one's best k."""
    cells = [Cell(p, n, KS, DRAWS, (501, i)) for n in TRIALS for i, p in enumerate(BASE)]
    raw = run_cells(cells, cache, "power-map")
    data = [[readouts(raw[3 * j + i], KS) for i in range(3)] for j in range(len(TRIALS))]
    errors = {readout: _errors(data, readout) for readout in READOUTS}
    rows = []
    for j, n in enumerate(TRIALS):
        row = [n]
        for readout in READOUTS:
            best = errors[readout][j].argmin()
            row.append(f"{fmt_k(KS[best])} / {errors[readout][j, best]:.3f}")
        rows.append(row)
    best = table(["Trials", *READOUTS], rows)
    late = np.array(TRIALS) >= GROWTH_FROM
    rows = []
    for readout, err in errors.items():
        fixed = err.mean(0).argmin()
        ks = np.array(KS)[err.argmin(1)]
        growth = np.polyfit(np.log(np.array(TRIALS)[late]), np.log(ks[late]), 1)[0]
        rows.append(
            [
                readout,
                f"{err.min(1).mean():.3f}",
                fmt_k(KS[fixed]),
                f"{err.mean(0)[fixed]:.3f}",
                f"{growth:.2f}",
            ]
        )
    summary = table(
        [
            "Readout",
            "At the best k for each n",
            "Best fixed k",
            "Error there",
            "Best k grows as n^",
        ],
        rows,
    )
    return (
        "The map. Every readout at 25 values of k from 10 to 40,000\n\n"
        f"Cloud, 2 and 4 tight camps over 4 options, {DRAWS} draws at each of 13 numbers of "
        "trials.\nEach cell is the best k for that readout, then its weighted error there.\n"
        + best
        + "\n\nThe mean weighted error over the 13 numbers of trials, and how fast the best k "
        f"grows,\nfitted from {GROWTH_FROM} trials.\n" + summary + "\n"
    )


def _pick(scores: Scores, population: Population | RealPopulation, n: int, name: str) -> np.ndarray:
    """Return one candidate's Cohesion per draw, for one population at one number of trials."""
    return scores[population.name][n][name]


def _rate(scores: Scores, n: int, name: str) -> float:
    """Return how often a camp draw outscores a cloud draw."""
    both = np.concatenate([_pick(scores, CAMPS_2, n, name), _pick(scores, CAMPS_4, n, name)])
    return _outscore(both, _pick(scores, CLOUD, n, name))


def rules(cache: Path | None = None) -> str:
    """Score every candidate at its rule on fresh seeds, against the package."""
    got = score(BASE, TRIALS, 601, cache, "power-rules")
    picks = {n: [resamples(DRAWS, (601, n, i)) for i in range(3)] for n in TRIALS}

    def errors(name: str, resample: int | None = None) -> np.ndarray:
        out = []
        for n in TRIALS:
            cloud, two, four = (_pick(got, p, n, name) for p in BASE)
            if resample is not None:
                cloud, two, four = (
                    values[pick[resample]]
                    for values, pick in zip((cloud, two, four), picks[n], strict=True)
                )
            out.append(error(cloud.mean(), (two.mean() + four.mean()) / 2))
        return np.array(out)

    resampled = {
        name: np.array([errors(name, r).mean() for r in range(len(picks[TRIALS[0]][0]))])
        for name in NAMES
    }
    bands = {"3 to 8": slice(0, 5), "10 to 25": slice(5, 9), "50 to 400": slice(9, None)}
    rows = []
    for name in NAMES:
        means = [np.mean([_pick(got, p, n, name).mean() for n in TRIALS]) for p in BASE]
        err = errors(name)
        low, high = interval((resampled[name] - resampled[PACKAGE.name])[:, None])
        rate = np.mean([_rate(got, n, name) for n in TRIALS])
        rows.append(
            [
                name,
                *(f"{mean:.3f}" for mean in means),
                f"{err.mean():.3f}",
                "" if name == PACKAGE.name else f"{low[0]:+.3f} to {high[0]:+.3f}",
                *(f"{err[band].mean():.3f}" for band in bands.values()),
                f"{rate:.3f}",
            ]
        )
    headers = ["Candidate", "Noise", "2 camps", "4 camps", "Error", "Less the package"]
    summary = table([*headers, *(f"Error, {band}" for band in bands), "Camps over noise"], rows)

    each = {name: errors(name) for name in NAMES}
    per_n = [[n, *(f"{each[name][j]:.3f}" for name in NAMES)] for j, n in enumerate(TRIALS)]
    rates = [[n, *(f"{_rate(got, n, name):.3f}" for name in NAMES)] for n in TRIALS]
    return (
        "The rules. Each candidate at its rule for k, on fresh seeds\n\n"
        f"Cloud, 2 and 4 tight camps over 4 options, {DRAWS} draws at each of 13 numbers of "
        "trials from 3 to 400.\nNoise and camps are mean readings over every number of trials. "
        "The error is the weighted error\nof experiments/k. Less the package is the 95% interval "
        "on the difference in error, from 1,000\nresamples of the same draws, so an interval "
        "below zero beats the package. Camps over noise is\nhow often a camp draw outscores a "
        "cloud draw.\n"
        + summary
        + "\n\nThe weighted error at each number of trials.\n"
        + table(["Trials", *NAMES], per_n)
        + "\n\nHow often a camp draw outscores a cloud draw, at each number of trials.\n"
        + table(["Trials", *NAMES], rates)
        + "\n"
    )


WIDE = (
    CLOUD,
    Population("1 camp", camps=1),
    Population("2 exact camps", camps=2, looseness=0.0),
    CAMPS_2,
    Population("2 camps, looseness 0.1", camps=2, looseness=0.1),
    Population("2 camps, looseness 0.3", camps=2, looseness=0.3),
    Population("2 camps, 9 to 1", camps=2, sizes="9 to 1"),
    Population("2 camps and a stray", camps=2, sizes="stray"),
    Population("2 camps, half loose", camps=2, sizes="half"),
    Population("3 tight camps", camps=3),
    CAMPS_4,
    Population("4 camps, looseness 0.1", camps=4, looseness=0.1),
    Population("4 camps, looseness 0.3", camps=4, looseness=0.3),
    Population("8 tight camps", camps=8),
    Population("cloud, spread 0.1", spread=0.1),
    Population("cloud, spread 0.3", spread=0.3),
    Population("cloud, spread 3", spread=3.0),
    Population("cloud, 2 options", options=2),
    Population("cloud, 16 options", options=16),
    Population("cloud, 30 observations", observations=30),
    Population("1 camp, 30 observations", camps=1, observations=30),
    Population("2 camps, 30 observations", camps=2, observations=30),
    Population("cloud, 100 observations", observations=100),
    Population("1 camp, 100 observations", camps=1, observations=100),
    Population("2 camps, 100 observations", camps=2, observations=100),
)
REAL = (
    RealPopulation("real values, cloud"),
    RealPopulation("real values, 1 camp", 1),
    RealPopulation("real values, 2 camps", 2),
    RealPopulation("real values, 4 camps", 4),
)
WIDE_TRIALS = (10, 50, 400)
REAL_TRIALS = (10, 50)

#: Pairs where the first should outscore the second.
PAIRS = (
    ("2 camps, looseness 0.3", "cloud"),
    ("4 camps, looseness 0.3", "cloud"),
    ("8 tight camps", "cloud"),
    ("2 camps, half loose", "cloud"),
    ("2 camps, 30 observations", "cloud, 30 observations"),
    ("real values, 4 camps", "real values, cloud"),
)

#: Populations that should read in this order, highest first.
ORDER = ("1 camp", "2 camps, 9 to 1", "2 tight camps", "3 tight camps", "4 tight camps")


def wide(cache: Path | None = None) -> str:
    """Score every candidate on more camps, looser camps, other noise and real values."""
    got = score(WIDE, WIDE_TRIALS, 602, cache, "power-wide")
    got.update(score(REAL, REAL_TRIALS, 603, cache, "power-real"))
    names = [p.name for p in (*WIDE, *REAL)]
    parts = [
        "Wider data. Each candidate at its rule for k\n\n"
        f"{DRAWS} draws of each population. A camp's looseness is how far its trials sit from "
        "its centre. Tight\nis 0.02. A cloud's spread is how widely it scatters its trials, and "
        "1 is uniform. Real values are\n500 draws a trial from a normal, measured as the package "
        "measures them."
    ]
    for n in WIDE_TRIALS:
        rows = [
            [name, *(f"{got[name][n][c].mean():.3f}" for c in NAMES)]
            for name in names
            if n in got[name]
        ]
        parts.append(f"The mean reading at {n} trials.\n" + table(["Population", *NAMES], rows))
    for n in REAL_TRIALS:
        rows = []
        for high, low in PAIRS:
            rows.append(
                [
                    f"{high} over {low}",
                    *(f"{_outscore(got[high][n][c], got[low][n][c]):.3f}" for c in NAMES),
                ]
            )
        parts.append(
            f"How often the first outscores the second, at {n} trials.\n"
            + table(["Pair", *NAMES], rows)
        )
    rows = []
    for n in WIDE_TRIALS:
        means = {c: [got[name][n][c].mean() for name in ORDER] for c in NAMES}
        rows.append(
            [f"In order at {n} trials", *("yes" if _falls(means[c]) else "no" for c in NAMES)]
        )
    for n in WIDE_TRIALS:
        rows.append(
            [
                f"2 tight camps at {n} trials",
                *(f"{got[CAMPS_2.name][n][c].mean():.3f}" for c in NAMES),
            ]
        )
    for size in (30, 100):
        name = f"1 camp, {size} observations"
        lowest = {c: min(got[name][n][c].mean() for n in WIDE_TRIALS) for c in NAMES}
        rows.append([f"Lowest for {name}", *(f"{lowest[c]:.3f}" for c in NAMES)])
    parts.append(
        "The two principles added for this experiment.\n"
        "One source reads near 1 at any number of observations. One camp, 2 camps at 9 to 1, 2, 3 "
        "and 4\ntight camps read in that order, with clean camps near the top. The lowest is over "
        f"{', '.join(map(str, WIDE_TRIALS))} trials.\n" + table(["Principle", *NAMES], rows)
    )
    return "\n\n".join(parts) + "\n"


def _outscore(high: np.ndarray, low: np.ndarray) -> float:
    """Return how often a draw of ``high`` outscores a draw of ``low``."""
    return float(outscore(high[:, None], low[:, None])[0])


def _falls(values: list[float]) -> bool:
    """Whether every value is below the one before it."""
    return bool(np.all(np.diff(values) < 0))
