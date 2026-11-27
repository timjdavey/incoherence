"""Kinds of noise across k, and the noise ceiling."""

from math import sqrt
from pathlib import Path

import numpy as np
from common import (
    DRAWS,
    FIXED_K,
    MAP_KS,
    TRIALS,
    Array,
    Cell,
    Population,
    fmt_k,
    interval,
    outscore,
    resamples,
    rule_ks,
    run_cells,
    table,
)

#: Seeds the draws. Changing it moves spread.txt and ceiling.txt.
PHASE = 5
WINNER = "8n"
RULES = (WINNER, "fixed 100", "20n")
SPREADS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
OPTIONS = (2, 4, 16, 64)

#: A trial sits in a corner when it puts more than this share on one option.
CORNER = 0.9

#: A coarser sweep than the map, to keep the run short: every other k, up to about 5,000.
SWEEP = MAP_KS[::2][:10]


def _tails(scores: Array, reference: int, seed: tuple[int, ...]) -> tuple[Array, Array]:
    """Return the 95th percentile of each column, and how it compares with the reference column.

    A plus reads lower than the reference by more than a tie, a minus reads
    higher by more than a tie and an equals sign is a tie.
    """
    again = np.percentile(scores[resamples(len(scores), seed)], 95, axis=1)
    low, high = interval(again - again[:, [reference]])
    marks = np.where(high < 0, "+", np.where(low > 0, "-", "="))
    return np.percentile(scores, 95, axis=0), marks


def _noise() -> list[tuple[str, Population]]:
    """Return every kind of exact noise: clouds at each spread and each number of options."""
    spreads = [(f"spread {s:g}", Population(f"cloud, spread {s:g}", spread=s)) for s in SPREADS]
    options = [
        (f"{o} options", Population(f"cloud, options {o}", options=o)) for o in OPTIONS if o != 4
    ]
    return spreads + options


def spread(cache: Path | None = None) -> str:
    """Return the 95th percentile of every kind of exact noise across k, as text."""
    listed = _noise()
    cells = []
    for index, (_, population) in enumerate(listed):
        for trials in TRIALS:
            ks = tuple(sorted({*SWEEP, *rule_ks(trials, RULES).values()}))
            cells.append(Cell(population, trials, ks, DRAWS, (PHASE, 1, index)))
    done = run_cells(cells, cache, "spread")
    summary, maps = [], {name: [] for name, _ in listed}
    for index, (name, population) in enumerate(listed):
        sample = population.draw(np.random.default_rng([PHASE, 2, index]), 20000)
        corner = (sample.max(axis=1) > CORNER).mean()
        for cell, scores in zip(cells, done, strict=True):
            if cell.population is not population:
                continue
            ks = np.array(cell.ks)
            at = {
                rule: int(np.flatnonzero(ks == k)[0])
                for rule, k in rule_ks(cell.trials, RULES).items()
            }
            p95, marks = _tails(scores, at[WINNER], (PHASE, 1, index, cell.trials))
            on_map = np.isin(ks, SWEEP)
            lowest = ks[on_map][np.argmin(p95[on_map])]
            row = [name, f"{corner:.0%}", cell.trials, fmt_k(lowest), f"{p95[at[WINNER]]:.3f}"]
            row += [f"{p95[at[rule]]:.3f} {marks[at[rule]]}" for rule in RULES[1:]]
            summary.append(row)
            maps[name].append([cell.trials, *(f"{value:.2f}" for value in p95[on_map])])
    headers = ["Trials", *(fmt_k(k) for k in SWEEP)]
    parts = [
        "Kinds of noise across k",
        f"Every kind of exact noise, {DRAWS} draws per cell. Spread is the concentration of the "
        "Dirichlet that\nscatters the trials over 4 options, and 1 is the cloud the rule was "
        "chosen on. In a corner is the share\nof trials that put more than "
        f"{CORNER:.0%} on one option.",
        f"The k at which the 95th percentile is lowest, on a sweep of {len(SWEEP)} values from "
        f"{fmt_k(SWEEP[0])} to {fmt_k(SWEEP[-1])}, then the 95th\npercentile under each rule. "
        f"The sign compares a rule with {WINNER} on the same draws. A plus reads lower by\nmore "
        "than a tie, a minus reads higher by more than a tie and an equals sign is a tie.\n"
        + table(["Noise", "In a corner", "Trials", "Lowest at", *RULES], summary),
        *(
            f"The 95th percentile at {name}.\n" + table(headers, rows)
            for name, rows in maps.items()
        ),
    ]
    return "\n\n".join(parts) + "\n"


CEILING_TRIALS = (10, 25, 50, 100, 400)
OBSERVATIONS = (10, 30, 100, 1000)
CEILING_OPTIONS = (4, 16)
CEILING_RULES = (WINNER, "fixed 100", "ceiling", "ceiling, 100 or more")


def ceiling_ks(trials: int, observations: int, options: int) -> dict[str, float]:
    """Return the k each rule gives, where two of them cap k at the noise ceiling.

    The ceiling is ``2m / root(s - 1)``. It can sit far below 100, so the
    second capped rule never lets the ceiling take k under 100.
    """
    limit = 2 * observations / sqrt(options - 1)
    grown = 8.0 * trials
    return {
        WINNER: grown,
        "fixed 100": FIXED_K,
        "ceiling": float(round(min(grown, limit))),
        "ceiling, 100 or more": float(round(min(grown, max(FIXED_K, limit)))),
    }


def ceiling(cache: Path | None = None) -> str:
    """Return the noise ceiling against the winner and fixed 100, as text."""
    kinds = (("cloud", 0), ("one source", 1), ("2 tight camps", 2), ("4 tight camps", 4))
    cells, keys = [], []
    for options in CEILING_OPTIONS:
        for size in OBSERVATIONS:
            for kind, camps in kinds:
                changes = {"looseness": 0.0} if camps == 1 else {}
                name = f"{kind}, options {options}, observations {size}"
                population = Population(
                    name, camps=camps, options=options, observations=size, **changes
                )
                for trials in CEILING_TRIALS:
                    ks = tuple(sorted(set(ceiling_ks(trials, size, options).values())))
                    cells.append(Cell(population, trials, ks, DRAWS, (PHASE, 3, len(keys))))
                    keys.append((options, size, kind, trials))
    scores = dict(zip(keys, run_cells(cells, cache, "ceiling"), strict=True))
    columns = {key: cell.ks for key, cell in zip(keys, cells, strict=True)}

    def read(options: int, size: int, kind: str, trials: int) -> Array:
        """Return one row per draw and one column per rule."""
        ks = ceiling_ks(trials, size, options)
        at = [columns[options, size, kind, trials].index(k) for k in ks.values()]
        return scores[options, size, kind, trials][:, at]

    rows = []
    tally = {rule: {"cloud": [0, 0, 0], "camps": [0, 0, 0]} for rule in CEILING_RULES[1:]}
    for options in CEILING_OPTIONS:
        for size in OBSERVATIONS:
            for kind, camps in kinds:
                for trials in CEILING_TRIALS:
                    values = read(options, size, kind, trials)
                    noise = camps < 2
                    seed = (PHASE, 4, options, size, camps, trials)
                    picks = resamples(DRAWS, seed)
                    percent = 95 if noise else 5
                    tails = np.percentile(values, percent, axis=0)
                    again = np.percentile(values[picks], percent, axis=1)
                    low, high = interval(again - again[:, [0]])
                    better, worse = (high < 0, low > 0) if noise else (low > 0, high < 0)
                    marks = np.where(better, "+", np.where(worse, "-", "="))
                    ks = ceiling_ks(trials, size, options)
                    row = [options, f"{size:,}", kind, trials]
                    row += [fmt_k(k) for name, k in ks.items() if name != "fixed 100"]
                    row.append(f"{tails[0]:.3f}")
                    row += [f"{tails[i]:.3f} {marks[i]}" for i in range(1, 4)]
                    if noise:
                        row.append("")
                    else:
                        cloud = read(options, size, "cloud", trials)
                        row.append("  ".join(f"{rate:.3f}" for rate in outscore(values, cloud)))
                    rows.append(row)
                    if kind == "one source":
                        continue
                    for i, rule in enumerate(CEILING_RULES[1:], 1):
                        tally[rule]["cloud" if noise else "camps"]["+=-".index(marks[i])] += 1
    counts = [
        [rule, *(" / ".join(map(str, tally[rule][kind])) for kind in ("cloud", "camps"))]
        for rule in CEILING_RULES[1:]
    ]
    parts = [
        "The noise ceiling",
        f"Trials of finite observations, {DRAWS} draws per cell. The ceiling is 2m / root(s - 1) "
        "for m observations\nover s options. Ceiling is min(8n, ceiling). Ceiling, 100 or more "
        "is min(8n, max(100, ceiling)).",
        "The ceiling.\n"
        + table(
            ["Options", *(f"{size:,} observations" for size in OBSERVATIONS)],
            [
                [options, *(f"{2 * size / sqrt(options - 1):,.0f}" for size in OBSERVATIONS)]
                for options in CEILING_OPTIONS
            ],
        ),
        "The tail is the 95th percentile for the cloud and one source, and the 5th for camps. "
        f"The sign compares\na rule with {WINNER} on the same draws. A plus is better by more "
        "than a tie, which is lower for the\ncloud and one source and higher for camps. Rate is "
        "how often a camp draw outscores a cloud with the\nsame observations, under each rule "
        "in turn.\n"
        + table(
            [
                "Options",
                "Observations",
                "Population",
                "Trials",
                f"k, {WINNER}",
                "Ceiling",
                "100 or more",
                f"Tail at {WINNER}",
                "Fixed 100",
                "Ceiling",
                "100 or more",
                "Rate",
            ],
            rows,
        ),
        f"Each rule against {WINNER}. The clouds, then the camps, on which its tail is better, "
        "tied and worse.\n" + table(["Rule", "Clouds", "Camps"], counts),
    ]
    return "\n\n".join(parts) + "\n"
