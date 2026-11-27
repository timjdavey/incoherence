"""Phase 1: check the formulas, then map every draw across k to find the best k."""

from math import log, sqrt
from pathlib import Path

import numpy as np
from common import (
    CAMPS_2,
    CAMPS_4,
    CLOUD,
    DRAWS,
    FIXED_K,
    MAP_KS,
    TIGHT,
    TRIALS,
    WEIGHT,
    Array,
    Cell,
    Population,
    Scored,
    divergences,
    entropy_read,
    fmt_k,
    judge_all,
    rule_ks,
    run_cells,
    table,
)

from incoherence._entropy import shannon

PHASE = 1
MAP_RULES = ("fixed 100", "4n", "8n", "20n")
POPULATIONS = (CLOUD, CAMPS_2, CAMPS_4)

#: How close to the best k counts as near it, on the weighted error.
NEAR = 0.005
LAW_KS = (24.0, 48.0, 100.0, 400.0, 1600.0, 6400.0)
LAW_SETS = 100


def laws() -> str:
    """Return the formula check as text."""
    parts = [
        "Phase 1. The formula check",
        _separate_values(),
        _random_values(),
        _exact_camps(),
        _cloud_pairs(),
        _sampling_noise(),
    ]
    return "\n\n".join(parts) + "\n"


def _separate_values() -> str:
    """Entropy read for a few exactly separate values, against ``8(m - 1) / (k - 8)``."""
    cases: list[tuple[str, Array, Array]] = [
        (f"{m} values, equal shares", np.linspace(0.1, 0.9, m), np.full(m, 100)) for m in (2, 3, 7)
    ]
    cases += [
        ("2 values, shares 82 and 18", np.array([0.1, 0.9]), np.array([82, 18])),
        (
            "7 values, shares 2, 1, 1, 1, 1, 1, 1",
            np.linspace(0.1, 0.9, 7),
            25 * np.array([2] + [1] * 6),
        ),
    ]
    rows = []
    for name, values, counts in cases:
        m = len(values)
        shares = counts / counts.sum()
        effective = np.sqrt(shares).sum() ** 2
        for k in LAW_KS:
            read = entropy_read(np.repeat(values, counts), k)
            rows.append(
                [
                    name,
                    fmt_k(k),
                    f"{read:.4f}",
                    f"{min(1.0, 8 * (m - 1) / (k - 8)):.4f}",
                    f"{min(1.0, 8 * (effective - 1) / (k - 8)):.4f}",
                ]
            )
    head = (
        "Formula 1. Exactly separate values. The entropy read, against 8(m - 1) / (k - 8).\n"
        "The last column swaps m for (sum of root shares) squared, which allows unequal shares.\n"
    )
    return head + table(["Values", "k", "Read", "Formula", "With shares"], rows)


def _random_values() -> str:
    """Entropy read for N uniform random values, against ``1 - k / (16N)``."""
    rows = []
    for count in (3, 10, 45, 300, 4950, 79800):
        rng = np.random.default_rng([PHASE, 2, count])
        sets = rng.uniform(0, 1, (LAW_SETS, count))
        row = [f"{count:,}"]
        for k in LAW_KS:
            read = np.mean([entropy_read(values, k) for values in sets])
            row.append(f"{read:.3f} / {max(0.0, 1 - k / (16 * count)):.3f}")
        rows.append(row)
    head = (
        f"Formula 2. N random values. Mean entropy read over {LAW_SETS} sets, then 1 - k / (16N).\n"
    )
    return head + table(["N", *(f"k = {fmt_k(k)}" for k in LAW_KS)], rows)


def _exact_camps() -> str:
    """Check formula 1 on real pairwise divergences, from exact camps with random centres."""
    rows = []
    for camps, values in ((2, 2), (4, 7)):
        population = Population(f"{camps} exact camps", camps=camps, looseness=0.0)
        for trials in (16, 100):
            pairs = trials * (trials - 1) / 2
            sizes = np.bincount(np.arange(trials) % camps)
            within = (sizes * (sizes - 1) / 2).sum() / pairs
            between = [
                sizes[a] * sizes[b] / pairs for a in range(camps) for b in range(a + 1, camps)
            ]
            effective = np.sqrt([within, *between]).sum() ** 2
            for k in (FIXED_K, 8.0 * trials):
                reads = []
                for draw in range(DRAWS):
                    rng = np.random.default_rng([PHASE, 3, camps, trials, draw])
                    reads.append(entropy_read(divergences(population.draw(rng, trials)), k))
                rows.append(
                    [
                        population.name,
                        trials,
                        fmt_k(k),
                        f"{np.mean(reads):.4f}",
                        f"{np.median(reads):.4f}",
                        f"{min(1.0, 8 * (values - 1) / (k - 8)):.4f}",
                        f"{min(1.0, 8 * (effective - 1) / (k - 8)):.4f}",
                    ]
                )
    head = (
        "Formula 1 on real pairs. Exact camps with random centres, "
        f"{DRAWS} draws.\nTwo camps give 2 separate values and four give 7.\n"
    )
    headers = ["Population", "Trials", "k", "Mean read", "Median read", "Formula", "With shares"]
    return head + table(headers, rows)


def _cloud_pairs() -> str:
    """How far a cloud's entropy falls short of its many-trial value, against ``k / (16N)``."""
    counts = (10, 25, 50, 100, 200, 400)
    ks = sorted({FIXED_K, *(8.0 * trials for trials in counts)})
    big = divergences(CLOUD.draw(np.random.default_rng([PHASE, 4]), 3000))
    limit = {k: entropy_read(big, k) for k in ks}
    rows = []
    for trials in counts:
        pairs = trials * (trials - 1) / 2
        draws = [
            divergences(CLOUD.draw(np.random.default_rng([PHASE, 5, trials, draw]), trials))
            for draw in range(DRAWS)
        ]
        for k in (FIXED_K, 8.0 * trials):
            short = limit[k] - np.mean([entropy_read(values, k) for values in draws])
            formula = k / (16 * pairs)
            rows.append(
                [
                    trials,
                    f"{pairs:,.0f}",
                    fmt_k(k),
                    f"{limit[k]:.4f}",
                    f"{short:.4f}",
                    f"{formula:.4f}",
                    f"{short / formula:.1f}",
                ]
            )
    head = (
        "Formula 2 on real pairs. A cloud's entropy read, short of the same cloud at 3,000 "
        f"trials.\nMean of {DRAWS} draws. The formula's shortfall is k / (16N).\n"
    )
    headers = ["Trials", "Pairs", "k", "At 3,000 trials", "Shortfall", "Formula", "Ratio"]
    return head + table(headers, rows)


def _pair_divergences(first: Array, second: Array) -> Array:
    """Divergence between matching rows of two tables of probabilities."""
    return shannon((first + second) / 2) - (shannon(first) + shannon(second)) / 2


def _sampling_noise() -> str:
    """Divergence between two samples, within one source and between two."""
    pairs = 2000
    rows = []
    for options in (2, 4, 16, 64):
        source = np.full(options, 1 / options)
        for size in (10, 30, 100, 1000):
            rng = np.random.default_rng([PHASE, 6, options, size])
            first = rng.multinomial(size, source, pairs) / size
            second = rng.multinomial(size, source, pairs) / size
            values = _pair_divergences(first, second)
            rows.append(
                [
                    options,
                    f"{size:,}",
                    f"{values.mean():.4f}",
                    f"{(options - 1) / (4 * size * log(2)):.4f}",
                    f"{values.std():.4f}",
                    f"{sqrt(2 * (options - 1)) / (4 * size * log(2)):.4f}",
                    f"{2 * size / sqrt(options - 1):,.0f}",
                ]
            )
    within = table(
        ["Options", "Observations", "Mean", "Formula", "Spread", "Formula", "Ceiling"], rows
    )

    one, other = np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.1, 0.2, 0.3, 0.4])
    sizes = (30, 100, 1000, 10000)
    rows, spreads = [], {"within": [], "between": []}
    for size in sizes:
        rng = np.random.default_rng([PHASE, 7, size])
        a, b = (rng.multinomial(size, one, pairs) / size for _ in range(2))
        c = rng.multinomial(size, other, pairs) / size
        inside, across = _pair_divergences(a, b), _pair_divergences(a, c)
        spreads["within"].append(inside.std())
        spreads["between"].append(across.std())
        rows.append(
            [f"{size:,}", f"{inside.std():.5f}", f"{across.mean():.4f}", f"{across.std():.5f}"]
        )
    powers = {
        name: np.polyfit(np.log(sizes), np.log(values), 1)[0] for name, values in spreads.items()
    }
    between = table(["Observations", "Spread within", "Mean between", "Spread between"], rows)
    return (
        f"Formulas 3 and 4. Two trials from one uniform source, {pairs:,} pairs.\n"
        "The ceiling is 2m / root(s - 1).\n"
        f"{within}\n\n"
        "Noise between two camps. Sources (0.4, 0.3, 0.2, 0.1) and (0.1, 0.2, 0.3, 0.4).\n"
        f"{between}\n"
        f"Spread within one source falls as observations to the power {powers['within']:.2f}.\n"
        f"Spread between two camps falls as observations to the power {powers['between']:.2f}."
    )


def map_cells() -> list[Cell]:
    """Return every cell of the map: three populations at every number of trials."""
    cells = []
    for index, population in enumerate(POPULATIONS):
        for trials in TRIALS:
            ks = sorted({*MAP_KS, *rule_ks(trials, MAP_RULES).values()})
            cells.append(Cell(population, trials, tuple(ks), DRAWS, (PHASE, index)))
    return cells


def band(scored: Scored, candidates: Array) -> tuple[int, Array, Array]:
    """Return the best column among the candidates, with its band and what is near it.

    The best k has the lowest weighted error. The band is every k that ties
    with it. Near is looser. It takes every k within ``NEAR`` of the best,
    whether or not the difference could be chance.
    """
    error = scored.point["error"]
    best = int(np.argmin(np.where(candidates, error, np.inf)))
    return best, scored.ties("error", best), error <= error[best] + NEAR


def _span(ks: Array) -> str:
    return fmt_k(ks.min()) if ks.min() == ks.max() else f"{fmt_k(ks.min())} to {fmt_k(ks.max())}"


def chart(cache: Path | None = None) -> str:
    """Return the map and what it says about the best k, as text."""
    cells = map_cells()
    names = [(cell.population.name, cell.trials) for cell in cells]
    scores = dict(zip(names, run_cells(cells, cache, "map"), strict=True))
    jobs = [
        (tuple(scores[population.name, trials] for population in POPULATIONS), (PHASE, trials))
        for trials in TRIALS
    ]
    shown = ("error", "cloud mean", "camps mean", "cloud 95th", "rate", "gap")
    summary, rules, maps = [], [], {name: [] for name in shown}
    fitted: list[float] = []
    for trials, scored in zip(TRIALS, judge_all(jobs), strict=True):
        ks = np.array(next(cell.ks for cell in cells if cell.trials == trials))
        on_map = np.isin(ks, MAP_KS)
        best, inside, near = band(scored, on_map)
        point = scored.point
        fitted.append(ks[best])
        summary.append(
            [
                trials,
                fmt_k(ks[best]),
                f"{ks[best] / trials:.0f}n",
                f"{point['error'][best]:.3f}",
                f"{point['cloud mean'][best]:.3f}",
                f"{point['camps mean'][best]:.3f}",
                _span(ks[inside & on_map]),
                _span(ks[near & on_map]),
            ]
        )
        for name, k in rule_ks(trials, MAP_RULES).items():
            at = int(np.flatnonzero(ks == k)[0])
            rules.append(
                [
                    trials,
                    name,
                    fmt_k(k),
                    f"{point['error'][at]:.3f}",
                    f"{point['error'][at] - point['error'][best]:+.3f}",
                    "yes" if inside[at] else "no",
                    "yes" if near[at] else "no",
                    f"{point['cloud mean'][at]:.3f}",
                    f"{point['2 camps mean'][at]:.3f}",
                    f"{point['4 camps mean'][at]:.3f}",
                    f"{point['cloud 95th'][at]:.3f}",
                    f"{point['rate'][at]:.3f}",
                    f"{point['gap'][at]:+.3f}",
                ]
            )
        for name in shown:
            sign = "+" if name == "gap" else ""
            maps[name].append([trials, *(f"{value:{sign}.2f}" for value in point[name][on_map])])

    logs, late = np.log(TRIALS), np.array(TRIALS) >= 13

    def power(values: list[float], keep: Array | slice = slice(None)) -> float:
        return float(np.polyfit(logs[keep], np.log(values)[keep], 1)[0])

    headers = ["Trials", *(fmt_k(k) for k in MAP_KS)]
    titles = {
        "error": "The weighted error.",
        "cloud mean": "The cloud's mean, which is the noise error.",
        "camps mean": "The mean of the tight camps. The camp error is 1 less this.",
        "cloud 95th": "The cloud's 95th percentile.",
        "rate": "How often a camp draw outscores a cloud draw.",
        "gap": "The camps' 5th percentile less the cloud's 95th.",
    }
    parts = [
        "Phase 1. The map",
        f"Cloud, 2 and 4 tight camps over 4 options. {DRAWS} draws per cell, 25 values of k.\n"
        f"A tight camp's trials are {1 - TIGHT:.0%} the camp's centre and {TIGHT:.0%} their own "
        "mix.\nA camp draw is a draw from either camp population.",
        "The best k at each number of trials.\n"
        "The weighted error is the cloud's mean, counted "
        f"{WEIGHT:g} times, plus what the camps' mean falls short of 1,\nall divided by "
        f"{WEIGHT + 1:g}. The best k has the lowest weighted error on the map. Band is every k "
        "on the map\nthat ties with it. Near is every k with an error within "
        f"{NEAR} of the best, whether or not the\ndifference could be chance.\n"
        + table(
            ["Trials", "Best k", "As n", "Error", "Cloud", "Camps", "Band", "Near"],
            summary,
        ),
        f"The best k grows as trials to the power {power(fitted):.2f}.\n"
        f"From 13 trials it grows as the power {power(fitted, late):.2f}.",
        "Each rule read off the map. Less best is the rule's weighted error less the error at "
        "the best k.\nThe three means are for principle 4, and the last three columns are the "
        "tails.\n"
        + table(
            [
                "Trials",
                "Rule",
                "k",
                "Error",
                "Less best",
                "In band",
                "Near",
                "Cloud",
                "2 camps",
                "4 camps",
                "Cloud 95th",
                "Rate",
                "Gap",
            ],
            rules,
        ),
        *(f"{titles[name]}\n{table(headers, maps[name])}" for name in shown),
    ]
    return "\n\n".join(parts) + "\n"
