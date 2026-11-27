"""Phase 2: with few trials, find whether the rule needs a floor and the minimum to read."""

from pathlib import Path

import numpy as np
from common import (
    CAMPS_2,
    CAMPS_4,
    CLOUD,
    DRAWS,
    DRAWS_TAILS,
    TRIALS,
    WEIGHT,
    WEIGHTS,
    Array,
    Cell,
    Population,
    Scored,
    cohesion_at,
    error,
    fmt_k,
    interval,
    judge_all,
    outscore,
    resamples,
    rule_ks,
    run_cells,
    table,
)

PHASE = 2
POPULATIONS = (CLOUD, CAMPS_2, CAMPS_4)
GROWTHS = ("fixed 100", "8n", "max(100, 8n)", "20n", "max(100, 20n)")
JOINS = ("100 + 8n", "root(100^2 + (8n)^2)")
RULES = (*GROWTHS, *JOINS)

#: The rule every other rule is compared with. It is the lowest balance the derivation gives.
BASE = "8n"
FEW = tuple(range(3, 17))
MANY = (25, 50, 100, 200, 400)

#: The share of pairings in which tight camps must outscore a cloud.
ENOUGH = 0.95

TITLES = {
    "error": "The weighted error. This is what picks the rule.",
    "cloud 95th": "Principle 1. The cloud's 95th percentile.",
    "2 camps 5th": "Principle 2. The 5th percentile of 2 tight camps.",
    "4 camps 5th": "Principle 2. The 5th percentile of 4 tight camps.",
    "order": "Principle 2. How often a draw of 2 camps outscores a draw of 4 camps.",
    "rate": "Principle 3. How often a camp draw outscores a cloud draw.",
    "gap": "Principle 3. The camps' 5th percentile less the cloud's 95th.",
    "rate 2": "How often a draw of 2 tight camps outscores a cloud draw.",
    "rate 4": "How often a draw of 4 tight camps outscores a cloud draw.",
    "cloud mean": "Principle 4. The cloud's mean.",
    "2 camps mean": "Principle 4. The mean of 2 tight camps.",
    "4 camps mean": "Principle 4. The mean of 4 tight camps.",
}


def cells(
    rules: tuple[str, ...] = RULES,
    populations: tuple = POPULATIONS,
    few: tuple[int, ...] = FEW,
    many: tuple[int, ...] = MANY,
    phase: int = PHASE,
) -> list[Cell]:
    """Return every cell: 2,000 draws at few trials and 200 at many."""
    out = []
    for index, population in enumerate(populations):
        for trials in (*few, *many):
            ks = tuple(sorted(set(rule_ks(trials, rules).values())))
            draws = DRAWS_TAILS if trials in few else DRAWS
            out.append(Cell(population, trials, ks, draws, (phase, index)))
    return out


def scored(
    cache: Path | None, tag: str = "few", **changes: object
) -> dict[int, tuple[Scored, dict[str, int]]]:
    """Return every measure at each number of trials, and the column each rule reads.

    The keyword arguments are passed on to :func:`cells`.
    """
    every = cells(**changes)  # type: ignore[arg-type]
    rules: tuple[str, ...] = changes.get("rules", RULES)  # type: ignore[assignment]
    phase = changes.get("phase", PHASE)
    populations = list(dict.fromkeys(cell.population for cell in every))
    counts = list(dict.fromkeys(cell.trials for cell in every))
    names = [(cell.population.name, cell.trials) for cell in every]
    scores = dict(zip(names, run_cells(every, cache, tag), strict=True))
    jobs = [
        (tuple(scores[population.name, trials] for population in populations), (phase, trials))
        for trials in counts
    ]
    out = {}
    for trials, result in zip(counts, judge_all(jobs), strict=True):  # type: ignore[arg-type]
        ks = sorted(set(rule_ks(trials, rules).values()))
        columns = {name: ks.index(k) for name, k in rule_ks(trials, rules).items()}
        out[trials] = (result, columns)
    return out


def _versus(results: dict, name: str, rival: str, against: str, counts: tuple[int, ...]) -> str:
    """Count the trial numbers at which one rule is better, tied and worse than another."""
    marks = [
        results[trials][0].mark(name, results[trials][1][against])[results[trials][1][rival]]
        for trials in counts
    ]
    return " / ".join(str(marks.count(mark)) for mark in "+=-")


def minimum(results: dict, name: str, rule: str) -> str:
    """Return the fewest trials from which a rate stays at 95% or more."""
    counts = sorted(results)
    enough = [
        results[trials][0].point[name][results[trials][1][rule]] >= ENOUGH for trials in counts
    ]
    if not enough[-1]:
        return "none"
    last_short = max((i for i, ok in enumerate(enough) if not ok), default=-1)
    return str(counts[last_short + 1])


def totals(results: dict) -> str:
    """Return each rule's weighted error over the grid, at every weight, against the base rule.

    The grid is the phase 1 one, which is close to even in the log of trials,
    so no stretch of trials counts for more than another.
    """

    def column(rule: str, trials: int) -> int:
        return results[trials][1][rule]

    def total(rule: str, name: str) -> tuple[float, Array]:
        point = np.mean([results[n][0].point[name][column(rule, n)] for n in TRIALS])
        again = np.mean([results[n][0].resampled[name][:, column(rule, n)] for n in TRIALS], axis=0)
        return float(point), again

    parts = {rule: (total(rule, "cloud mean"), total(rule, "camps mean")) for rule in RULES}
    rows = []
    for rule in RULES:
        (noise, noise_again), (camps, camps_again) = parts[rule]
        (base_noise, base_noise_again), (base_camps, base_camps_again) = parts[BASE]
        row = [rule, f"{noise:.3f}", f"{1 - camps:.3f}"]
        for weight in WEIGHTS:
            low, high = interval(
                error(noise_again, camps_again, weight)
                - error(base_noise_again, base_camps_again, weight)
            )
            mark = "" if rule == BASE else " +" if high < 0 else " -" if low > 0 else " ="
            row.append(f"{error(noise, camps, weight):.3f}{mark}")
        more_noise, less_missed = noise - base_noise, camps - base_camps
        if rule == BASE:
            row.append("")
        elif more_noise > 0 and less_missed > 0:
            row.append(f"above {less_missed / more_noise:.2f}")
        elif more_noise > 0:
            row.append("any")
        elif less_missed < 0:
            row.append(f"below {less_missed / more_noise:.2f}")
        else:
            row.append("none")
        rows.append(row)
    return (
        f"The weighted error over the grid of {len(TRIALS)} numbers of trials, from {TRIALS[0]} "
        f"to {TRIALS[-1]}. Noise error is the cloud's mean\nand camp error is what the camps' "
        "mean falls short of 1. Then the weighted error at each weight, where\nthe weight is "
        f"how many times the noise error counts. The experiment uses {WEIGHT:g}. The sign "
        f"compares a rule\nwith {BASE} on the same draws. The last column is the weights at "
        f"which {BASE} has the lower error.\n"
        + table(
            [
                "Rule",
                "Noise error",
                "Camp error",
                *(f"Weight {weight:g}" for weight in WEIGHTS),
                f"{BASE} wins at",
            ],
            rows,
        )
    )


def report(cache: Path | None = None) -> str:
    """Return phase 2 as text."""
    results = scored(cache)
    counts = sorted(results)
    parts = [
        "Phase 2. Few trials",
        f"Cloud, 2 and 4 tight camps over 4 options. {DRAWS_TAILS:,} draws per cell from "
        f"{FEW[0]} to {FEW[-1]} trials, {DRAWS} above.\n"
        f"Beside each value is how the rule compares with {BASE} on the same draws.\n"
        "A plus is better by more than a tie, a minus is worse by more than a tie and an equals "
        "sign is a tie.",
        "The k each rule gives.\n"
        + table(
            ["Trials", *RULES],
            [[n, *(fmt_k(k) for k in rule_ks(n, RULES).values())] for n in counts],
        ),
        totals(results),
    ]
    for name, title in TITLES.items():
        sign = "+" if name == "gap" else ""
        rows = []
        for trials in counts:
            result, columns = results[trials]
            marks = result.mark(name, columns[BASE])
            values = ((result.point[name][columns[rule]], marks[columns[rule]]) for rule in RULES)
            rows.append([trials, *(f"{value:{sign}.3f} {mark}" for value, mark in values)])
        parts.append(f"{title}\n" + table(["Trials", *RULES], rows))

    judged = ("error", "cloud 95th", "2 camps 5th", "4 camps 5th", "order", "rate", "gap")
    spans = (
        ("3 to 12", tuple(n for n in counts if n <= 12)),
        ("13 to 400", tuple(n for n in counts if n > 12)),
    )
    rows = [
        [rule, span, *(_versus(results, name, rule, BASE, ns) for name in judged)]
        for rule in RULES
        if rule != BASE
        for span, ns in spans
    ]
    parts.append(
        f"Each rule against {BASE}. The numbers of trials at which it is better, tied and "
        "worse.\n" + table(["Rule", "Trials", *judged], rows)
    )

    rows = [
        [rule, minimum(results, "rate 2", rule), minimum(results, "rate 4", rule)] for rule in RULES
    ]
    parts.append(
        f"The minimum number of trials. The fewest from which tight camps outscore a cloud in "
        f"{ENOUGH:.0%} of\npairings, and keep doing so at every larger number tested.\n"
        + table(["Rule", "2 camps", "4 camps"], rows)
    )

    rows = []
    for rule in RULES:
        row = [rule]
        for name in ("cloud mean", "2 camps mean", "4 camps mean"):
            means = np.array([results[n][0].point[name][results[n][1][rule]] for n in counts])
            settled = means[-1]
            from_10 = means[counts.index(10) :]
            row += [
                f"{means.min():.3f} to {means.max():.3f}",
                f"{np.ptp(from_10):.3f}",
                f"{means[0] - settled:+.3f}",
                f"{means[counts.index(10)] - settled:+.3f}",
            ]
        rows.append(row)
    headers = ["Rule"]
    for population in ("Cloud", "2 camps", "4 camps"):
        headers += [f"{population} range", "From 10", "3 less 400", "10 less 400"]
    parts.append(
        "Principle 4. The range of each mean across trials, the range from 10 trials, and how "
        "far the mean\nat 3 and at 10 trials sits from the mean at 400. A minus reads lower at "
        "few trials.\n" + table(headers, rows)
    )
    return "\n\n".join(parts) + "\n"


FLOOR_RULES = (BASE, "max(100, 8n)")
FLOOR_TRIALS = (3, 4, 5, 6, 8, 10, 12)
FLOOR_DRAWS = 1000

#: Nine kinds of noise. Each moves one or two parts of the cloud the rule was chosen on.
NOISE = {
    "cloud": CLOUD,
    "spread 0.1": Population("cloud, spread 0.1", spread=0.1),
    "spread 0.3": Population("cloud, spread 0.3", spread=0.3),
    "spread 3": Population("cloud, spread 3", spread=3.0),
    "2 options": Population("cloud, options 2", options=2),
    "16 options": Population("cloud, options 16", options=16),
    "30 observations": Population("cloud, observations 30", observations=30),
    "100 observations": Population("cloud, observations 100", observations=100),
    "16 options, 30 observations": Population(
        "cloud, options 16, observations 30", options=16, observations=30
    ),
}


def _floor_camps() -> dict[str, tuple[Population, str]]:
    """Return eighteen kinds of camps, each with the noise it is set against."""
    out = {}
    for count in (2, 4):
        for loose in (0.0, 0.02, 0.05, 0.1, 0.2):
            name = f"{count} camps, looseness {loose:g}"
            out[name] = (Population(name, camps=count, looseness=loose), "cloud")
        for size in (30, 100):
            name = f"{count} tight camps, {size} observations"
            out[name] = (Population(name, camps=count, observations=size), f"{size} observations")
        name = f"{count} tight camps, 16 options"
        out[name] = (Population(name, camps=count, options=16), "16 options")
        name = f"{count} tight camps, 16 options, 30 observations"
        out[name] = (
            Population(name, camps=count, options=16, observations=30),
            "16 options, 30 observations",
        )
    return out


def _lowest() -> str:
    """Return the lowest score any ensemble can read, under each rule."""
    rows = []
    for trials in (3, 4, 5, 6, 7, 8):
        pairs = trials * (trials - 1) // 2
        ends, middles = np.linspace(0, 1, pairs), (np.arange(pairs) + 0.5) / pairs
        reads = (
            min(cohesion_at(ends, k), cohesion_at(middles, k))
            for k in rule_ks(trials, FLOOR_RULES).values()
        )
        rows.append([trials, pairs, *(f"{read:.3f}" for read in reads)])
    return (
        "The lowest score an ensemble can read. Its pairwise divergences are spread evenly "
        "from 0 to 1,\nwhich is as far from bunched as they can be.\n"
        + table(["Trials", "Pairs", *FLOOR_RULES], rows)
    )


def floor(cache: Path | None = None) -> str:
    """Return the floor of 100 against no floor on wider data, as text."""
    camps = _floor_camps()
    populations = {**NOISE, **{name: population for name, (population, _) in camps.items()}}
    cells = [
        Cell(population, trials, tuple(rule_ks(trials, FLOOR_RULES).values()), FLOOR_DRAWS, (20, i))
        for i, population in enumerate(populations.values())
        for trials in FLOOR_TRIALS
    ]
    keys = [(name, trials) for name in populations for trials in FLOOR_TRIALS]
    scores = dict(zip(keys, run_cells(cells, cache, "floor"), strict=True))
    index = {name: i for i, name in enumerate(populations)}

    def pairs(names: dict, read: object) -> list[list[object]]:
        rows = []
        for name in names:
            values = (read(scores[name, trials]) for trials in FLOOR_TRIALS)  # type: ignore[operator]
            rows.append([name, *(f"{bare:.2f} {held:.2f}" for bare, held in values)])
        return rows

    def mean_of(names: dict, read: object) -> list[object]:
        values = [
            np.mean([read(scores[name, trials]) for name in names], axis=0)  # type: ignore[operator]
            for trials in FLOOR_TRIALS
        ]
        return ["Mean of all", *(f"{bare:.2f} {held:.2f}" for bare, held in values)]

    def p95(values: Array) -> Array:
        return np.percentile(values, 95, axis=0)

    def mean(values: Array) -> Array:
        return values.mean(axis=0)

    headers = [*(str(trials) for trials in FLOOR_TRIALS)]
    rates, tally = [], {trials: [0, 0, 0] for trials in FLOOR_TRIALS}
    for name, (_, noise) in camps.items():
        row: list[object] = [name]
        for trials in FLOOR_TRIALS:
            high, low = scores[name, trials], scores[noise, trials]
            rate = outscore(high, low)
            picks = zip(
                resamples(FLOOR_DRAWS, (20, index[name], trials)),
                resamples(FLOOR_DRAWS, (20, index[noise], trials)),
                strict=True,
            )
            again = np.stack([outscore(high[a], low[b]) for a, b in picks])
            bottom, top = interval(again[:, 1] - again[:, 0])
            mark = "+" if bottom > 0 else "-" if top < 0 else "="
            tally[trials]["+=-".index(mark)] += 1
            row.append(f"{rate[0]:.2f} {rate[1]:.2f} {mark}")
        rates.append(row)
    parts = [
        "Phase 2. The floor on wider data",
        f"Plain 8n against max(100, 8n), from {FLOOR_TRIALS[0]} to {FLOOR_TRIALS[-1]} trials, "
        f"where the two differ. {FLOOR_DRAWS:,} draws per cell, 4 options and exact\n"
        "probabilities unless stated. Each pair of numbers is no floor and then the floor.",
        "The k each rule gives.\n"
        + table(
            ["Trials", *FLOOR_RULES],
            [[n, *(fmt_k(k) for k in rule_ks(n, FLOOR_RULES).values())] for n in FLOOR_TRIALS],
        ),
        _lowest(),
        "Noise, mean.\n" + table(["Noise", *headers], [*pairs(NOISE, mean), mean_of(NOISE, mean)]),
        "Noise, 95th percentile.\n"
        + table(["Noise", *headers], [*pairs(NOISE, p95), mean_of(NOISE, p95)]),
        "Camps, mean.\n" + table(["Camps", *headers], pairs(camps, mean)),
        "How often a camp draw outscores a draw of the matching noise. A plus means the floor "
        "is better by\nmore than a tie, a minus means no floor is and an equals sign is a tie.\n"
        + table(["Camps", *headers], rates),
        f"The kinds of camps, of {len(camps)}, on which the floor is better, tied and worse at "
        "telling camps from noise.\n"
        + table(
            ["Trials", "Floor better", "Tied", "No floor better"],
            [[trials, *tally[trials]] for trials in FLOOR_TRIALS],
        ),
    ]
    return "\n\n".join(parts) + "\n"
