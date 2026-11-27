"""Phase 4: confirm the winner on data that was not used to choose it."""

import time
from pathlib import Path

import numpy as np
import phase2
from common import (
    CAMPS_2,
    CAMPS_4,
    CLOUD,
    DRAWS,
    DRAWS_TAILS,
    RULES,
    RealPopulation,
    cohesion_at,
    fmt_k,
    rule_ks,
    table,
)

from incoherence import measure, measure_counts
from incoherence._entropy import points_for

PHASE = 4
WINNER = "8n"
PAIR = ("fixed 100", WINNER)
JUDGED = ("error", "cloud 95th", "2 camps 5th", "4 camps 5th", "order", "rate", "gap")
REAL = (RealPopulation("cloud"), RealPopulation("2 camps", 2), RealPopulation("4 camps", 4))
REAL_TRIALS = (10, 13, 16, 25, 50, 100)


def _judged(results: dict) -> str:
    """Lay out the judged measures under fixed 100 and the winner, one row per trial count."""
    rows = []
    for trials, (result, columns) in results.items():
        fixed, winner = columns["fixed 100"], columns[WINNER]
        row = [trials, fmt_k(rule_ks(trials, PAIR)[WINNER])]
        for name in JUDGED:
            sign = "+" if name == "gap" else ""
            values, mark = result.point[name], result.mark(name, fixed)[winner]
            row.append(f"{values[fixed]:{sign}.3f}  {values[winner]:{sign}.3f} {mark}")
        rows.append(row)
    return table(["Trials", "k", *JUDGED], rows)


def _fresh(cache: Path | None) -> str:
    """Score the grid the rule was chosen on again, with seeds no earlier phase used."""
    results = phase2.scored(cache, "fresh", phase=PHASE)
    least = [phase2.minimum(results, name, WINNER) for name in ("rate 2", "rate 4")]
    return (
        f"Fresh seeds. Cloud, 2 and 4 tight camps over 4 options. {DRAWS_TAILS:,} draws to 16 "
        f"trials, {DRAWS} above.\n"
        + _judged(results)
        + f"\nUnder the winner, 2 tight camps outscore a cloud in 95% of pairings from {least[0]} "
        f"trials and 4 from {least[1]}.\n\n"
        "Fresh seeds, every rule of phase 2. " + phase2.totals(results)
    )


def _real(cache: Path | None) -> str:
    """Score real-valued trials, measured as the package measures them."""
    results = phase2.scored(
        cache, "real", rules=PAIR, populations=REAL, few=(), many=REAL_TRIALS, phase=PHASE + 10
    )
    return (
        f"Real values. Every trial is {REAL[0].observations} draws from a normal. A cloud gives "
        "each trial its own mean and\nspread. A camp shares one mean and spread, so its trials "
        f"differ only by sampling. {DRAWS} draws per cell.\n" + _judged(results)
    )


def figure_2(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Rebuild figure 2 of the Cohesion paper, which the notebook did not seed."""
    centres = {
        "Figure 2, Current": (100, 100, 100, 101, 101),
        "Figure 2, Option B": (82, 110, 111, 83, 110),
        "Figure 2, Option A": (83, 105, 95, 85, 90),
    }
    edges = np.linspace(65, 125, 51)
    out = {}
    for name, means in centres.items():
        counts = [np.histogram(rng.normal(mean, 5, 10000), edges)[0] for mean in means]
        out[name] = np.asarray(measure_counts(counts).divergences)
    return out


def figures_5_and_6() -> dict[str, np.ndarray]:
    """Rebuild figures 5 and 6 of the Cohesion paper, with the notebook's own seeds.

    The notebook used numpy's legacy global generator, so this does too.
    """
    state = np.random.RandomState(44)  # noqa: NPY002

    def patterns(count: int = 2) -> list[list[float]]:
        return [
            [state.uniform(0, 20), max(state.normal(1, 0.5), 0.1), state.randint(1, 10)]
            for _ in range(count)
        ]

    def bursts(alphabet: list | None = None) -> list[np.ndarray]:
        out = []
        for _ in range(40):
            letter = (
                patterns() if alphabet is None else alphabet[state.choice(range(len(alphabet)))]
            )
            out.append(np.concatenate([state.normal(*pattern) for pattern in letter]))
        return out

    two = [patterns() for _ in range(2)]
    variation = bursts(two)
    two[1][0][0] = two[0][0][0]
    matched = bursts(two)
    state.seed(52)
    five = bursts([patterns() for _ in range(5)])
    state.seed(32)
    none = bursts()
    named = {
        "Figures 5 and 6, two patterns": matched,
        "Figures 5 and 6, same two with small variation": variation,
        "Figures 5 and 6, five patterns": five,
        "Figures 5 and 6, no patterns": none,
    }
    return {name: np.asarray(measure(trials).divergences) for name, trials in named.items()}


def reference() -> dict[str, np.ndarray]:
    """Rebuild the three test ensembles of ``tests/test_paper.py``."""
    rng = np.random.default_rng(1)
    same = [rng.normal(0, 1, 500) for _ in range(10)]
    camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
    scatter = [rng.normal(rng.uniform(-4, 4), rng.uniform(0.3, 2), 500) for _ in range(10)]
    named = {"Ten matching trials": same, "Two camps": camps, "Scattered trials": scatter}
    return {name: np.asarray(measure(trials).divergences) for name, trials in named.items()}


def _papers() -> str:
    """Check the figures in the improvements page keep the order the papers give them."""
    runs = [figure_2(np.random.default_rng([PHASE, 2, run])) for run in range(10)]
    runs_by_name = {name: [run[name] for run in runs] for name in runs[0]}
    groups = [runs_by_name, figures_5_and_6(), reference()]
    rows, kept = [], []
    for group in groups:
        scores = []
        for name, values in group.items():
            many = values if isinstance(values, list) else [values]
            trials = round((1 + (1 + 8 * len(many[0])) ** 0.5) / 2)
            ks = rule_ks(trials, PAIR)
            fixed, winner = (np.mean([cohesion_at(v, k) for v in many]) for k in ks.values())
            scores.append((fixed, winner))
            rows.append([name, trials, fmt_k(ks[WINNER]), f"{fixed:.3f}", f"{winner:.3f}"])
        kept.append(all(np.all(np.diff(np.array(scores), axis=0) < 0, axis=0)))
    return (
        "The papers' orderings. Each group is listed in the order its paper gives, highest "
        "first.\nFigure 2 is the mean of 10 seeded runs, as the notebook did not seed it.\n"
        + table(["Ensemble", "Trials", "k", "At 100", "Winner"], rows)
        + "\nEvery group keeps its order under both rules: "
        + ("yes" if all(kept) else "no")
        + "."
    )


GRID_KS = (24, 48, 80, 104, 128, 200, 400, 800, 1600, 3200, 12800)


def _grid() -> str:
    """Check the density grid has converged at every k the winner reaches."""
    rows = []
    for k in GRID_KS:
        trials = k // 8
        draws = 2 if trials > 400 else 20
        worst, total, count = 0.0, 0.0, 0
        for index, population in enumerate((CLOUD, CAMPS_2, CAMPS_4)):
            for draw in range(draws):
                rng = np.random.default_rng([PHASE, 3, index, k, draw])
                values = population.divergences(rng, trials)
                coarse = cohesion_at(values, k)
                fine = cohesion_at(values, k, 4 * (points_for(k) - 1) + 1)
                worst, total, count = (
                    max(worst, abs(fine - coarse)),
                    total + fine - coarse,
                    count + 1,
                )
        rows.append(
            [fmt_k(k), f"{trials:,}", f"{points_for(k):,}", f"{total / count:+.4f}", f"{worst:.4f}"]
        )
    return (
        "The grid. Cohesion on a grid four times finer, less Cohesion on the grid the package "
        "uses.\nCloud, 2 and 4 tight camps, 20 draws of each, or 2 at 1,600 trials.\n"
        + table(["k", "Trials", "Grid points", "Mean move", "Largest move"], rows)
    )


def report(cache: Path | None = None) -> str:
    """Return phase 4 as text."""
    parts = [
        "Phase 4. Confirm",
        f"The winner, {WINNER}, against fixed 100, on data no earlier phase used.\nEach pair of "
        "numbers is fixed 100 and then the winner. The sign compares them on the same draws.\n"
        "A plus means the winner is better by more than a tie, a minus worse, an equals sign a "
        "tie.",
        _fresh(cache),
        _real(cache),
        _papers(),
        _grid(),
    ]
    return "\n\n".join(parts) + "\n"


def cost(cache: Path | None = None) -> str:
    """Return the time and memory of one run at 1,600 trials. Timings are not checked."""
    del cache
    trials = 1600
    rng = np.random.default_rng([PHASE, 4])
    start = time.perf_counter()
    values = CAMPS_4.divergences(rng, trials)
    took = time.perf_counter() - start
    rows = [["Pairwise divergences", "", f"{len(values):,} pairs", f"{took:.2f}"]]
    for name in ("fixed 100", WINNER, "20n"):
        k = RULES[name](trials)
        best = np.inf
        for _ in range(3):
            start = time.perf_counter()
            cohesion_at(values, k)
            best = min(best, time.perf_counter() - start)
        megabytes = 8 * points_for(k) / 1e6
        points = f"{points_for(k):,} points, {megabytes:.1f} MB each"
        rows.append([f"Last step, {name}", fmt_k(k), points, f"{best:.2f}"])
    return (
        "Phase 4. The cost of one run at 1,600 trials\n\n"
        "4 tight camps over 4 options. Seconds are the best of 3 on one machine, so this table "
        "is not checked.\n" + table(["Step", "k", "Size", "Seconds"], rows) + "\n"
    )
