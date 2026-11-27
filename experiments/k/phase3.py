"""Phase 3: move one part of the data at a time, to find what no rule for k fixes."""

from pathlib import Path

import numpy as np
from common import (
    CAMPS_2,
    CAMPS_4,
    CLOUD,
    DRAWS,
    TIGHT,
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

PHASE = 3
WINNER = "8n"
RULES = ("fixed 100", WINNER)
TRIALS = (10, 50, 400)

#: The share of pairings in which camps must outscore a cloud to count as apart from it.
ENOUGH = 0.95

LOOSENESS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
SPREADS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
OBSERVATIONS = (10, 30, 100, 1000)
OPTIONS = (2, 16, 64)


def _camps(count: int, **changes: object) -> Population:
    words = [f"{count} camps", *(f"{key} {value}" for key, value in changes.items())]
    return Population(", ".join(words), camps=count, **changes)  # type: ignore[arg-type]


def _cloud(**changes: object) -> Population:
    words = ["cloud", *(f"{key} {value}" for key, value in changes.items())]
    return Population(", ".join(words), **changes)  # type: ignore[arg-type]


def populations() -> list[Population]:
    """Return every population phase 3 scores, each moving one part of the data."""
    out = [CLOUD, CAMPS_2, CAMPS_4]
    out += [_camps(count, looseness=loose) for count in (2, 4) for loose in LOOSENESS]
    out += [_cloud(spread=spread) for spread in SPREADS]
    out += [_camps(8), _camps(2, sizes="9 to 1")]
    out += [_camps(count, sizes=sizes) for sizes in ("stray", "half") for count in (2, 4)]
    for size in OBSERVATIONS:
        out += [_cloud(observations=size), _camps(1, looseness=0.0, observations=size)]
        out += [_camps(count, observations=size) for count in (2, 4)]
    for options in OPTIONS:
        out += [_cloud(options=options), *(_camps(count, options=options) for count in (2, 4))]
    return out


class Scores:
    """Scores for every population at each number of trials, one column per rule."""

    def __init__(self, cache: Path | None) -> None:
        every = populations()
        cells, columns = [], []
        for index, population in enumerate(every):
            for trials in TRIALS:
                ks = rule_ks(trials, RULES)
                unique = tuple(sorted(set(ks.values())))
                cells.append(Cell(population, trials, unique, DRAWS, (PHASE, index)))
                columns.append([unique.index(k) for k in ks.values()])
        done = run_cells(cells, cache, "limits")
        self._scores = {
            (cell.population.name, cell.trials): scores[:, column]
            for cell, scores, column in zip(cells, done, columns, strict=True)
        }
        self._index = {population.name: index for index, population in enumerate(every)}

    def get(self, population: Population, trials: int) -> Array:
        """Return one row per draw and one column per rule."""
        return self._scores[population.name, trials]

    def picks(self, population: Population, trials: int) -> Array:
        """Return the draws each resample picks for this population."""
        return resamples(DRAWS, (PHASE, self._index[population.name], trials))

    def tail(self, population: Population, trials: int, percent: float) -> str:
        """Return a percentile under each rule, and how the winner compares with fixed 100.

        A plus means the winner is better by more than a tie. That is lower
        for the 95th percentile, which is read on noise, and higher for the 5th.
        """
        scores = self.get(population, trials)
        fixed, winner = np.percentile(scores, percent, axis=0)
        again = np.percentile(scores[self.picks(population, trials)], percent, axis=1)
        low, high = interval(again[:, 1] - again[:, 0])
        better, worse = (high < 0, low > 0) if percent > 50 else (low > 0, high < 0)
        return f"{fixed:.3f}  {winner:.3f} {'+' if better else '-' if worse else '='}"

    def means(self, population: Population, trials: int) -> str:
        """Return the mean under each rule."""
        fixed, winner = self.get(population, trials).mean(axis=0)
        return f"{fixed:.3f}  {winner:.3f}"

    def rate(self, high: Population, low: Population, trials: int) -> tuple[Array, str]:
        """Return how often ``high`` outscores ``low`` under each rule, with the comparison."""
        first, second = self.get(high, trials), self.get(low, trials)
        rates = outscore(first, second)
        picks = zip(self.picks(high, trials), self.picks(low, trials), strict=True)
        again = np.stack([outscore(first[a], second[b]) for a, b in picks])
        bottom, top = interval(again[:, 1] - again[:, 0])
        mark = "+" if bottom > 0 else "-" if top < 0 else "="
        return rates, f"{rates[0]:.3f}  {rates[1]:.3f} {mark}"


MEAN, RATE = "Mean at 100  winner", "Rate at 100  winner"


def _tightness(scores: Scores) -> tuple[str, list[list[object]]]:
    """Loosen the camps a step at a time until they are a cloud."""
    rows, verdicts = [], []
    for count in (2, 4):
        for trials in TRIALS:
            means, apart = [], []
            for loose in LOOSENESS:
                population = _camps(count, looseness=loose)
                rates, text = scores.rate(population, CLOUD, trials)
                means.append(scores.get(population, trials).mean(axis=0))
                apart.append(rates >= ENOUGH)
                rows.append(
                    [
                        f"{count} camps",
                        f"{loose:g}",
                        trials,
                        scores.means(population, trials),
                        scores.tail(population, trials, 5),
                        text,
                    ]
                )
            steps = np.diff(np.array(means), axis=0)
            cloud = scores.get(CLOUD, trials).mean(axis=0)
            # The number of steps, counted from exact camps, before the first that falls short.
            held = np.logical_and.accumulate(np.array(apart), axis=0).sum(axis=0)
            verdicts.append(
                [
                    f"{count} camps",
                    trials,
                    *(
                        "none" if steps_held == 0 else f"{LOOSENESS[steps_held - 1]:g}"
                        for steps_held in held
                    ),
                    *(f"{step:+.3f}" for step in steps.max(axis=0)),
                    *(f"{gap:+.3f}" for gap in (np.array(means) - cloud).min(axis=0)),
                ]
            )
    first = table(
        [
            "Camps",
            "Looseness",
            "Trials",
            "Mean at 100  winner",
            "5th at 100  winner",
            "Rate at 100  winner",
        ],
        rows,
    )
    second = table(
        [
            "Camps",
            "Trials",
            "Apart to, at 100",
            "Winner",
            "Largest rise, at 100",
            "Winner",
            "Lowest less cloud, at 100",
            "Winner",
        ],
        verdicts,
    )
    text = (
        "Tightness. A trial is 1 - looseness of its camp's centre plus looseness of its own mix,\n"
        f"so 0 is an exact camp, {TIGHT:g} is a tight camp and 1 is a cloud. Rate is how often a "
        "draw\noutscores a cloud draw.\n"
        f"{first}\n\n"
        f"Apart to is the loosest step at which every tighter step outscores a cloud in "
        f"{ENOUGH:.0%} of pairings.\nLargest rise is the biggest rise in the mean from one step "
        "to the next looser one, which should\nnot be above zero. Lowest less cloud is the "
        "lowest mean on the sweep less the cloud's mean, which\nshould not be below zero.\n"
        f"{second}"
    )
    return text, verdicts


def _spread(scores: Scores) -> tuple[str, bool]:
    """Scatter the cloud widely or narrowly."""
    rows, fails = [], False
    for spread in SPREADS:
        population = _cloud(spread=spread)
        for trials in TRIALS:
            tail = scores.tail(population, trials, 95)
            fails = fails or tail.endswith("-")
            rows.append([f"{spread:g}", trials, scores.means(population, trials), tail])
    text = (
        "Cloud spread. The concentration of the Dirichlet that scatters the trials. Below 1 "
        "pushes them to\nthe corners and above 1 pulls them to the middle. A minus means the "
        "winner reads above fixed 100.\n"
        + table(["Spread", "Trials", "Mean at 100  winner", "95th at 100  winner"], rows)
    )
    return text, fails


def _shapes(scores: Scores) -> str:
    """Score the shapes and camp sizes the main grid leaves out."""
    stray, half = "stray", "half"
    listed = [
        ("2 tight camps", CAMPS_2),
        ("4 tight camps", CAMPS_4),
        ("8 tight camps", _camps(8)),
        ("2 camps, 9 to 1", _camps(2, sizes="9 to 1")),
        ("2 camps and a stray trial", _camps(2, sizes=stray)),
        ("4 camps and a stray trial", _camps(4, sizes=stray)),
        ("2 camps, half cloud", _camps(2, sizes=half)),
        ("4 camps, half cloud", _camps(4, sizes=half)),
        ("cloud", CLOUD),
    ]
    rows = []
    for name, population in listed:
        for trials in TRIALS:
            rate = "" if population is CLOUD else scores.rate(population, CLOUD, trials)[1]
            percent = 95 if population is CLOUD else 5
            rows.append(
                [
                    name,
                    trials,
                    scores.means(population, trials),
                    scores.tail(population, trials, percent),
                    rate,
                ]
            )
    order = [
        [
            trials,
            scores.rate(CAMPS_2, CAMPS_4, trials)[1],
            scores.rate(CAMPS_4, _camps(8), trials)[1],
        ]
        for trials in TRIALS
    ]
    strays = []
    for count, base in ((2, CAMPS_2), (4, CAMPS_4)):
        for trials in TRIALS:
            moved = scores.get(_camps(count, sizes=stray), trials).mean(axis=0)
            moved = moved - scores.get(base, trials).mean(axis=0)
            strays.append([f"{count} camps", trials, f"{moved[0]:+.3f}", f"{moved[1]:+.3f}"])
    return (
        "Shapes and camp sizes. The tail is the 5th percentile, or the 95th for the cloud.\n"
        + table(
            [
                "Population",
                "Trials",
                MEAN,
                "Tail at 100  winner",
                RATE,
            ],
            rows,
        )
        + "\n\nHow often fewer camps outscore more camps.\n"
        + table(["Trials", "2 above 4, at 100  winner", "4 above 8, at 100  winner"], order)
        + "\n\nOne stray trial. The mean with the stray trial less the mean without it.\n"
        + table(["Camps", "Trials", "At 100", "Winner"], strays)
    )


def _observations(scores: Scores) -> tuple[str, list[list[object]]]:
    """Give each trial a finite number of observations."""
    rows, drifts = [], []
    for size in OBSERVATIONS:
        cloud = _cloud(observations=size)
        listed = [
            ("cloud", cloud),
            ("one source", _camps(1, looseness=0.0, observations=size)),
            ("2 tight camps", _camps(2, observations=size)),
            ("4 tight camps", _camps(4, observations=size)),
        ]
        for name, population in listed:
            noise = population.camps < 2
            means = []
            for trials in TRIALS:
                means.append(scores.get(population, trials).mean(axis=0))
                rows.append(
                    [
                        f"{size:,}",
                        name,
                        trials,
                        scores.means(population, trials),
                        scores.tail(population, trials, 95 if noise else 5),
                        "" if noise else scores.rate(population, cloud, trials)[1],
                    ]
                )
            if population.camps:
                drifts.append(
                    [
                        f"{size:,}",
                        name,
                        *(f"{mean[1]:.3f}" for mean in means),
                        f"{means[-1][0] - means[0][0]:+.3f}",
                        f"{means[-1][1] - means[0][1]:+.3f}",
                    ]
                )
    text = (
        "Observations per trial. Each trial's probabilities are replaced by the shares of that "
        "many\nobservations drawn from them. The tail is the 5th percentile for camps and the "
        "95th otherwise.\nRate is against the cloud with the same observations.\n"
        + table(
            [
                "Observations",
                "Population",
                "Trials",
                MEAN,
                "Tail at 100  winner",
                RATE,
            ],
            rows,
        )
        + "\n\nThe winner's mean at each number of trials, and the move from 10 to 400 trials "
        "under each rule.\n"
        + table(
            [
                "Observations",
                "Population",
                *(f"{trials} trials" for trials in TRIALS),
                "Moved at 100",
                "Winner",
            ],
            drifts,
        )
    )
    return text, drifts


def _options(scores: Scores) -> str:
    """Change the number of options."""
    rows = []
    for options in (2, 4, 16, 64):
        changes = {} if options == 4 else {"options": options}
        cloud = _cloud(**changes) if changes else CLOUD
        listed = [("cloud", cloud)]
        listed += [
            (f"{count} tight camps", _camps(count, **changes) if changes else base)
            for count, base in ((2, CAMPS_2), (4, CAMPS_4))
        ]
        for name, population in listed:
            for trials in TRIALS:
                noise = population.camps == 0
                rows.append(
                    [
                        options,
                        name,
                        trials,
                        scores.means(population, trials),
                        scores.tail(population, trials, 95 if noise else 5),
                        "" if noise else scores.rate(population, cloud, trials)[1],
                    ]
                )
    return (
        "Options. The tail is the 5th percentile for camps and the 95th for the cloud. Rate is "
        "against the\ncloud with the same options.\n"
        + table(
            [
                "Options",
                "Population",
                "Trials",
                MEAN,
                "Tail at 100  winner",
                RATE,
            ],
            rows,
        )
    )


def report(cache: Path | None = None) -> str:
    """Return phase 3 as text."""
    scores = Scores(cache)
    tightness, _ = _tightness(scores)
    spread, fails = _spread(scores)
    observations, _ = _observations(scores)
    ks = [[trials, *(fmt_k(k) for k in rule_ks(trials, RULES).values())] for trials in TRIALS]
    parts = [
        "Phase 3. Limits",
        f"The winner, {WINNER}, against fixed 100. {DRAWS} draws per cell, 4 options unless "
        "stated.\nEach pair of numbers is fixed 100 and then the winner. The sign after a pair "
        "compares them on the\nsame draws. A plus means the winner is better by more than a tie, "
        "a minus means it is worse and\nan equals sign is a tie.",
        "The k each rule gives.\n" + table(["Trials", *RULES], ks),
        tightness,
        spread
        + "\nThe winner reads above fixed 100 at some spread: "
        + ("yes" if fails else "no")
        + ".",
        _shapes(scores),
        observations,
        _options(scores),
    ]
    return "\n\n".join(parts) + "\n"
