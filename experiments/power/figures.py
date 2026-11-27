"""The toy figure of error-cohesion and the papers' figures, under each candidate.

The toy is one question with four options, sampled 30 times. Its three
ensembles are rebuilt here from its seeded generator, so this file does not
read the error-cohesion repository. Every ensemble shares one mean prediction
and one KL variance, and only how the samples gather differs.
"""

import numpy as np
from candidates import NAMES, read, trials_of
from common import table
from phase4 import figure_2, figures_5_and_6, reference

import incoherence as ic

#: The toy's settings, as in ``toy/main.py`` of error-cohesion.
TOY_SAMPLES = 30
TOY_OPTIONS = 4
TOY_MEAN = np.log([0.45, 0.35, 0.12, 0.08])
TOY_VARIANCE = 0.8


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _kl_variance(p: np.ndarray) -> float:
    """Return the average KL divergence from the normalised geometric mean to each sample."""
    mean = _softmax(np.log(p).mean(0))
    return max(0.0, float((mean * (np.log(mean) - np.log(p))).sum(1).mean()))


def _toy_shapes(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return the toy's centred directions in log space, one row per sample."""

    def jitter() -> np.ndarray:
        return 0.06 * rng.standard_normal((TOY_SAMPLES, TOY_OPTIONS))

    two = np.repeat([[1, -1, 0, 0], [-1, 1, 0, 0]], TOY_SAMPLES // 2, axis=0) + jitter()
    cloud = rng.standard_normal((TOY_SAMPLES, TOY_OPTIONS))
    five = (
        np.repeat(
            [
                [1.2, -0.4, -0.4, -0.4],
                [-0.4, 1.2, -0.4, -0.4],
                [-0.4, -0.4, 1.2, -0.4],
                [-0.4, -0.4, -0.4, 1.2],
                [0.6, 0.6, -0.6, -0.6],
            ],
            TOY_SAMPLES // 5,
            axis=0,
        )
        + jitter()
    )
    out = {"Toy, no camps": cloud, "Toy, five camps": five, "Toy, two camps": two}
    return {name: z - z.mean(0) for name, z in out.items()}


def _toy_build(shape: np.ndarray) -> np.ndarray:
    """Scale a shape by bisection until the ensemble has the toy's KL variance."""
    low, high = 0.0, 60.0
    for _ in range(80):
        scale = (low + high) / 2
        if _kl_variance(_softmax(TOY_MEAN + scale * shape)) < TOY_VARIANCE:
            low = scale
        else:
            high = scale
    return _softmax(TOY_MEAN + scale * shape)


def _toy() -> dict[str, np.ndarray]:
    """Return the divergences of the toy's three ensembles, as its figure orders them."""
    shapes = _toy_shapes(np.random.default_rng(0))
    order = ("Toy, two camps", "Toy, five camps", "Toy, no camps")
    return {
        name: np.asarray(ic.measure_counts(_toy_build(shapes[name])).divergences) for name in order
    }


def _certain() -> dict[str, np.ndarray]:
    """Return two camps of fully certain trials, 14 on one option and 26 on another."""
    rows = np.repeat(np.eye(4)[:2], (14, 26), axis=0)
    return {"Two fully certain camps, 14 and 26": np.asarray(ic.measure_counts(rows).divergences)}


def _group(title: str, named: dict[str, np.ndarray | list[np.ndarray]]) -> str:
    """Lay out every candidate's Cohesion for a group of ensembles."""
    rows = []
    for name, values in named.items():
        many = values if isinstance(values, list) else [values]
        trials = trials_of(many[0])
        reads = [read(v, trials) for v in many]
        rows.append([name, trials, *(f"{np.mean([r[c] for r in reads]):.3f}" for c in NAMES)])
    return title + "\n" + table(["Ensemble", "Trials", *NAMES], rows)


def report() -> str:
    """Return the figures under every candidate as text."""
    runs = [figure_2(np.random.default_rng([4, 2, run])) for run in range(10)]
    second = {name: [run[name] for run in runs] for name in runs[0]}
    parts = [
        "The figures. Each ensemble under every candidate, highest first as its source gives them",
        _group(
            "The toy figure of error-cohesion, 30 samples over 4 options, and two camps that are "
            "each fully certain.",
            {**_toy(), **_certain()},
        ),
        _group(
            "Figure 2 of the Cohesion paper, the mean of the 10 seeded runs of experiments/k.",
            second,
        ),
        _group(
            "Figures 5 and 6 of the Cohesion paper, with the notebook's seeds.", figures_5_and_6()
        ),
        _group("The three test ensembles of tests/test_paper.py.", reference()),
    ]
    return "\n\n".join(parts) + "\n"
