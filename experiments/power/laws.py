"""What each power reads on exact values, against a formula for each."""

import numpy as np
from candidates import entropies, raw_at, readouts
from common import CAMPS_2, CLOUD, cohesion_at, fmt_k, table

from incoherence._entropy import densities, density_entropy, points_for

LAW_KS = (100.0, 400.0, 1600.0)
SETS = 100


def _entropy(values: np.ndarray, k: float) -> tuple[float, float]:
    """Return the entropy read at power 1 and at power 0.5, from the raw numbers."""
    got = entropies(np.array([[raw_at(values, k)]]), (k,))
    return float(got["power 1"][0, 0]), float(got["power 0.5"][0, 0])


def _check() -> str:
    """Check the raw numbers give what the package gives, at both powers."""
    rng = np.random.default_rng([7, 0])
    worst = 0.0
    for population in (CLOUD, CAMPS_2):
        for trials in (5, 40):
            values = population.divergences(rng, trials)
            for k in (20.0, 100.0, 320.0):
                scores = np.array([[raw_at(values, k)]])
                mine = readouts(scores, (k,))
                one = density_entropy(densities(values, k, points_for(k)), k, 1.0)
                gaps = (
                    abs(float(mine["0.5 squared"][0, 0]) - cohesion_at(values, k)),
                    abs(float(mine["1 plain"][0, 0]) - (1 - float(one))),
                )
                worst = max(worst, *gaps)
    assert worst < 1e-12, worst
    return (
        "The check. Over 12 sets of divergences from a cloud and 2 tight camps, the raw numbers "
        "give\nthe package's Cohesion at power 0.5, and its entropy at power 1, to within 1e-12."
    )


def _equal() -> str:
    """M equal values: power 1 reads the chance two match, power 0.5 counts the values."""
    rows = []
    for m in (1, 2, 3, 7, 16):
        values = np.linspace(0.1, 0.9, m)
        for k in LAW_KS:
            one, half = _entropy(values, k)
            rows.append(
                [
                    m,
                    fmt_k(k),
                    f"{1 - one:.4f}",
                    f"{(k / m - 4) / (k - 4):.4f}",
                    f"{half:.4f}",
                    f"{min(1.0, 8 * (m - 1) / (k - 8)):.4f}",
                ]
            )
    head = (
        "Law 1. M values, exactly separate, with equal shares.\n"
        "Power 1 reads 1 - H as the chance two values match, 1 / m, or (k / m - 4) / (k - 4) at "
        "this k.\nPower 0.5 reads H as one equal step for each extra value, "
        "8(m - 1) / (k - 8).\n"
    )
    headers = ["Values", "k", "Power 1, 1 - H", "Formula", "Power 0.5, H", "Formula"]
    return head + table(headers, rows)


def _unequal() -> str:
    """Unequal shares: power 1 follows the biggest bunch, power 0.5 counts the strays."""
    cases = {
        "2 values, shares 82 and 18": (82, 18),
        "2 values, shares 99 and 1": (99, 1),
        "2 values, 999 and 1": (999, 1),
        "7 values, shares 2, 1, 1, 1, 1, 1, 1": (2, 1, 1, 1, 1, 1, 1),
        "7 values, shares 94, 1, 1, 1, 1, 1, 1": (94, 1, 1, 1, 1, 1, 1),
    }
    rows = []
    for name, counts in cases.items():
        counts = np.array(counts)
        shares = counts / counts.sum()
        values = np.repeat(np.linspace(0.1, 0.9, len(counts)), counts)
        matched, places = float(shares @ shares), float(np.sqrt(shares).sum() ** 2)
        for k in LAW_KS:
            one, half = _entropy(values, k)
            rows.append(
                [
                    name,
                    fmt_k(k),
                    f"{1 - one:.4f}",
                    f"{(k * matched - 4) / (k - 4):.4f}",
                    f"{half:.4f}",
                    f"{min(1.0, 8 * (places - 1) / (k - 8)):.4f}",
                ]
            )
    head = (
        "Law 2. Exactly separate values with unequal shares p.\n"
        "Power 1 swaps 1 / m for the sum of p squared, the chance two values match. A stray "
        "barely counts.\nPower 0.5 swaps m for the sum of root p, squared. A stray of 1% counts a "
        "fifth of a value.\nSo the formulas are (k sum p^2 - 4) / (k - 4) and "
        "8((sum root p)^2 - 1) / (k - 8).\n"
    )
    headers = ["Values", "k", "Power 1, 1 - H", "Formula", "Power 0.5, H", "Formula"]
    return head + table(headers, rows)


def _random() -> str:
    """N random values: power 1 reads 1 / N at any k, power 0.5 reads ``1 - k / (16N)``."""
    rows = []
    for count in (3, 10, 45, 300, 4950):
        rng = np.random.default_rng([7, 1, count])
        sets = rng.uniform(0, 1, (SETS, count))
        reads = {k: np.array([_entropy(values, k) for values in sets]).mean(axis=0) for k in LAW_KS}
        rows.append(
            [
                f"{count:,}",
                f"{1 / count:.4f}",
                *(f"{1 - reads[k][0]:.4f}" for k in LAW_KS),
                *(f"{reads[k][1]:.3f} / {max(0.0, 1 - k / (16 * count)):.3f}" for k in LAW_KS),
            ]
        )
    head = (
        f"Law 3. N random values, uniform on [0, 1]. The mean over {SETS} sets.\n"
        "Power 1 reads 1 - H as 1 / N whatever k is, so it needs no rule for k.\n"
        "Power 0.5 reads H as 1 - k / (16N), so k has to grow with N. This gives the 8n rule.\n"
    )
    headers = [
        "N",
        "1 / N",
        *(f"Power 1, k = {fmt_k(k)}" for k in LAW_KS),
        *(f"Power 0.5, k = {fmt_k(k)}" for k in LAW_KS),
    ]
    return head + table(headers, rows)


def report() -> str:
    """Return the laws as text."""
    parts = ["The laws. What each power reads on exact values", _check(), _equal()]
    parts += [_unequal(), _random()]
    return "\n\n".join(parts) + "\n"
