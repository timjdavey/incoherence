"""Run the k experiments and store their tables, or check the stored tables still hold.

Run it from the repository root::

    uv run python experiments/k/run.py              # every experiment
    uv run python experiments/k/run.py laws map     # some of them
    uv run python experiments/k/run.py --check      # rerun and compare with what is stored
"""

import argparse
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import noise
import phase1
import phase2
import phase3
import phase4

RESULTS = Path(__file__).parent / "results"
NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")

Experiment = Callable[[Path | None], str]

EXPERIMENTS: dict[str, Experiment] = {
    "laws": lambda cache: phase1.laws(),
    "map": phase1.chart,
    "few": phase2.report,
    "floor": phase2.floor,
    "limits": phase3.report,
    "spread": noise.spread,
    "ceiling": noise.ceiling,
    "confirm": phase4.report,
    "cost": phase4.cost,
}


#: Experiments whose tables hold timings, so a check only confirms they still run.
UNCHECKED = {"cost"}


def moved(stored: str, fresh: str) -> list[str]:
    """Return the lines where a value moved by more than one in its last printed place."""
    old, new = stored.splitlines(), fresh.splitlines()
    if len(old) != len(new):
        return [f"{len(old)} lines stored, {len(new)} lines now"]
    out = []
    for number, (before, after) in enumerate(zip(old, new, strict=True), 1):
        if before == after:
            continue
        a, b = NUMBER.findall(before), NUMBER.findall(after)
        same_words = NUMBER.sub("#", before).split() == NUMBER.sub("#", after).split()
        if not same_words or len(a) != len(b) or not all(map(_close, a, b)):
            out.append(f"line {number}\n  stored: {before}\n  now:    {after}")
    return out


def _close(before: str, after: str) -> bool:
    """Whether two printed numbers are within one in the last printed place."""
    before, after = before.replace(",", ""), after.replace(",", "")
    places = max(len(text.partition(".")[2]) for text in (before, after))
    return abs(float(before) - float(after)) <= 1.5 * 10**-places


def main() -> int:
    """Run the named experiments and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", choices=[*EXPERIMENTS, []], help="default: all")
    parser.add_argument("--check", action="store_true", help="compare with the stored tables")
    parser.add_argument("--cache", type=Path, help="keep raw scores here between runs")
    args = parser.parse_args()
    failed = False
    for name in args.names or EXPERIMENTS:
        start = time.perf_counter()
        fresh = EXPERIMENTS[name](None if args.check else args.cache)
        path = RESULTS / f"{name}.txt"
        took = f"{time.perf_counter() - start:.0f}s"
        if not args.check:
            path.write_text(fresh)
            print(f"{name}: wrote {path.relative_to(Path.cwd())} in {took}")
            continue
        problems = [] if name in UNCHECKED else moved(path.read_text(), fresh)
        failed = failed or bool(problems)
        print(f"{name}: {'MOVED' if problems else 'holds'} in {took}")
        print("\n".join(problems), end="\n" if problems else "")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
