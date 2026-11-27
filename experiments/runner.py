"""Run a set of experiments and store their tables, or check the stored tables still hold.

Each experiment folder has a ``run.py`` that names its experiments and hands
them to :func:`main`. An experiment takes a cache folder, or ``None``, and
returns its table as text.
"""

import argparse
import re
import time
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")

Experiment = Callable[[Path | None], str]


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


def main(
    experiments: Mapping[str, Experiment],
    results: Path,
    description: str,
    unchecked: Collection[str] = (),
) -> int:
    """Run the experiments named on the command line and return an exit code.

    Parameters
    ----------
    experiments
        Each experiment by name, in the order a full run takes them.
    results
        The folder that holds each experiment's table, as ``<name>.txt``.
    description
        One line for ``--help``.
    unchecked
        Experiments whose tables hold timings, so a check only confirms they still run.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("names", nargs="*", choices=[*experiments, []], help="default: all")
    parser.add_argument("--check", action="store_true", help="compare with the stored tables")
    parser.add_argument("--cache", type=Path, help="keep raw scores here between runs")
    args = parser.parse_args()
    failed = False
    for name in args.names or experiments:
        start = time.perf_counter()
        fresh = experiments[name](None if args.check else args.cache)
        path = results / f"{name}.txt"
        took = f"{time.perf_counter() - start:.0f}s"
        if not args.check:
            results.mkdir(parents=True, exist_ok=True)
            path.write_text(fresh)
            print(f"{name}: wrote {path.resolve().relative_to(Path.cwd())} in {took}")
            continue
        problems = [] if name in unchecked else moved(path.read_text(), fresh)
        failed = failed or bool(problems)
        print(f"{name}: {'MOVED' if problems else 'holds'} in {took}")
        print("\n".join(problems), end="\n" if problems else "")
    return int(failed)
