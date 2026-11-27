"""Run the k experiments and store their tables, or check the stored tables still hold.

Run it from the repository root::

    uv run python experiments/k/run.py              # every experiment
    uv run python experiments/k/run.py laws map     # some of them
    uv run python experiments/k/run.py --check      # rerun and compare with what is stored
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import noise
import phase1
import phase2
import phase3
import phase4
import runner

RESULTS = Path(__file__).parent / "results"

EXPERIMENTS: dict[str, runner.Experiment] = {
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


if __name__ == "__main__":
    sys.exit(runner.main(EXPERIMENTS, RESULTS, __doc__.splitlines()[0], unchecked={"cost"}))
