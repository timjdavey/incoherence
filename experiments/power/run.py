"""Run the power experiments and store their tables, or check the stored tables still hold.

Run it from the repository root::

    uv run python experiments/power/run.py              # every experiment
    uv run python experiments/power/run.py laws map     # some of them
    uv run python experiments/power/run.py --check      # rerun and compare with what is stored
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "k"))

import divergences
import figures
import laws
import observations
import runner

RESULTS = Path(__file__).parent / "results"

EXPERIMENTS: dict[str, runner.Experiment] = {
    "laws": lambda cache: laws.report(),
    "observations": lambda cache: observations.report(),
    "map": divergences.chart,
    "rules": divergences.rules,
    "wide": divergences.wide,
    "figures": lambda cache: figures.report(),
}


if __name__ == "__main__":
    sys.exit(runner.main(EXPERIMENTS, RESULTS, __doc__.splitlines()[0]))
