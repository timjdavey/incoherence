# Running

How the experiment code is built, how to run it, where its results go and how CI checks them.

## Commands

Run these from the repository root.

| What | Command |
| :-- | :-- |
| Run every experiment and store its table | `uv run python experiments/k/run.py` |
| Run some of them | `uv run python experiments/k/run.py laws map` |
| Check the stored tables still hold | `uv run python experiments/k/run.py --check` |
| Keep raw scores between runs, while working on the analysis | `uv run python experiments/k/run.py --cache .context/k-cache` |

- **A full run takes about seven and a half minutes on eight cores.** The map and the sweep of noise across k are
  most of it. The cells run in parallel, and the result does not depend on the number of cores.
- **A `--check` run never reads the cache.** It reruns every draw from its seed.

## Files

| File | What it runs | Table |
| :-- | :-- | :-- |
| `common.py` | Populations, scoring at an explicit k, the rules, the weighted error, the measures and the tie rule | |
| `phase1.py` | The formula check and the map | `laws.txt`, `map.txt` |
| `phase2.py` | Every rule on the weighted error, then the floor on wider data | `few.txt`, `floor.txt` |
| `phase3.py` | The limits, one part of the data at a time | `limits.txt` |
| `parked.py` | The two parked parts phase 3 sent back, which are kinds of noise across k and the noise ceiling | `spread.txt`, `ceiling.txt` |
| `phase4.py` | Fresh seeds, real values, the papers' orderings, the grid and the cost | `confirm.txt`, `cost.txt` |
| `run.py` | The runner and the check | |

## How it is built

- **Numpy, scipy and the package only.** No plots.
- **Every experiment passes k explicitly,** so results don't depend on the package default.
  `_measures.cohesion()` reads `COHESION_K` and `COHESION_POINTS` itself, so the experiments
  call `densities` and `density_entropy` directly, with `points_for(k)`.
- **Divergences are worked out once per draw.** Every k and every rule then reads the same
  divergences, so the map costs little more than one rule.
- **Every draw has its own seed.** It is made from the root seed, the phase, the population, the
  number of trials and the draw, so cells can run in any order.
- **Each phase stores its results before the next one is written.** They are text tables under
  `experiments/k/results/`, with what each phase decided in the [results](results/README.md).
- **The folder sits outside `src/`,** so it is not in the wheel.

## How CI checks it

- **A `--check` run reruns the tables** and fails when any value moves by more than one in the
  last printed place.
- **CI checks three of them:** `laws`, `few` and `confirm`. Together they cover the formulas, the
  comparison that picks the rule and the real-valued route. The full check is too slow for every
  pull request, so run it yourself after any change to the estimator.
- **CI runs it on the locked versions only.**
- **`cost.txt` holds timings from one machine,** so a check only confirms it still runs.
- **If a table moves, rerun it and commit it,** with the reason in the pull request.
