# Why Cohesion uses two powers

Why the package reads entropy at power 1 for each trial and at power 0.5 in Cohesion's last step,
how we judged each candidate, and how to run the tests. What they found is in the
[results](results/README.md).

- **The package runs one entropy estimate twice.** It reads each trial's observations at power 1.
  It reads the pairwise divergences, in Cohesion's last step, at power 0.5.
- **The power decides what "spread out" means.** Power 1 follows the biggest bunch of values.
  Power 0.5 counts how many separate values there are. No choice of `k` turns one into the other.
- **Each level needs a different one, so the package keeps both.** A trial's entropy should track
  Shannon entropy and shrug off a few stray observations. Cohesion asks how many separate distances
  there are. So `COHESION_POWER` stays 0.5 and the default `k` stays `8n`. See [the
  results](results/README.md) for what each power wins and loses.

## Words

| Word | What it means |
| :-- | :-- |
| Power | What each density is raised to before its variance is taken, the `power` of `density_entropy` |
| Level 1 | Each trial's entropy, from its observations |
| Level 2 | Cohesion's last step, the entropy of the pairwise divergences |
| Readout | How a level 2 entropy `H` becomes Cohesion, such as `(1 − H)²` |
| Candidate | A readout with a rule for its `k`, such as power 0.5 squared at `8n` |
| Separate values | Values far enough apart that their bumps don't overlap |

The words of the [k experiments](../k/README.md#words), such as trial, draw, cloud, camp and
looseness, mean the same here.

## What each power reads

Each formula is checked on exact values in [laws.txt](results/laws.txt).

| | Power 1 | Power 0.5 |
| :-- | :-- | :-- |
| m separate values, shares p | `1 − H = (kΣp² − 4) / (k − 4)`, the chance two values match | `H = 8((Σ√p)² − 1) / (k − 8)`, one equal step for each value |
| N random values | `1 − H = 1 / N`, at any `k` | `H = 1 − k / (16N)` |
| What dominates | The biggest bunch. A stray value barely counts | The number of places. A stray counts for more |
| Needs `k` to grow with the data | No | Yes, which is the `8n` rule |
| Right for | Level 1: it tracks Shannon entropy and ignores strays | Level 2: the question is how many distances there are |

### A worked example

Two clean camps of trials give two separate divergence values. Pairs within a camp sit near 0 and
pairs across sit at one distance. An even split puts about half the pairs at each. A 9 to 1 split
puts 82% within, since `0.9² + 0.1² = 0.82`. At `k = 400`:

| Shares of pairs | Power 1, `1 − H` | Power 0.5, `H` |
| :-- | :-- | :-- |
| 50 and 50, an even split | 0.495 | 0.020 |
| 82 and 18, a 9 to 1 split | 0.702 | 0.016 |
| 99 and 1, almost one camp | 0.980 | 0.004 |

- **Power 0.5 reads two values as nearly as bunched as one.** `H` is 0.02, so Cohesion is
  `(1 − 0.02)² = 0.96`.
- **Power 1 reads the same two values as half bunched.** Squared as the paper does, Cohesion is
  `0.495² = 0.25`. Two clean camps should read near 1, so power 1 needs a new readout.

## How a candidate is judged

- **The four principles and the weighted error of the k experiments, unchanged.** See [how a rule
  is judged](../k/README.md#how-a-rule-is-judged). Noise should read 0, tight camps 1, and a false
  pattern counts double.
- **Two more principles, added after seeing the candidates.** We say so because they were not
  fixed in advance.

| # | Principle | Measured by |
| :-- | :-- | :-- |
| 5 | One camp reads near 1 at any number of observations a trial | One camp at 30 observations a trial |
| 6 | Fewer camps read higher, and clean camps near the top | One camp, then 2 camps split 9 to 1, then 2, 3 and 4 even camps, falling in that order |

- **Level 1 is judged on tests with a right answer.** Agreement with Shannon entropy on the same
  data binned, the rise from stray observations, bias at few observations and telling a real
  difference from none.

### The candidates

Every candidate scores the same draws. One pass keeps three raw numbers per draw at every `k`, and
each candidate is read off those.

| Candidate | Readout | `k` |
| :-- | :-- | :-- |
| `0.5 squared, 8n` | Power 0.5, `(1 − H)²`, the package | `8n` |
| `0.5 squared, 100` | The same, as in the papers | 100 |
| `1 squared, 56` | Power 1, `(1 − H)²`, the paper with only the power swapped | 56, its best fixed `k` |
| `1 plain` | Power 1, `1 − H` | 100 and 160 |
| `1 doubled, capped` | Power 1, `2(1 − H)` capped at 1 | 100 |
| `1 - H squared` | Power 1, `1 − H²`, not `(1 − H)²` | 100 |
| `1 log squared` | Power 1 read as a count of values, `1 / Σp²`, then the log of that count over the log of the most, squared | 200 and `20n` |

### Level 1 runs five ways

Taking the root of each density makes each observation's bump reach twice as far, so the package's
grid cuts off part of power 0.5's tail. Two variants repair that.

| Variant | Why |
| :-- | :-- |
| Power 1, `k = 10` | The package |
| Power 1, `k = 20` | The control, so a gap between the powers can't come from `k` alone |
| Power 0.5, `k = 10` | The package with only the power swapped |
| Power 0.5, `k = 20` | Narrower bumps, so less of the tail is cut off |
| Power 0.5, wide grid | The grid runs 1.5 past each end, so none is cut off |

## Running

Run these from the repository root.

| What | Command |
| :-- | :-- |
| Run every experiment and store its table | `uv run python experiments/power/run.py` |
| Run some of them | `uv run python experiments/power/run.py laws map` |
| Check the stored tables still hold | `uv run python experiments/power/run.py --check` |
| Keep raw numbers between runs, while working on the analysis | `uv run python experiments/power/run.py --cache .context/power-cache` |

| File | What it runs | Table |
| :-- | :-- | :-- |
| `candidates.py` | The raw numbers per draw, and every candidate read off them | |
| `laws.py` | What each power reads on exact values | `laws.txt` |
| `observations.py` | Level 1 at each power | `observations.txt` |
| `divergences.py` | Level 2: the map across `k`, each candidate at its rule, and wider data | `map.txt`, `rules.txt`, `wide.txt` |
| `figures.py` | The toy figure of error-cohesion and the papers' figures | `figures.txt` |
| `run.py` | The list of experiments, for the runner in `experiments/runner.py` | |

- **It shares the k experiments' code.** Populations, the weighted error, resampling and tables
  come from `experiments/k/common.py`, and the papers' figures from `experiments/k/phase4.py`.
- **A `--check` run reruns every draw from its seed.** It never reads the cache, and it fails when
  any value moves by more than one in the last printed place.
- **The laws check the package first.** `laws.py` stops unless the raw numbers give the package's
  Cohesion to within 1e-12, and `observations.py` stops unless its power 1 gives the package's
  incoherence.
- **CI does not run these.** One test in `tests/test_measures.py` checks both formulas on exact
  values.
