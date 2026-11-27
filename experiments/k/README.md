# Choosing Cohesion's k

Why we tested Cohesion's `k`, how we judged each rule for it, and how to run the tests. What they
found is in the [results](results/README.md).

- **k is Cohesion's resolution.** Cohesion's last step measures how bunched the pairwise divergences
  are. Each divergence is spread into a narrow bump, and `k` sets how narrow it is.
- **The Cohesion paper fixed k at 100.** The package now uses `8n` by default, from
  `cohesion_k()` in `src/incoherence/_measures.py`. Pass `cohesion_k=100` to `measure` for the
  paper's value.
- **Plain `8n` wins.** With n trials, `k = 8n` has a [weighted error](#how-a-rule-is-judged) of
  0.300, against 0.327 for fixed 100. Adding a floor of 100 makes it worse. See [the
  results](results/README.md#plain-8n-wins).
- **The minimum is 11 trials.** That is the fewest at which tight camps outscore noise in 95% of
  draws. Under fixed 100 it is 10. See [the minimum](results/README.md#the-minimum-is-11-trials).
- **The package default is now `8n`.** It moves Figure 2 of the Cohesion paper by 0.22. See [what
  changing the default does](results/README.md#what-changing-the-default-does).

## Words

| Word | What it means |
| :-- | :-- |
| Trial | One distribution, or one set of observations, in the ensemble |
| Draw | One seeded ensemble. Each number of trials gets 200 draws. Picking the rule uses 2,000 from 3 to 16 trials, where the tails decide |
| Cloud | Every trial scattered at random, with no camps. It is the noise case, so it should read 0 |
| Camp | A group of trials that sit close together. Tight camps should read 1 |
| Looseness | How far a camp's trials sit from its centre. 0 is an exact camp, 0.02 is a tight camp and 1 is a cloud |
| Spread | How widely a cloud scatters its trials. 1 is uniform, below 1 pushes trials to the corners and above 1 pulls them to the middle |
| Rule | A way of setting k from the number of trials, such as fixed 100 or `8n` |
| Tie | Two rules whose difference could be chance. See [how ties are called](#how-ties-are-called) |

## Where 8n comes from

Two formulas give the error at each end. Both are checked against measured values in
[laws.txt](results/laws.txt).

| Formula | Setup | Predicts | True value |
| :-- | :-- | :-- | :-- |
| `8(m − 1) / (k − 8)` | m exactly separate divergence values | The entropy read | 0 |
| `1 − k / (16N)` | N random divergence values | The entropy read | 1 |

- **Too low a k blurs camps, and too high a k resolves noise.** The two errors are equal
  when `k(k − 8) = 128N`. With N = n(n − 1) / 2 pairs, that is exactly `k = 8n`.
- **The constant assumes two camps.** Four exact camps give seven separate values, and the balance
  moves to about `20n`. So the derivation fixes the form, and the constant is a guess at how many
  camps there are.
- **Counting noise twice pulls the balance down,** to about `6n` for two camps and `14n` for four.
- **Sampling noise sets a ceiling on useful k.** Two trials of m observations over s options from
  one source differ only by chance. A bump narrower than that chance gap resolves noise, which gives
  a ceiling of `2m / √(s − 1)`. It was tested and [not used](results/README.md#the-noise-ceiling).

## How a rule is judged

| # | Principle | Measured by |
| :-- | :-- | :-- |
| 1 | Noise must not read as a pattern | The noise error, which is the cloud's mean |
| 2 | Clear camps read high, and in the right order | The camp error, which is what the mean of 2 and 4 tight camps falls short of 1 |
| 3 | Camps and noise stay apart | How often a camp draw outscores a cloud draw |
| 4 | The same population reads the same at any number of trials | The range of the mean across n. Any drift at few trials must be downward |

- **Raising k reads everything higher, noise and camps alike.** So principles 1 and 2 pull against
  each other, and a rule judged on one of them alone always wins it.
- **The weighted error puts both on one scale.** It is twice the noise error plus the camp error,
  all divided by 3. So 0 is perfect and 1 is as wrong as a rule can be.
- **A false pattern counts double.** The weight of 2 is our choice, not a finding. The results give
  the weight at which each rival would win instead.
- **The rule with the lowest weighted error wins.** It is the plain mean over 13 numbers of trials
  from 3 to 400, spaced close to evenly in the log of trials.
- **Principles 3 and 4 break ties, in order.** After that a tie goes to the rule that reads lower on
  noise, then the simpler one.

### A worked example

A cloud and tight camps at 4 trials, with and without a floor of 100.

| Rule | k | Noise error | Camp error | At equal weights | Weighted error |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `8n`, no floor | 32 | 0.527 | 0.366 | 0.446 | 0.473 |
| `8n`, floor of 100 | 100 | 0.633 | 0.250 | 0.441 | 0.505 |

- **The floor reads camps 0.12 closer to 1, and noise 0.11 further from 0.** It has not made
  Cohesion more accurate. It has read everything higher.
- **At equal weights the two are close to a draw.** The floor is ahead by 0.005.
- **With a false pattern counted double, no floor wins.** Its weighted error is
  `(2 × 0.527 + 0.366) / 3 = 0.473`, against 0.505 with the floor.

### How ties are called

- **Every rule scores the same draws.** Two rules are compared draw by draw, so the luck of the draw
  cancels.
- **Two rules tie when the interval on their difference includes zero.** The draws are resampled
  1,000 times, and the interval is the middle 95% of the difference.

## What it runs on

| Part | Values |
| :-- | :-- |
| Trials | 3, 4, 5, 6, 8, 10, 13, 16, 25, 50, 100, 200 and 400. Picking the rule also runs every number from 3 to 16 |
| Shapes | A cloud, and 2 and 4 tight camps. Trials are dealt to camps in turn |
| Data | Exact probabilities over 4 options, so k is the only thing moving |
| Cloud and camp centres | Uniform over every mix, a Dirichlet with a concentration of 1 |
| Wider data | Spreads, options, observations per trial, looseness and uneven camps, each moved one at a time |
| Confirmation | Fresh seeds, real values up to 100 trials, the papers' figures and one run at 1,600 trials |

- **The measure itself is held fixed.** Cohesion's last step keeps its power of 0.5, its squaring
  and the divergence. Each trial's continuous entropy keeps power 1, as the code has it. The rule is
  picked on exact probabilities, so that second power never enters the choice. Why the two powers
  differ is tested in [experiments/power](../power/README.md).
- **The density grid follows k.** `points_for(k)` in `src/incoherence/_entropy.py` gives `20k + 1`
  points. It is checked for convergence at every k a rule reaches.
- **The k for real values stays at 10.** It is `DEFAULT_K` in `src/incoherence/_api.py`.

## Running

Run these from the repository root.

| What | Command |
| :-- | :-- |
| Run every experiment and store its table | `uv run python experiments/k/run.py` |
| Run some of them | `uv run python experiments/k/run.py laws map` |
| Check the stored tables still hold | `uv run python experiments/k/run.py --check` |
| Keep raw scores between runs, while working on the analysis | `uv run python experiments/k/run.py --cache .context/k-cache` |

| File | What it runs | Table |
| :-- | :-- | :-- |
| `common.py` | Populations, scoring at an explicit k, the rules, the weighted error and the tie rule | |
| `phase1.py` | The formula check, and every draw scored at 25 values of k | `laws.txt`, `map.txt` |
| `phase2.py` | Every rule on the weighted error, then the floor on wider data | `few.txt`, `floor.txt` |
| `phase3.py` | What no rule fixes, one part of the data at a time | `limits.txt` |
| `noise.py` | Kinds of noise across k, and the noise ceiling | `spread.txt`, `ceiling.txt` |
| `phase4.py` | Fresh seeds, real values, the papers' figures, the grid and the cost | `confirm.txt`, `cost.txt` |
| `run.py` | The list of experiments, for the runner and the check in `experiments/runner.py` | |

- **A full run takes about seven and a half minutes on eight cores.** The result does not depend on
  the number of cores.
- **Every experiment passes k explicitly,** so the results don't depend on the package default.
- **A `--check` run reruns every draw from its seed.** It never reads the cache, and it fails when
  any value moves by more than one in the last printed place. `cost.txt` holds timings, so it is
  only run.
- **CI does not run the check.** Run it yourself after any change to the estimator. If a table
  moves, rerun it and commit it with the reason.
