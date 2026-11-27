# Results

What the k experiments found. Every number comes from the tables in this folder, and the
[README](../README.md) says how each rule is judged.

- **Plain `8n` wins.** Its weighted error is 0.300. Fixed 100 has 0.327, and every other rule sits
  between.
- **It wins because it reads noise lowest.** So `8n` wins at any weight above 1.72. At equal weights
  `20n` wins.
- **A floor of 100 is not more accurate.** It reads camps higher and noise higher.
- **The minimum is 11 trials.** Under fixed 100 it is 10.
- **Ten observations a trial is the one limit `8n` makes worse.** Loose camps are the largest limit,
  and no rule for k fixes either.
- **The package now uses `8n`.** Pass `cohesion_k=100` for fixed 100. See [what changing the
  default does](#what-changing-the-default-does).

## Tables

| Table | What's in it |
| :-- | :-- |
| [laws.txt](laws.txt) | The formula check |
| [map.txt](map.txt) | Every draw scored at 25 values of k, the best k at each number of trials and each rule read off it |
| [few.txt](few.txt) | Seven rules on the weighted error and on every principle, from 3 to 400 trials |
| [floor.txt](floor.txt) | The floor against no floor, on nine kinds of noise and eighteen kinds of camps |
| [limits.txt](limits.txt) | `8n` against fixed 100, one part of the data at a time |
| [spread.txt](spread.txt) | Nine kinds of exact noise across k |
| [ceiling.txt](ceiling.txt) | The noise ceiling on trials of finite observations |
| [confirm.txt](confirm.txt) | Fresh seeds, real values, the papers' figures and the grid |
| [cost.txt](cost.txt) | The time of one run at 1,600 trials |

## The best k grows with trials

The weighted error at the best k, and under two rules. The best k is the one with the lowest
weighted error, out of 25 values from 10 to 40,000.

| Trials | Best k | As n | At the best k | At `8n` | At fixed 100 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 4 | 40 | 10n | 0.479 | 0.480 | 0.507 |
| 10 | 159 | 16n | 0.311 | 0.317 | 0.312 |
| 25 | 317 | 13n | 0.214 | 0.221 | 0.247 |
| 100 | 1,260 | 13n | 0.163 | 0.163 | 0.223 |
| 400 | 3,560 | 9n | 0.150 | 0.150 | 0.222 |

- **The best k grows linearly, as the derivation says.** It grows as trials to the power 1.03, and
  from 4 trials it sits at 9n to 16n.
- **`8n` stays within 0.009 of the best k at every number of trials,** and within 0.003 from 50.
- **A fixed k can only be right at one number of trials.** Fixed 100 is close to the best k from 6
  to 13 trials, then falls behind by 0.032 at 25 trials and 0.072 at 400.
- **`4n` was dropped here.** Its weighted error is above that of `8n` at every number of trials from
  4.

The formulas behind `8n` hold where their assumptions hold, see [laws.txt](laws.txt). Real pairs
share trials, so they carry more noise than the formula for N random values says. At `8n` a cloud
falls short by 1.2 times the formula at 10 trials and 8.6 times at 400.

## Plain 8n wins

Each rule over the 13 numbers of trials, on the same draws.

| Rule | Noise error | Camp error | Weighted error | `8n` wins at a weight |
| :-- | :-- | :-- | :-- | :-- |
| `8n` | 0.316 | 0.269 | 0.300 | |
| `20n` | 0.356 | 0.199 | 0.304 | Above 1.72 |
| `√(100² + (8n)²)` | 0.351 | 0.214 | 0.305 | Above 1.55 |
| `max(100, 8n)` | 0.344 | 0.229 | 0.306 | Above 1.39 |
| `max(100, 20n)` | 0.363 | 0.193 | 0.306 | Above 1.59 |
| `100 + 8n` | 0.367 | 0.194 | 0.309 | Above 1.45 |
| Fixed 100 | 0.349 | 0.283 | 0.327 | Any |

- **`8n` has the lowest weighted error.** Every other rule is behind by more than a tie.
- **Every rival but fixed 100 buys its camps with noise.** Each reads camps 0.04 to 0.08 closer to 1
  than `8n` does, and noise 0.03 to 0.05 further from 0.
- **The result leans on the weight.** At equal weights `20n` wins, at 0.278 against 0.292. `8n` wins
  from a weight of 1.72 up, and by more as the weight grows.
- **Fixed 100 loses at any weight.** It is behind on both errors.
- **`8n` is not ahead at every number of trials.** The floor is ahead from 7 to 12 trials, by 0.007
  at most. `20n` is ahead from 9 to 25 trials, by 0.005 at most. `8n` is ahead of both from 3 to 6
  trials, by up to 0.041.

## A floor of 100 makes it worse

The floor against no floor on wider data, from 3 trials. Each cell gives no floor first, then the
floor of 100.

| Trials | Lowest score any data can read | Noise, mean of nine kinds | Kinds of camps the floor tells from noise better, of 18 |
| :-- | :-- | :-- | :-- |
| 3 | 0.09, 0.69 | 0.65, 0.80 | 9 |
| 4 | 0.00, 0.35 | 0.52, 0.64 | 9 |
| 5 | 0.00, 0.11 | 0.45, 0.53 | 17 |
| 6 | 0.00, 0.02 | 0.40, 0.46 | 18 |
| 8 | 0.00, 0.00 | 0.35, 0.37 | 17 |

- **The floor caps the bottom of the scale.** With 3 trials there are 3 pairs. At k = 100, three
  values that could not be further apart still read 0.69.
- **The floor reads seven of the nine kinds of noise higher, at every number of trials to 8.** The
  worst is noise in the corners. At a spread of 0.1, its 95th percentile at 4 trials is 0.78 with
  the floor and 0.54 without.
- **Its gain is real, but it costs more noise than it buys.** From 5 to 8 trials it tells camps from
  noise better on nearly every kind. It does so by reading camps higher by more than it reads noise
  higher. By 10 trials the two are close.

## The minimum is 11 trials

The fewest trials from which tight camps outscore a cloud in 95% of draws, at every larger number
too.

| Rule | 2 tight camps | 4 tight camps |
| :-- | :-- | :-- |
| `8n` | 6 | 11 |
| Fixed 100 and `max(100, 8n)` | 4 | 10 |
| `20n` | 4 | 8 |

- **The docs should say 11.** At 10 trials, 4 tight camps outscore a cloud in 93% of pairings. At 11
  it is 96%, and it stays above 95% at every larger number tested.

### The same camps read higher with more trials

| Population | Fixed 100, from 10 to 400 trials | `8n`, from 10 to 400 trials |
| :-- | :-- | :-- |
| Cloud, mean | 0.31 to 0.18 | 0.31 to 0.15 |
| 2 tight camps, mean | 0.85 to 0.84 | 0.83 to 0.95 |
| 4 tight camps, mean | 0.56 to 0.54 | 0.53 to 0.77 |

- **This is the cost of `8n` on principle 4.** The range is about ten times wider than under fixed
  100.
- **The drift at few trials is downward, which principle 4 allows.** At 3 trials, 4 tight camps read
  0.68 under `8n`, below the 0.77 they read at 400. With the floor they read 0.80, which is above
  it.

## What no rule for k fixes

`8n` against fixed 100, one part of the data at a time. Scores are means unless stated. The last
column says whether comparing with shuffled data, as `floor()` does for incoherence, would fix it.

| Limit | The number | Would a shuffled baseline fix it? |
| :-- | :-- | :-- |
| Too few trials | Under 11 trials, 4 tight camps don't outscore a cloud in 95% of pairings. A cloud reads 0.66 at 3 trials and 0.31 at 10 | Likely. The cause is the number of pairs, which the baseline shares. Not tested |
| Loose camps | 4 camps read 0.5 once looseness reaches 0.1, under both rules. The gain of `8n` is gone by 0.2. From 0.3 it reads up to 0.05 lower than fixed 100 | No |
| Camps that do not read lower as they loosen | At 400 trials, 4 camps read 0.47 at a looseness of 0.3 and 0.49 at 0.5. Under fixed 100 it is 0.50 and 0.53 | No |
| Half camps and half cloud | 2 camps read 0.25 against a cloud's 0.15, at 400 trials | No |
| Many camps | 8 tight camps read 0.47 at 400 trials. Fixed 100 does not tell them from a cloud in 95% of pairings before 400 trials. `8n` does from 50 | No |
| One stray trial | It costs 2 tight camps 0.13 at 10 trials and 0.01 at 400 | No |
| Many options | A cloud over 64 options reads 0.64 at 400 trials, and 0.73 under fixed 100. At 10 trials over 64 options, 2 tight camps outscore it 70% of the time, against 99% under fixed 100 | No. Shuffling gives one source, which reads higher still |
| Narrow clouds | A cloud at a spread of 10 reads 0.83 at 400 trials, and 0.89 under fixed 100 | No. Its trials nearly agree, so read incoherence first |
| Noise in the corners | At a spread of 0.1 the 95th percentile is 0.11 at 25 trials, against 0.06 under fixed 100 | Likely. Not tested |
| Few observations | At 30 observations and 400 trials, 2 tight camps read 0.40 and one source reads 0.77 | No. One source reads above the camps, so a baseline on Cohesion alone points the wrong way |
| Ten observations a trial | A cloud's 95th percentile rises with trials, from 0.09 at 50 to 0.23 at 400. Over 16 options it reaches 0.66 | No |
| Scores at different numbers of trials | See [the same camps read higher](#the-same-camps-read-higher-with-more-trials) | Likely for the cloud. Not tested |

- **Camps stay apart from a cloud far longer than their score stays high.** From 50 trials both
  rules tell camps from a cloud in 95% of pairings up to a looseness of 0.7.
- **Each population mostly reads as expected.** Fewer camps outscore more, 9 to 1 camps read 0.97, a
  stray trial matters less as trials grow, and half cloud sits between cloud and camps. The one
  failure is the bump in loose camps above.

### Kinds of noise across k

The 95th percentile of nine kinds of noise, on a sweep of k. At 3 trials `8n` reads most of them
higher, so the table starts from 4.

| Noise | Trials in a corner | `8n` against fixed 100 |
| :-- | :-- | :-- |
| Spread 0.1 | 53% | Lower from 4 to 10 trials, by up to 0.22. Higher from 13, by 0.050 at most |
| Spread 0.3 | 17% | Lower from 4 to 10 trials. Within 0.005 from 13 |
| Spread 1 | 0% | Tied from 4 to 16 trials. Lower from 25 |
| Spread 3, 10 and 30 | 0% | Higher to 10 trials, by 0.04 at most. Lower from 13 or 16 |
| 2 options | 20% | Higher to 5 trials, by 0.03 at most, and from 16, by 0.027 at most |
| 16 and 64 options | 0% | Higher to 10 trials, by 0.05 at most. Lower from 13, by up to 0.09 |

- **No single k reads every kind of noise lowest.** Noise in the corners reads lowest at the lowest
  k on the sweep. An even cloud reads lowest at a k that grows with trials, from 80 at 10 trials to
  1,260 at 400.
- **Noise in the corners stays low under every rule.** Its 95th percentile under `8n` is 0.11 at 25
  trials and 0.03 at 400. `20n` reads it higher, 0.47 against 0.27 at 10 trials.

### The noise ceiling

Each rule against `8n`, on 40 clouds and 80 kinds of camps with finite observations.

| Rule | Clouds better, tied and worse | Camps better, tied and worse |
| :-- | :-- | :-- |
| `min(8n, ceiling)` | 5, 11 and 24 | 11, 26 and 43 |
| `min(8n, max(100, ceiling))` | 9, 17 and 14 | 10, 31 and 39 |
| Fixed 100 | 9, 9 and 22 | 26, 5 and 49 |

- **Ten observations a trial break `8n`.** Ten observations can only give a few different
  divergences, and a high k reads those as camps. Over 16 options the cloud's 95th percentile is
  0.66 at 400 trials, against 0.10 under fixed 100.
- **The ceiling is too low to use as it stands.** At 30 observations over 16 options it gives k =
  15, and the cloud's 95th percentile at 400 trials is 0.43 against 0.18.
- **Holding the ceiling at 100 or more fixes 10 observations and nothing else.** At 30 and 100
  observations it reads the cloud higher than `8n` does. So the rule has no ceiling, and ten
  observations a trial is a limit for the docs.

## Confirmation

- **Fresh seeds give the same order.** `8n` has a weighted error of 0.300, `20n` 0.304, the floor
  0.305 and fixed 100 0.327. `8n` again wins from a weight of 1.72, and the minimum is 6 and 11
  again.
- **Real values agree from 13 trials.** At 100 trials the weighted error is 0.203 against 0.245
  under fixed 100. At 10 trials fixed 100 is ahead by 0.002.
- **The grid has converged at every k.** A grid four times finer moves Cohesion by 0.004 at most,
  from k = 24 to k = 12,800.
- **1,600 trials is not slow.** The last step takes about 0.1 seconds at k = 12,800, against 0.05 at
  k = 100.

## What changing the default does

| Ensemble | Trials | Fixed 100 | `8n` |
| :-- | :-- | :-- | :-- |
| Figure 2, Current | 5 | 0.991 | 0.993 |
| Figure 2, Option B | 5 | 0.837 | 0.615 |
| Figure 2, Option A | 5 | 0.257 | 0.028 |
| Figures 5 and 6, two patterns | 40 | 0.869 | 0.874 |
| Figures 5 and 6, no patterns | 40 | 0.140 | 0.135 |
| Test ensemble, two camps | 10 | 0.831 | 0.794 |

- **Figure 2 of the Cohesion paper no longer matches.** It has 5 trials, so its k falls to 40,
  and Options A and B fall by 0.22. The order is kept. A floor of 100 would leave all three
  unchanged. `cohesion_k=100` gives the paper's values back.
- **The minimum in the docs moved from 10 trials to 11.**
- **Three limits are new to the [FAQ](../../../docs/faq.md#should-i-change-cohesion_k).** They are
  ten observations a trial, noise in the corners and many options at few trials.
- **With unequal weights the package counts trials by weight,** as `1 / Σw²` of the weights scaled
  to sum to 1. That keeps a trial with almost no weight from moving k. It was not tested here.

## Open questions

- **Is 2 the right weight?** It is a choice. `8n` wins at any weight above 1.72, and `20n` wins at
  equal weights.
- **Should the weighted error use more than one cloud?** It is worked out on one cloud. The [floor
  table](floor.txt) reads nine kinds of noise, but only for the floor.
- **A shuffled baseline would change what k has to do.** It reads a score against noise of the same
  size, so it would take over principles 1 and 4. The rule for k would then only need to keep camps
  and noise apart.
- **A shuffled baseline cannot be read like `floor()`.** With finite observations, one source reads
  higher than camps do. So a score above the baseline would not mean a pattern.
- **Weights and real values with few observations were not tested.**
