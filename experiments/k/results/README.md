# Results

What each phase found and what it decided. Every number comes from the tables in this folder.

- **Plain `8n` wins.** Its [weighted error](../design.md#how-we-judge-a-rule) over the grid is
  0.300. Fixed 100 has 0.327, and every other rule sits between.
- **It wins because it reads noise lowest.** No rule that was run has a lower noise error, so
  `8n` wins at any weight above 1.72. At equal weights `20n` wins.
- **A floor of 100 is not more accurate.** It reads camps higher and noise higher. It also
  stops 3 trials from ever reading below 0.69.
- **The minimum is 11 trials.** Under fixed 100 it is 10.
- **Fixed 100 loses on both errors.** It reads noise higher and camps lower than `8n`.
- **The package default has not changed.** See [what changing it would do](#what-changing-the-default-would-do).

## Tables

| Table | Phase | What's in it |
| :-- | :-- | :-- |
| [laws.txt](laws.txt) | 1 | The formula check |
| [map.txt](map.txt) | 1 | The map, the best k at each number of trials and each rule read off it |
| [few.txt](few.txt) | 2 | Seven rules on the weighted error and on every principle, from 3 to 400 trials |
| [floor.txt](floor.txt) | 2 | The floor against no floor, on nine kinds of noise and eighteen kinds of camps |
| [limits.txt](limits.txt) | 3 | The winner against fixed 100, one part of the data at a time |
| [spread.txt](spread.txt) | Parked | Nine kinds of exact noise across k |
| [ceiling.txt](ceiling.txt) | Parked | The noise ceiling on trials of finite observations |
| [confirm.txt](confirm.txt) | 4 | Fresh seeds, real values, the papers' orderings and the grid |
| [cost.txt](cost.txt) | 4 | The time of one run at 1,600 trials |

## Phase 1. The best k grows with trials

### The formulas hold where their assumptions hold

| Formula | Holds | Breaks |
| :-- | :-- | :-- |
| `8(m − 1) / (k − 8)` | Within 5% from k = 100 for 2 or 3 values. Two values read 0.083 against 0.087 at k = 100 | When bumps overlap. Seven values read 0.47 against 0.52 at k = 100 |
| `1 − k / (16N)` | Within 0.01 when k is well under 16N. 300 values read 0.908 against 0.917 at k = 400 | When k is over 16N. It predicts 0, and 3 values read 0.14 at k = 100 |
| `(s − 1) / (4m ln 2)` | Within 6% from 100 observations over 2 to 16 options | When observations are not many more than options. 64 options and 10 observations read 0.86 against 2.27 |
| `√(2(s − 1)) / (4m ln 2)` | Within 8% over the same range | The same |

- **Unequal shares read lower than the first formula says.** Two values holding 82% and 18% of
  the pairs read 0.064 against 0.087. Swapping m for `(Σ √share)²` gives 0.067.
- **Four exact camps read half the formula at k = 100.** They read 0.26 against 0.52. Their
  centres are random, so some of the seven values sit too close to count as separate.
- **Real pairs carry more noise than the second formula says.** Pairs share trials, so they are
  not N separate values. At `8n` a cloud's entropy falls short by 1.2 times the formula at 10
  trials and 8.6 times at 400.
- **Noise between two camps shrinks as 1 / √m.** The fitted power is −0.51, against −1.00 within
  one source. At 100 observations it is five times wider, 0.045 against 0.009.

### The best k grows linearly

The weighted error at the best k on the map, and under two rules.

| Trials | Best k | As n | At the best k | At `8n` | At fixed 100 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 4 | 40 | 10n | 0.479 | 0.480 | 0.507 |
| 10 | 159 | 16n | 0.311 | 0.317 | 0.312 |
| 25 | 317 | 13n | 0.214 | 0.221 | 0.247 |
| 100 | 1,260 | 13n | 0.163 | 0.163 | 0.223 |
| 400 | 3,560 | 9n | 0.150 | 0.150 | 0.222 |

- **Saw.** The best k grows as trials to the power 1.03. From 4 trials it sits at 9n to 16n.
- **Saw.** `8n` stays within 0.009 of the best k at every number of trials, and within 0.003
  from 50.
- **Saw.** Fixed 100 is close to the best k from 6 to 13 trials, where 100 is about 8n to 16n.
  It then falls behind, by 0.032 at 25 trials and 0.072 at 400.
- **Means.** Growth is linear in n, as the derivation says. A fixed k can only be right at one
  number of trials.
- **Did.** `20n` came back for phase 2, as the best k sits above `8n`. `4n` was dropped. Its
  weighted error is above that of `8n` at every number of trials from 4.
- **Did.** No ceiling of 1,000. The best k had not levelled off by 400 trials.

## Phase 2. Plain 8n wins

Each rule over the grid of 13 numbers of trials, on the same draws.

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
- **Every rival but fixed 100 buys its camps with noise.** Each reads camps 0.04 to 0.08 closer
  to 1 than `8n` does, and noise 0.03 to 0.05 further from 0.
- **The result leans on the weight, and here is how far.** At equal weights `20n` wins, at 0.278
  against 0.292. `8n` wins from a weight of 1.72 up, and by more as the weight grows.
- **Fixed 100 loses at any weight.** It is behind on both errors.
- **`8n` is not ahead at every number of trials.** The floor is ahead from 7 to 12 trials, by
  0.007 at most. `20n` is ahead from 9 to 25 trials, by 0.005 at most. `8n` is ahead of both
  from 3 to 6 trials, by up to 0.041, and ahead of `20n` again from 100.
- **The tails lean the other way at few trials.** The cloud's 95th percentile is 0.962 under
  `8n` and 0.924 with the floor at 3 trials. The two tie from 7 trials.

### The floor on wider data

Each pair is no floor and then the floor of 100.

| Trials | Lowest score any data can read | Noise, mean of nine kinds | Kinds of camps the floor tells from noise better, of 18 |
| :-- | :-- | :-- | :-- |
| 3 | 0.09, 0.69 | 0.65, 0.80 | 9 |
| 4 | 0.00, 0.35 | 0.52, 0.64 | 9 |
| 5 | 0.00, 0.11 | 0.45, 0.53 | 17 |
| 6 | 0.00, 0.02 | 0.40, 0.46 | 18 |
| 8 | 0.00, 0.00 | 0.35, 0.37 | 17 |

- **The floor caps the bottom of the scale.** With 3 trials there are 3 pairs. At k = 100 three
  values that could not be further apart still read 0.69.
- **The floor reads seven of the nine kinds of noise higher, at every number of trials to 8.**
  The worst is noise in the corners. At a spread of 0.1 its 95th percentile at 4 trials is 0.78
  with the floor and 0.54 without. The other two kinds read 0.02 lower at most.
- **By 10 trials the two are close.** The noise reads 0.32 under both, and the floor is better
  on 13 kinds of camps.
- **The floor's gain is real.** From 5 to 8 trials it tells camps from noise better on nearly
  every kind. Over 16 options, 2 tight camps outscore noise 99% of the time against 52% at 5
  trials.
- **It gets that gain by reading camps higher by more than it reads noise higher.** That is
  what the weighted error counts against it.

### The minimum number of trials

| Rule | 2 tight camps | 4 tight camps |
| :-- | :-- | :-- |
| `8n` | 6 | 11 |
| Fixed 100 and `max(100, 8n)` | 4 | 10 |
| `20n` | 4 | 8 |

- **The docs should say 11.** At 10 trials, 4 tight camps outscore a cloud in 93% of pairings.
  At 11 it is 96%, and it stays above 95% at every larger number tested.

### What the winner costs on principle 4

| Population | Fixed 100, from 10 to 400 trials | `8n`, from 10 to 400 trials |
| :-- | :-- | :-- |
| Cloud, mean | 0.31 to 0.18 | 0.31 to 0.15 |
| 2 tight camps, mean | 0.85 to 0.84 | 0.83 to 0.95 |
| 4 tight camps, mean | 0.56 to 0.54 | 0.53 to 0.77 |

- **The same camps read higher with more trials.** The range is about ten times wider than
  under fixed 100.
- **The drift at few trials is downward, which principle 4 allows.** At 3 trials, 4 tight camps
  read 0.68 under `8n`, below the 0.77 they read at 400. With the floor they read 0.80, which
  is above it.

## Phase 3. What no rule for k fixes

The winner is `8n`. Scores are means unless stated.

| Limit | The number | Would a shuffled baseline fix it? |
| :-- | :-- | :-- |
| Too few trials | Under 11 trials, 4 tight camps don't outscore a cloud in 95% of pairings. A cloud reads 0.66 at 3 trials and 0.31 at 10 | Likely. The cause is the number of pairs, which the baseline shares. Not tested |
| Loose camps | 4 camps read 0.5 once looseness reaches 0.1, under both rules. The winner's gain is gone by 0.2. From 0.3 it reads up to 0.05 lower than fixed 100 | No |
| Camps that do not read lower at each step | At 400 trials, 4 camps read 0.47 at a looseness of 0.3 and 0.49 at 0.5. Under fixed 100 it is 0.50 and 0.53 | No |
| Half camps and half cloud | 2 camps read 0.25 against a cloud's 0.15, at 400 trials | No |
| Many camps | 8 tight camps read 0.47 at 400 trials. Fixed 100 does not tell them from a cloud in 95% of pairings before 400 trials. The winner does from 50 | No |
| One stray trial | It costs 2 tight camps 0.13 at 10 trials and 0.01 at 400 | No |
| Many options | A cloud over 64 options reads 0.64 at 400 trials, and 0.73 under fixed 100. At 10 trials over 64 options, 2 tight camps outscore it 70% of the time, against 99% under fixed 100 | No. Shuffling gives one source, which reads higher still |
| Narrow clouds | A cloud at a spread of 10 reads 0.83 at 400 trials, and 0.89 under fixed 100 | No. Its trials nearly agree, so read incoherence first |
| Noise in the corners | At a spread of 0.1 the 95th percentile is 0.11 at 25 trials, against 0.06 under fixed 100 | Likely. Not tested |
| Few observations | At 30 observations and 400 trials, 2 tight camps read 0.40 and one source reads 0.77 | No. One source reads above the camps, so a baseline on Cohesion alone points the wrong way |
| Ten observations a trial | A cloud's 95th percentile rises with trials, from 0.09 at 50 to 0.23 at 400. Over 16 options it reaches 0.66 | No |
| Scores at different numbers of trials | See [principle 4](#what-the-winner-costs-on-principle-4) | Likely for the cloud. Not tested |

- **Camps stay apart from a cloud far longer than their score stays high.** From 50 trials both
  rules tell camps from a cloud in 95% of pairings up to a looseness of 0.7.
- **At 10 trials the winner reads a little below fixed 100,** as its k is 80. Tight camps read
  0.82 against 0.85.
- **What each population should read mostly holds.** Fewer camps outscore more, 9 to 1 camps read
  0.97, a stray trial matters less as trials grow, and half cloud sits between cloud and camps.
  The one failure is the bump in loose camps above.

### Kinds of noise across k

Phase 3 found the winner reading a cloud at a spread of 0.1 above fixed 100, which fired this
part. It reads the 95th percentile of nine kinds of noise on a sweep of k. At 3 trials `8n`
reads most of them higher, and the table starts from 4.

| Noise | Trials in a corner | `8n` against fixed 100 |
| :-- | :-- | :-- |
| Spread 0.1 | 53% | Lower from 4 to 10 trials, by up to 0.22. Higher from 13, by 0.050 at most |
| Spread 0.3 | 17% | Lower from 4 to 10 trials. Within 0.005 from 13 |
| Spread 1 | 0% | Tied from 4 to 16 trials. Lower from 25 |
| Spread 3, 10 and 30 | 0% | Higher to 10 trials, by 0.04 at most. Lower from 13 or 16 |
| 2 options | 20% | Higher to 5 trials, by 0.03 at most, and from 16, by 0.027 at most |
| 16 and 64 options | 0% | Higher to 10 trials, by 0.05 at most. Lower from 13, by up to 0.09 |

- **No single k reads every kind of noise lowest.** Noise in the corners reads lowest at the
  lowest k on the sweep. An even cloud reads lowest at a k that grows with trials, from 80 at
  10 trials to 1,260 at 400.
- **Noise in the corners stays low under every rule.** Its 95th percentile under `8n` is 0.11 at
  25 trials and 0.03 at 400.
- **`20n` reads it higher than `8n` from 4 trials.** At 10 trials it is 0.47 against 0.27.

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
- **The ceiling is too low to use as it stands.** At 30 observations over 16 options it gives
  k = 15, and the cloud's 95th percentile at 400 trials is 0.43 against 0.18.
- **Holding the ceiling at 100 or more fixes 10 observations and nothing else.** At 30 and 100
  observations it reads the cloud higher than `8n` does, and it brings the constant 100 back.
- **So the rule has no ceiling,** and ten observations a trial is a limit for the docs.

## Phase 4. Confirm

- **Fresh seeds give the same order.** Over the grid `8n` has a weighted error of 0.300, `20n`
  0.304, the floor 0.305 and fixed 100 0.327. `8n` again wins from a weight of 1.72.
- **The minimum is 6 and 11 again.**
- **Real values agree from 13 trials.** At 100 trials the weighted error is 0.203 against 0.245
  under fixed 100. At 10 trials fixed 100 is ahead by 0.002.
- **The papers' orderings hold under both rules, but one figure moves a lot.** Figure 2 has 5
  trials, so its k falls to 40. See below.
- **The grid has converged at every k.** A grid four times finer moves Cohesion by 0.004 at
  most, from k = 24 to k = 12,800.
- **1,600 trials is not slow.** The last step takes about 0.1 seconds at k = 12,800, against 0.05
  at k = 100.

## What changing the default would do

| Ensemble | Trials | Fixed 100 | `8n` |
| :-- | :-- | :-- | :-- |
| Figure 2, Current | 5 | 0.991 | 0.993 |
| Figure 2, Option B | 5 | 0.837 | 0.615 |
| Figure 2, Option A | 5 | 0.257 | 0.028 |
| Figures 5 and 6, two patterns | 40 | 0.869 | 0.874 |
| Figures 5 and 6, no patterns | 40 | 0.140 | 0.135 |
| Test ensemble, two camps | 10 | 0.831 | 0.794 |

- **Figure 2 of the Cohesion paper would no longer match.** Options A and B fall by 0.22. The
  order is kept. A floor of 100 would leave all three unchanged.
- **The minimum in the docs moves from 10 trials to 11.**
- **Three limits are new to the docs.** They are ten observations a trial, noise in the
  corners and many options at few trials.
- **Two things were not tested.** They are weights, and real values with few observations.

## Choices made while running

The design left these open, so they are stated here.

- **A cloud is uniform over every mix,** a Dirichlet with a concentration of 1. A camp's centre
  is drawn the same way.
- **A tight camp has a looseness of 0.02.** See the [glossary](../glossary.md).
- **Trials are dealt to camps in turn.** So 4 camps hold fewer than 2 trials each under 8 trials.
- **The camp error pools the 2 and the 4 tight camps,** with the same number of draws of each.
- **The weighted error over the grid is a plain mean** of the 13 numbers of trials.
- **The map's k are whole numbers.** A rule always gives one, and it puts a grid point at 0.
- **Phase 2 runs every number of trials from 3 to 16,** so the minimum is not a guess between
  two tested numbers.
- **The floor table uses 1,000 draws,** and the sweep of noise across k uses every other k on
  the map, up to 5,030, to keep the run short.
- **Phase 4 confirms every rule of phase 2 on fresh seeds,** not just the winner, as the margins
  between rules are small.
