# Results

What the power experiments found. Every number comes from the tables in this folder, and the
[README](../README.md) says how each candidate is judged.

- **Power 1 wins on noise and at few trials.** Every power 1 readout reads noise lower than the
  package. Six of the eight rivals have a lower weighted error, from 0.261 to 0.295 against 0.301.
  Most of that lead comes from reading camps low, which the weighted error punishes half as much.
- **Power 0.5 reads clean camps near 1.** Under the package, two tight camps read 0.83 to 0.95.
  Every power 1 readout but one reads them at 0.70 or below.
- **The one power 1 readout that reaches the top runs out of room there.** Doubled and capped, it
  reads one camp, a 9 to 1 split and an even pair all at 0.96 or more.
- **The package wins from 50 trials.** Its weighted error there is 0.162. The best rival has 0.193.
- **Level 1 is the other way round.** Power 1 sits about twice as close to Shannon entropy, and
  power 0.5 moves about four times too far when 1% of a trial strays. Power 0.5 detects a little
  better, 0.800 against 0.778.
- **So the package keeps both powers.** Nothing in the code changes.

## Tables

| Table | What's in it |
| :-- | :-- |
| [laws.txt](laws.txt) | What each power reads on exact values, against a formula for each |
| [observations.txt](observations.txt) | Level 1: Shannon entropy, strays, few observations and detection |
| [map.txt](map.txt) | Every readout at 25 values of `k`, and how its best `k` grows |
| [rules.txt](rules.txt) | Each candidate at its rule on fresh seeds, against the package |
| [wide.txt](wide.txt) | 1 to 8 camps, loose and uneven camps, six kinds of noise and real values |
| [figures.txt](figures.txt) | The toy figure of error-cohesion, two fully certain camps and the papers' figures |

## The formulas hold

| Setup, at `k = 400` | Power 1, `1 − H` | Formula | Power 0.5, `H` | Formula |
| :-- | :-- | :-- | :-- | :-- |
| 2 values, equal shares | 0.4950 | 0.4949 | 0.0201 | 0.0204 |
| 7 values, equal shares | 0.1339 | 0.1342 | 0.1221 | 0.1224 |
| 2 values, shares 99 and 1 | 0.9800 | 0.9800 | 0.0040 | 0.0041 |
| 300 random values | 0.0033 | 0.0033 | 0.909 | 0.917 |

- **Power 1 reads `1 − H` as the chance two values match.** For up to 7 separate values it holds to
  within 0.001 at every `k` from 100. For N random values it is `1 / N` at any `k`, to within 0.006.
- **Power 0.5 reads `H` as one step for each separate value.** A stray of 1% counts a fifth of a
  step. For random values `1 − k / (16N)` holds once there are many of them, as in the
  [k experiments](../../k/results/laws.txt).

## The best k only grows at power 0.5

| Readout | Best `k` grows as | At the best `k` for each n | At the best fixed `k` |
| :-- | :-- | :-- | :-- |
| `0.5 squared` | `n^0.92` | 0.297 | 0.325, at 159 |
| `1 squared` | `n^−0.14` | 0.300 | 0.303, at 56 |
| `1 plain` | `n^0.00` | 0.292 | 0.292, at 159 |
| `1 doubled, capped` | `n^0.00` | 0.252 | 0.252, at 159 |
| `1 - H squared`, that is `1 − H²` | `n^0.00` | 0.275 | 0.275, at 224 |
| `1 log squared` | `n^0.97` | 0.265 | 0.279, at 224 |

- **Power 1 needs no rule for k.** Its best `k` stays put as trials grow, or even falls. So a fixed
  `k` costs it 0.003 at most. That follows from `1 − H = 1 / N` at any `k`, above.
- **Power 0.5 needs `k` to grow.** A fixed `k` costs it 0.028, which is why the package uses `8n`.
- **Reading power 1 as a count brings the growth back.** The log count readout counts values, as
  power 0.5 does.

## Power 1 wins on noise and at few trials

Each candidate at its rule on fresh seeds, over 13 numbers of trials from 3 to 400. Less the
package is the 95% interval on the difference in weighted error.

| Candidate | Noise | 2 tight camps | Weighted error | Less the package | From 50 trials |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `0.5 squared, 8n`, the package | 0.317 | 0.846 | 0.301 | | 0.162 |
| `0.5 squared, 100` | 0.350 | 0.852 | 0.328 | +0.026 to +0.029 | 0.223 |
| `1 squared, 56` | 0.031 | 0.257 | 0.301 | −0.001 to +0.003 | 0.290 |
| `1 plain, 100` | 0.106 | 0.468 | 0.295 | −0.008 to −0.004 | 0.262 |
| `1 plain, 160` | 0.088 | 0.438 | 0.291 | −0.012 to −0.007 | 0.259 |
| `1 doubled, capped, 100` | 0.211 | 0.902 | 0.261 | −0.042 to −0.038 | 0.197 |
| `1 - H squared, 100` | 0.192 | 0.710 | 0.287 | −0.016 to −0.012 | 0.231 |
| `1 log squared, 200` | 0.146 | 0.616 | 0.278 | −0.025 to −0.020 | 0.216 |
| `1 log squared, 20n` | 0.125 | 0.605 | 0.265 | −0.038 to −0.034 | 0.193 |

- **Power 1 reads noise 1.5 to 10 times lower.** Plain at 100 reads a cloud at 0.106, a third of
  the package's 0.317.
- **It tells camps from noise sooner.** At 5 trials a camp draw outscores a cloud draw 0.86 of
  the time under plain at 100, against 0.72 under the package.
- **The package catches up and passes them.** From 50 trials its weighted error is 0.162, and the
  best rival's is 0.193.
- **Most of the lead is bought by reading camps low.** The weighted error rewards a low noise
  reading twice as much as a high camp reading. Plain at 100 reads two tight camps at 0.47.

## Power 0.5 reads clean camps near 1

Principles 5 and 6, added for this experiment, from [wide.txt](wide.txt), at 10, 50 and 400 trials.

| Candidate | 2 tight camps | 1 camp at 30 observations, lowest | Principle 6 order holds |
| :-- | :-- | :-- | :-- |
| `0.5 squared, 8n`, the package | 0.83 to 0.95 | 0.760 | Yes |
| `0.5 squared, 100` | 0.85 | 0.817 | Yes |
| `1 squared, 56` | 0.24 to 0.26 | 0.285 | Yes |
| `1 plain, 100` | 0.45 to 0.46 | 0.364 | Yes |
| `1 plain, 160` | 0.42 to 0.43 | 0.250 | Yes |
| `1 doubled, capped, 100` | 0.88 to 0.89 | 0.727 | No, one camp and 9 to 1 both read 1 |
| `1 - H squared, 100` | 0.69 to 0.70 | 0.595 | Yes |
| `1 log squared, 200` | 0.60 to 0.61 | 0.379 | Yes |
| `1 log squared, 20n` | 0.58 to 0.60 | 0.119 | Yes |

- **Every power 1 readout but the cap reads clean camps at 0.70 or below.** Two camps are the
  clearest pattern Cohesion looks for, so they should read near 1.
- **One camp drops furthest at power 1.** Thirty observations a trial make one camp look loose.
  Plain at 100 reads it at 0.36 to 0.39, against 0.76 to 0.87 for the package.
- **The cap reaches the top by throwing the top away.** Anything with `1 − H` of a half or more
  reads 1. One camp and a 9 to 1 split read 1.000, and two exact even camps 0.96 to 0.98, so they
  can't be told apart.
- **Its lead fades with trials.** The cap is ahead from 4 to 25 trials and behind from 50, where the
  package reads two tight camps at 0.93 and four at 0.70, against 0.88 and 0.31.

## The figures keep their order

| Ensemble | Package | `0.5 squared, 100` | `1 plain, 100` | `1 doubled, capped` | `1 log squared, 200` |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Toy, two camps | 0.828 | 0.765 | 0.335 | 0.670 | 0.484 |
| Toy, five camps | 0.352 | 0.274 | 0.078 | 0.157 | 0.115 |
| Toy, no camps | 0.033 | 0.027 | 0.017 | 0.033 | 0.010 |
| Two fully certain camps, 14 and 26 | 0.950 | 0.842 | 0.482 | 0.964 | 0.680 |

- **Every candidate keeps every group in its source's order.** That holds for the toy and for
  Figures 2, 5 and 6 of the Cohesion paper, see [figures.txt](figures.txt).
- **Only power 0.5 and the cap read two certain camps near 1.** Power 1 plain reads the toy's two
  camps at 0.34.

## Level 1 is the other way round

Each trial's entropy, from [observations.txt](observations.txt).

| Test | Right answer | Power 1 | Power 1, `k = 20` | Power 0.5 | Power 0.5, `k = 20` | Power 0.5, wide grid |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Gap to Shannon entropy, each at its closest bins: 5, 15, 3, 5 and 3 | 0 | 0.066 | 0.040 | 0.139 | 0.122 | 0.154 |
| Rank agreement with Shannon in 5 bins | 1 | 0.950 | 0.965 | 0.892 | 0.933 | 0.881 |
| Rise when 1% of a trial strays | 0.021, Shannon in 10 bins | 0.028 | 0.020 | 0.085 | 0.082 | 0.090 |
| Bias at 5 observations | 0 | 0.148 | 0.164 | 0.132 | 0.189 | 0.138 |
| Detection, mean of eight kinds | 1 | 0.777 | 0.778 | 0.783 | 0.800 | 0.780 |

- **Power 1 sits about twice as close to Shannon entropy.** No repair to power 0.5 closes the gap.
- **Power 0.5 moves about four times too far for strays.** The root lifts low densities, so a few
  far observations count for more than their share.
- **Power 0.5 detects a little better, and the control shows why.** At the same `k = 20` it
  detects at 0.800, against 0.778 for power 1, so the gap comes from the power. Most of it is in
  spotting scattered observations, at 0.77 against 0.65. That is the same pull towards strays as
  the row above.
- **Power 1 at `k = 20` sits closer to Shannon still,** at 15 bins. The package keeps `k = 10`,
  the papers' value. This is noted, not acted on.
