# Design

How the experiment picks a rule for k: the phases, how a rule is judged and the data it runs on.
What it found is in the [results](results/README.md).

- **It runs in four phases.** Each phase asks one question, and its answer decides what the next
  phase needs.
- **Most rules are never run.** They are read off the [map](glossary.md), and parts that need a
  result to justify them are [parked](scope.md#parked).

## Phases

| Phase | Question | What runs | What it decides |
| :-- | :-- | :-- | :-- |
| 1. Laws and map | Does the best k grow with trials, and how fast? | The [formula check](derivation.md#the-formulas-to-check) on made up divergence values. Then the map over the [phase 1 grid](#what-we-test-against) | Picks the growth, or keeps fixed 100 |
| 2. Few trials | Which rule has the lowest weighted error, and what is the minimum number of trials? | The growth from phase 1, with and without a floor of 100, from 3 to 400 trials. Then the floor on [wider data](#what-we-test-against) | Picks the rule and the minimum |
| 3. Limits | What does no rule fix? | The winner against fixed 100, one part of the data at a time | Gives the list for the docs, and fires the [parked](scope.md#parked) triggers |
| 4. Confirm | Does it hold on data not used to choose? | Fresh seeds, real values up to 100 trials, the papers' orderings, and one run at 1,600 trials | Change the default, or report why not |

- **The map replaces most rule runs.** It scores every draw at 25 values of k from 10 to 40,000.
  A rule is a path through it, so a rule that was never run can still be read off.
- **The best k has the lowest weighted error on the map,** see
  [how we judge a rule](#how-we-judge-a-rule).
- **Phase 1 keeps fixed 100 if it sits inside the band at every number of trials.** Phases 2 and
  3 then run for fixed 100 alone, to give the minimum and the limits.
- **Otherwise phase 1 carries the rules the derivation gives.** The power of n that the best k
  grows by says which. Near 1 it is `8n`, with `4n` or `20n` beside it if the best k sits
  clearly below or above `8n`.
- **Phase 2 picks the rule.** Two or three rules are left by then, so each gets 2,000 draws at
  few trials, where the rules differ most.
- **Phase 3 never changes the rule on its own.** It can only fire a trigger, which sends one
  parked part back through phases 1 and 2.

## How we judge a rule

| # | Principle | Measured by |
| :-- | :-- | :-- |
| 1 | Noise must not read as a pattern | The noise error, which is the cloud's mean. Noise should read 0 |
| 2 | Clear camps read high, and in the right order | The camp error, which is what the mean of the tight camps falls short of 1. Then how often fewer camps score above more camps |
| 3 | Camps and noise stay apart | How often a camp draw outscores a cloud draw, then the camps' 5th percentile minus the cloud's 95th |
| 4 | The same population reads the same at any number of trials | The range of the mean across n, and any drift at few trials must be downward |
| 5 | The rule has a derivation and no tuned constants | Judgement |

- **Raising k reads everything higher, noise and camps alike.** So principles 1 and 2 pull
  against each other, and a rule that is judged on one of them alone always wins it.
- **The weighted error puts both on one scale.** It is the noise error counted twice, plus the
  camp error, all divided by 3. So 0 is perfect and 1 is as wrong as a rule can be.
- **A false pattern counts double.** The weight of 2 is our choice and not a finding. It says
  Cohesion should lean pessimistic. Phase 2 reports the weight at which each rival would win
  instead, so you can see how far the result leans on it.
- **The rule with the lowest weighted error over the grid wins.** The grid is the 13 numbers of
  trials in [what we test against](#what-we-test-against). They are close to even in the log of
  trials, so no stretch of trials counts for more than another.
- **No rule is the reference.** Every rule is compared with every other on the same draws. The
  first design measured rules against fixed 100, see [what was cut](scope.md#cut).
- **Principles 3 to 5 break ties, in order.** After that, ties go to the more pessimistic rule,
  then the simpler one, then the cheaper one. Pessimistic means it reads lower on noise and on
  few trials. Cheaper means less time and memory at 1,600 trials.

### How ties are called

- **Every rule scores the same draws.** Two rules are compared draw by draw, so the luck of the
  draw cancels.
- **Two rules tie when the interval on their difference includes zero.** The draws are resampled
  1,000 times, and the interval is the middle 95% of the difference.
- **The tails are reported but do not pick the rule.** The cloud's 95th percentile and the
  camps' 5th sit beside the means in every table. At 200 draws a 95th percentile rests on the
  top 10, which is why phase 2 uses 2,000.

### A worked example of the weighted error

Phase 2 scored a cloud and tight camps at 4 trials, with and without a floor of 100.

| Rule | k | Noise error | Camp error | At equal weights | Weighted error |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `8n`, no floor | 32 | 0.527 | 0.366 | 0.446 | 0.473 |
| `8n`, floor of 100 | 100 | 0.633 | 0.250 | 0.441 | 0.505 |

- **The floor reads camps 0.12 closer to 1, and noise 0.11 further from 0.** It has not made
  Cohesion more accurate. It has read everything higher.
- **At equal weights the two are close to a draw.** The floor is ahead by 0.005.
- **With a false pattern counted double, no floor wins.** Its weighted error is
  `(2 × 0.527 + 0.366) / 3 = 0.473`, against 0.505 with the floor.

## Elements of the rule

| Part | Values that run | Decided in |
| :-- | :-- | :-- |
| Growth | Fixed 100 against whatever the map's band allows | Phase 1 |
| Constant for linear in n | `8n`, with `4n` and `20n` parked. See [why these three](derivation.md#why-these-constants) | Phases 1 and 2 |
| Floor | None, 100 | Phase 2 |
| Join | Hard `max`, with added and smooth parked | Phase 2 |
| Ceiling | None, with 1,000 and the noise ceiling parked | Phase 3 |
| Trial count with weights | The effective count `1 / Σw²` | A test, not an experiment |

- **The trial count with weights is settled by one property.** A trial with almost no weight must
  not move k, as `tests/test_measures.py` already asks of Cohesion. The plain count fails that,
  and the effective count is the simplest that passes.

## What we test against

Phase 1 and phase 2 use the smallest grid that can decide the rule.

| Part | Phase 1 and 2 values | Why |
| :-- | :-- | :-- |
| Trials | 3, 4, 5, 6, 8, 10, 13, 16, 25, 50, 100, 200, 400 | At `8n` this covers k from 24 to 3,200. Phase 2 also runs every number from 3 to 16 |
| Shape | Cloud, 2 and 4 tight camps | Noise, camps and their order |
| Data | Exact probabilities over 4 options | No sampling noise, so k is the only thing moving |

Phase 2 then sets the floor against no floor on wider data, from 3 to 12 trials, where the two
give a different k.

| Part | Values for the floor | Why |
| :-- | :-- | :-- |
| Noise | Nine kinds. The cloud, spreads of 0.1, 0.3 and 3, 2 and 16 options, 30 and 100 observations, and 16 options with 30 observations | A floor chosen on one cloud may not hold on another |
| Camps | Eighteen kinds. 2 and 4 camps at five steps of looseness, at 30 and 100 observations, over 16 options, and over 16 options with 30 observations | Tight camps of exact probabilities are the easiest case |

Phase 3 takes the winner and fixed 100, then moves one part at a time at 10, 50 and 400 trials.

| Part | Phase 3 values | Why |
| :-- | :-- | :-- |
| Tightness | A sweep from exact camps to loose | Gives how loose camps can get before they read like a cloud |
| Cloud spread | A sweep from wide to narrow | Principle 1 rests on the cloud, so one recipe is not enough |
| Shape | One source, 8 camps | The cases the main grid leaves out |
| Camp sizes | 9 to 1, one stray trial, half camp and half cloud | Real camps are uneven |
| Observations per trial | 10, 30, 100, 1,000 | In an unsaved probe this moved Cohesion more than k did |
| Options | 2, 16, 64 | In an unsaved probe a cloud over 64 options read 0.4 |

### What each population should read

These are written before the run, so every population can fail.

| Population | Should read |
| :-- | :-- |
| Cloud, at any spread | Low. This is principle 1 |
| 2, 4 and 8 tight camps | High, with fewer camps above more camps |
| Camps as they loosen | Lower at each step, and never below the cloud |
| One source | The same at any number of trials. The level is not judged, since the [FAQ](../../docs/faq.md) says not to read Cohesion when incoherence is low |
| 9 to 1 | High. A small camp is still a camp |
| One stray trial | Closer to the same ensemble without the stray trial as trials grow |
| Half camp and half cloud | Between the cloud and the full camps |
