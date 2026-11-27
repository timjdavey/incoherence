# Scope

What the experiment leaves out, what comes back only if a result calls for it, and what stays
fixed.

## Cut

| Cut | Why |
| :-- | :-- |
| Linear in pairs | It has no derivation, and it reaches k = 1.8 million at 1,600 trials. The map shows if the best k grows that fast |
| `n^0.4` and `29√n` as runs | The map's fitted power answers both. A power near 0.4 or 0.5 brings that one back |
| A floor of 50 | Nothing argues for 50. Phase 2 tests none against 100 |
| The main grid above 400 trials | 400 trials is 79,800 pairs, and k = 3,200 at `8n`. One run at 1,600 stays in phase 4 |
| 8 camps in the main grid | 2 and 4 decide the order. 8 is a limit in phase 3 |
| Data kind as its own part | Counts are the observations row. Real values are phase 4 |
| Weights as an experiment, and the effective pairs count | One property decides the trial count, see [elements of the rule](design.md#elements-of-the-rule). Effective pairs also passes it, but it is the less simple of the two |
| 2,000 draws outside phase 2 | Tails only decide the floor and the minimum |

## Parked

| Parked | Runs only if | Note |
| :-- | :-- | :-- |
| `4n` and `20n` | The band in phase 1 sits clearly below or above `8n` | |
| The added join `100 + 8n` and the smooth join `√(100² + (8n)²)` | The floor wins phase 2, and the mean steps at 12.5 trials by more than a tie | A hard `max(100, 8n)` bends at 12.5 trials, close to the 10 the FAQ gives as the minimum |
| A ceiling of 1,000 | The map shows the best k levelling off, or the run at 1,600 trials is too slow | It may be about cost, not accuracy. `MAX_K = 1000` in `src/incoherence/_result.py` already caps the observation k for that reason |
| The [noise ceiling](derivation.md#the-noise-ceiling) | In phase 3, tight camps with finite observations read lower as trials grow | An unsaved probe found a ceiling only mattered for exact camps |
| Cloud spread in the pass bar | Phase 3 finds a spread where the winner reads above fixed 100 | |

## Held fixed

- **The definition of the measure.** The power of 0.5, the squaring and the divergence stay as
  the papers have them.
- **The density grid follows k and is not searched.** `points_for(k)` in
  `src/incoherence/_entropy.py` gives `20k + 1` points, so 64,001 at k = 3,200.
- **The grid is checked for convergence at every k a rule reaches.** Convergence at k = 100 says
  nothing about k = 3,200, or about k = 12,800 in the run at 1,600 trials.
- **The k for real-valued observations stays at 10.** It is `DEFAULT_K` in
  `src/incoherence/_result.py`, and it shapes the divergences that Cohesion's k then reads.
- **A shuffled baseline.** Comparing Cohesion to shuffled data, as `floor()` does for incoherence,
  is a separate feature.

## What gets in

An experiment goes in if it meets one of the first three and always meets the fourth.

- **It could change the rule.**
- **It sets a limit we would write in the docs.**
- **It tests a part a user actually varies,** such as trials, observations or options.
- **It is seeded and fast enough for CI.**

## Open questions

- **A shuffled baseline would change the order of the principles.** It reads a score against
  noise of the same size, so it would take over principles 1 and 4. The rule for k would then
  only need to keep camps and noise apart, and principle 3 would go first.
- **The noise ceiling can't be worked out on every data kind.** Probabilities have no count of
  observations, and real values have no count of options. If its trigger fires it needs a
  fallback for both.
- **The time budget is not set.** Nobody has timed the map at 400 trials or the run at 1,600.
