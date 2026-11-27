# Choosing Cohesion's k

Why we tested Cohesion's `k`, and what we found.

- **k is Cohesion's resolution.** Cohesion's last step measures how bunched the pairwise
  divergences are. Each divergence is spread into a narrow bump, and `k` sets how narrow it is.
- **Today k is fixed at 100.** It is `COHESION_K` in `src/incoherence/_measures.py`, the value the
  Cohesion paper uses.
- **The right k does depend on the number of trials.** With n trials there are n(n − 1) / 2
  pairs, so more trials give more divergences to resolve. The proposal was `k = 8n`, see
  [where it comes from](derivation.md).
- **Plain `8n` wins.** Rules are judged on a [weighted error](design.md#how-we-judge-a-rule),
  where reading noise as a pattern counts double. `8n` has the lowest, fixed 100 the highest,
  and a floor of 100 makes it worse. See the [results](results/README.md).
- **The package default has not changed yet.**

## Pages

| Page | What's in it |
| :-- | :-- |
| [Results](results/README.md) | What each phase found and decided, with the stored tables |
| [Design](design.md) | The four phases, how a rule is judged, the parts of a rule and the data each phase runs on |
| [Derivation](derivation.md) | Where `8n` and the noise ceiling come from, and the formulas phase 1 checks |
| [Scope](scope.md) | Most of the first design is cut or parked. What went, what came back, what is held fixed, and open questions |
| [Running](running.md) | How to run it, how the code is built, where results are stored and how CI checks them |
| [Glossary](glossary.md) | The one word we use for each idea |

## Results

The experiment set out to give these five things. The [results](results/README.md) hold the
detail.

| Outcome | Phase | Result |
| :-- | :-- | :-- |
| **A rule for k.** Its form, its constant, and whether it has a floor or a ceiling | 1 and 2 | Plain `8n`, with no floor and no ceiling. Its weighted error over the grid is 0.300, against 0.327 for fixed 100. It wins at any weight above 1.72, and `20n` wins at equal weights |
| **A minimum number of trials.** The smallest from which tight camps outscore the cloud in 95% of draws, and keep doing so at every larger number. The docs state the worse of 2 and 4 camps | 2 | 11 trials. 2 tight camps need 6 and 4 tight camps need 11. Under fixed 100 it is 10 |
| **A check of the formulas.** They are the only derivation the rule has, so they are checked against measured values | 1 | They hold where their assumptions hold. Real pairs share trials, so at `8n` they carry 1.2 to 8.6 times the noise the formula gives |
| **What no rule for k fixes.** Each item has a number, such as how loose camps can get before they read like a cloud, and says whether a shuffled baseline would fix it | 3 | [Twelve limits](results/README.md#phase-3-what-no-rule-for-k-fixes). Loose camps are the largest. Ten observations a trial is the one `8n` makes worse |
| **A confirmation on data not used to choose.** Fresh seeds and real values. The figures in [improvements](../../docs/improvements.md) must also keep the order the papers give them | 4 | Fresh seeds give the same order of rules, and real values agree from 13 trials. The papers' orderings hold, but [Figure 2 moves by 0.22](results/README.md#what-changing-the-default-would-do) |
