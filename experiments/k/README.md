# Choosing Cohesion's k

Why we are testing Cohesion's `k`, and what we found. Nothing has run yet.

- **k is Cohesion's resolution.** Cohesion's last step measures how bunched the pairwise
  divergences are. Each divergence is spread into a narrow bump, and `k` sets how narrow it is.
- **Today k is fixed at 100.** It is `COHESION_K` in `src/incoherence/_measures.py`, the value the
  Cohesion paper uses.
- **The right k should depend on the number of trials.** With n trials there are n(n − 1) / 2
  pairs, so more trials give more divergences to resolve. The proposal is `k = 8n`, see
  [where it comes from](derivation.md).
- **Fixed 100 stays unless a rule beats it.** A rule must pass the noise bar and then win by
  more than a tie, see [how we judge a rule](design.md#how-we-judge-a-rule).

## Pages

| Page | What's in it |
| :-- | :-- |
| [Design](design.md) | The four phases, how a rule is judged, the parts of a rule and the data each phase runs on |
| [Derivation](derivation.md) | Where `8n` and the noise ceiling come from, and the formulas phase 1 checks |
| [Scope](scope.md) | Most of the first design is cut or parked. What went, what comes back and when, what is held fixed, and open questions |
| [Running](running.md) | How the code is built, where results are stored and how CI checks them |
| [Glossary](glossary.md) | The one word we use for each idea |

## Results

The experiment is done when it gives these five things. Each row fills in as its phase runs.

| Outcome | Phase | Result |
| :-- | :-- | :-- |
| **A rule for k.** Its form, its constant, and whether it has a floor or a ceiling | 1 and 2 | Not run |
| **A minimum number of trials.** The smallest from which tight camps outscore the cloud in 95% of draws, and keep doing so at every larger number. The docs state the worse of 2 and 4 camps | 2 | Not run |
| **A check of the formulas.** They are the only derivation the rule has, so they are checked against measured values | 1 | Not run |
| **What no rule for k fixes.** Each item has a number, such as how loose camps can get before they read like a cloud, and says whether a shuffled baseline would fix it | 3 | Not run |
| **A confirmation on data not used to choose.** Fresh seeds and real values. The figures in [improvements](../../docs/improvements.md) must also keep the order the papers give them | 4 | Not run |
