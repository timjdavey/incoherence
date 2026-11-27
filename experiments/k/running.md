# Running

How the experiment code is built, where its results go and how CI checks them.

- **Numpy, scipy and the package only.** No plots.
- **Every experiment passes k explicitly,** so results don't depend on the package default.
  `_measures.cohesion()` reads `COHESION_K` and `COHESION_POINTS` itself, so the experiments
  call `densities` and `density_entropy` directly, with `points_for(k)`.
- **Divergences are worked out once per draw.** Every k and every rule then reads the same
  divergences, so the map costs little more than one rule.
- **Each phase stores its results before the next one is written.** They are text tables under
  `experiments/k/results/`, with what the phase decided written beside them.
- **A `--check` run reruns them** and fails when any value moves by more than one in the last
  printed place. CI runs it on the locked versions only.
- **The folder sits outside `src/`,** so it is not in the wheel.
