# Derivation

Where `8n` and the noise ceiling come from, and the formulas phase 1 checks against measured
values.

## The formulas to check

| Formula | Setup | Predicts | True value |
| :-- | :-- | :-- | :-- |
| `8(m − 1) / (k − 8)` | m exactly separate divergence values | The entropy read | 0 |
| `1 − k / (16N)` | N random divergence values | The entropy read | 1 |
| `(s − 1) / (4m ln 2)` | Two trials from one source, m observations over s options | Their mean divergence | 0 |
| `√(2(s − 1)) / (4m ln 2)` | The same two trials | The spread of that divergence | 0 |

- **The check covers what the formulas leave out.** That means values that hold unequal shares of
  the pairs, as uneven camps give, and real pairwise divergences, where pairs share trials.
- **Noise between two camps is measured too.** It is expected to shrink as 1 / √m, not 1 / m, so
  it is wider than noise within one source. The check measures it before the noise ceiling runs.

## Where 8n comes from

- **The first two formulas give `8n`.** For two separate values the errors are equal when
  `k(k − 8) = 128N`. With N = n(n − 1) / 2 pairs that is exactly `k = 8n`.
- **The constant assumes two camps.** Two exact camps give two separate values. Four give seven,
  and the balance moves to about `20n`. So the derivation fixes the form, and the constant is a
  guess at how many camps there are.
- **A weight on the noise error lowers the balance.** With the noise error counted w times, the
  balance moves to about `8n × √((m − 1) / w)` for m separate values. At the
  [weight of 2](design.md#how-we-judge-a-rule) that is about `6n` for two camps and `14n` for
  four.
- **So `8n` is the lowest equal balance.** Two camps are the fewest that make a pattern, and
  every pattern with more camps balances at a higher k.

### Why these constants

- **`8n` is the balance for two camps** and `20n` is the balance for four.
- **`4n` is the pessimistic half** of `8n`.

## The noise ceiling

- **Sampling noise sets how fine k can usefully go.** Two trials from one source sit apart by the
  mean in the third formula, with the spread in the fourth. A bump narrower than that spread
  resolves noise.
- **So the ceiling is `2m / √(s − 1)`.** That is the k whose bump is as wide as the spread. At 16
  options and 100 observations it is 52, which `8n` passes at 7 trials.
- **It is a ceiling and not a growth.** As a growth it would ignore the number of trials, so
  three trials of 1,000 observations would get a k over 1,000 for three pairs.

The measured columns are from the [formula check](results/laws.txt), 2,000 seeded pairs over a
uniform source.

| Options | Observations | Mean | Mean formula | Spread | Spread formula | Ceiling |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 4 | 30 | 0.038 | 0.036 | 0.030 | 0.029 | 35 |
| 4 | 100 | 0.011 | 0.011 | 0.009 | 0.009 | 115 |
| 16 | 100 | 0.057 | 0.054 | 0.021 | 0.020 | 52 |
| 64 | 1,000 | 0.023 | 0.023 | 0.004 | 0.004 | 252 |

## What the runs found

- **The form holds.** The best k grows as trials to the power 1.03, see
  [phase 1](results/README.md#phase-1-the-best-k-grows-with-trials).
- **The best k sits between the two weighted balances.** It is at 9n to 16n from 4 trials, and
  `8n` stays within 0.009 of its weighted error.
- **The ceiling is not used.** It helps at 10 observations a trial and costs at 30 and 100, see
  [the noise ceiling](results/README.md#the-noise-ceiling).
