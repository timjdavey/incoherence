# FAQ

Every example runs as written after this setup.

```python
import numpy as np
import incoherence as ic
```

## What counts as high?

There is no magic number. Incoherence is a relative measure, so it is up to you to find the levels of incoherence and cohesion that matter in your field or use case.

"p-value < 0.05" is a mythical threshold that gives people a false sense of security and so we don't want to fall for that same trap.

## How many observations does each trial need?

There is no minimum, but fewer observations means a higher incoherence. Incoherence is actively designed to assume your trials disagree until the data shows otherwise.

Fewer observations means more noise, so trials from the same source look more different and incoherent.

```python
for size in [10, 30, 100, 500, 5000]:
    scores = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        scores.append(ic.incoherence([rng.normal(0, 1, size) for _ in range(10)]))
    print(f"{size:5} observations  mean={np.mean(scores):.2f}  highest={np.max(scores):.2f}")
```

```text
   10 observations  mean=0.14  highest=0.20
   30 observations  mean=0.07  highest=0.13
  100 observations  mean=0.05  highest=0.08
  500 observations  mean=0.02  highest=0.03
 5000 observations  mean=0.01  highest=0.01
```

## What would my data score if the trials agreed?

You can calculate what we call the `result.floor()`.

It pools every observation, then returns them randomly into trials of the same sizes. The shuffled trials share one source, so any incoherence left comes only from sample size.

This method is called a permutation test.

```python
rng = np.random.default_rng(1)
same = [rng.normal(0, 1, 30) for _ in range(10)]
camps = [rng.normal(3 if i % 2 else -3, 1, 30) for i in range(10)]

for name, trials in [("same", same), ("camps", camps)]:
    result = ic.measure(trials)
    print(f"{name:6} incoherence={result.incoherence:.2f}  floor={result.floor(rng=rng):.2f}")
```

```text
same   incoherence=0.10  floor=0.09
camps  incoherence=0.55  floor=0.04
```

`same` sits on its floor, so nothing says its trials disagree. `camps` is far above its floor, so the disagreement is real.

The floor is the mean of 20 shuffles. Each shuffle is measured with the same bins, `bounds`, `k` and weights as the result. The shuffles are random, so here the floor moves by about 0.01 from one call to the next. Pass a seed as `rng` to fix it, or raise `repeats` to steady it.

A table of probabilities has no floor. Probabilities do not say how many observations each trial holds, and the floor depends on that. Pass counts to `measure_counts` if you want one.

## How many trials do you need?

Incoherence is designed to be a fact about your system, not your data. Copying every trial leaves it unchanged. But a small ensemble is a small sample of your system, so it can read lower and varies more. In the table below, the mean rises from 0.29 with 2 trials to 0.50 with 50.

Practically speaking use at least ten for incoherence and eleven for cohesion.

Few trials push cohesion up, because it looks at pairs of trials. Two trials make one pair, five make 10 and ten make 45. A handful of pairs always looks bunched, so cohesion reads high whatever the data.

In the example every trial wanders off on its own, so cohesion should be low. Each row is twenty ensembles, showing the mean with the lowest and highest in brackets.

```python
for count in [2, 3, 5, 10, 50]:
    scores = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        trials = [rng.normal(rng.uniform(-4, 4), rng.uniform(0.3, 2), 500) for _ in range(count)]
        result = ic.measure(trials)
        scores.append([result.incoherence, result.cohesion])
    mean, low, high = np.mean(scores, 0), np.min(scores, 0), np.max(scores, 0)
    print(f"{count:2} trials", end="")
    for i, name in enumerate(["incoherence", "cohesion"]):
        print(f"  {name} {mean[i]:.2f} ({low[i]:.2f} to {high[i]:.2f})", end="")
    print()
```

```text
 2 trials  incoherence 0.29 (0.03 to 0.60)  cohesion 0.99 (0.98 to 1.00)
 3 trials  incoherence 0.38 (0.11 to 0.60)  cohesion 0.52 (0.33 to 0.95)
 5 trials  incoherence 0.45 (0.09 to 0.68)  cohesion 0.37 (0.22 to 0.92)
10 trials  incoherence 0.47 (0.16 to 0.63)  cohesion 0.28 (0.13 to 0.73)
50 trials  incoherence 0.50 (0.47 to 0.54)  cohesion 0.22 (0.17 to 0.26)
```

Two trials score a cohesion of nearly 1 whatever the data. Even ten is not many. The mean is near where it settles, but one ensemble of ten scored 0.73 and another scored 0.13. Incoherence is just as jumpy, with two trials scoring anywhere from 0.03 to 0.60.

## What does a cohesion of 1 mean?

Cohesion only makes sense when Incoherence is high. Do not read it on it's own.

Cohesion is 1 when every pair of trials is the same distance apart. That is true when the trials all agree (incoherence low), and when they all disagree by the same amount (incoherence high).

## Should I bin my data?

Only if it is already whole numbers or categories. Cut real-valued measurements into bins and the answer depends on how many bins you picked.

| Your data | Use |
| :-- | :-- |
| Real-valued measurements | `measure(trials)` |
| Whole numbers, or categories coded as numbers | `measure(trials, bins=edges)` with one bin per value |
| Counts or probabilities you already hold | `measure_counts(counts)` |

Give one more edge than you have values, because the last bin includes its upper edge. For the values 0, 1 and 2, use `[0, 1, 2, 3]`. With `[0, 1, 2]` the 1s and 2s share a bin, so trials that disagree look identical.

```python
ones_and_twos = [[1, 1, 1], [2, 2, 2]]

print(f"[0, 1, 2]     incoherence={ic.incoherence(ones_and_twos, bins=[0, 1, 2]):.2f}")
print(f"[0, 1, 2, 3]  incoherence={ic.incoherence(ones_and_twos, bins=[0, 1, 2, 3]):.2f}")
```

```text
[0, 1, 2]     incoherence=0.00
[0, 1, 2, 3]  incoherence=0.79
```

The two camps below are the same data in every row, and only the number of bins changes.

```python
rng = np.random.default_rng(1)
camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]

print(f"unbinned   incoherence={ic.incoherence(camps):.2f}")
for bins in [5, 10, 50, 200]:
    print(f"{bins:3} bins   incoherence={ic.incoherence(camps, bins=bins):.2f}")
```

```text
unbinned   incoherence=0.57
  5 bins   incoherence=0.46
 10 bins   incoherence=0.33
 50 bins   incoherence=0.19
200 bins   incoherence=0.15
```

The score falls because incoherence divides by the largest entropy a trial could have, which grows with every bin you add. The gap between each trial and the pool stays at about one bit, however fine the bins. If you must bin real values, fix the edges once and use them for every ensemble you compare.

## How do I compare two ensembles?

Set `bounds` to the same range for both, using the range the quantity could take. By default each ensemble is scaled by its own lowest and highest observation, which makes the score blind to size. Two camps 6 units apart then score the same as two camps 0.6 apart.

```python
rng = np.random.default_rng(1)
wide = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
narrow = [rng.normal(0.3 if i % 2 else -0.3, 0.1, 500) for i in range(10)]

for name, trials in [("wide", wide), ("narrow", narrow)]:
    own = ic.incoherence(trials)
    shared = ic.incoherence(trials, bounds=(-8, 8))
    print(f"{name:7} own scale={own:.2f}  shared scale={shared:.2f}")
```

```text
wide    own scale=0.57  shared scale=0.59
narrow  own scale=0.57  shared scale=0.18
```

On a shared scale the narrow split counts for less, because it is small next to what was possible.

## How should I set `bounds`?

Set them to the range the quantity could take, and no wider. The entropy estimate is made on a scale that runs from the low bound to the high one, so the bounds decide how much of that scale your data fills. Widen them and the same trials look closer together, so the score falls. Here is the `wide` ensemble from the last answer inside four sets of bounds.

```python
for limit in [8, 20, 50, 100]:
    score = ic.incoherence(wide, bounds=(-limit, limit))
    print(f"bounds=({-limit}, {limit})".ljust(20) + f"incoherence={score:.2f}")
```

```text
bounds=(-8, 8)      incoherence=0.59
bounds=(-20, 20)    incoherence=0.49
bounds=(-50, 50)    incoherence=0.27
bounds=(-100, 100)  incoherence=0.15
```

Leaving `bounds` unset does not avoid the choice. The default uses the lowest and highest observation, so one stray value sets the scale for you. It stretches the scale to fit, which squeezes everything else together. Below, one observation in `wide` is changed to 60.

```python
stray = [trial.copy() for trial in wide]
stray[0][0] = 60

print(f"as drawn          incoherence={ic.incoherence(wide):.2f}")
print(f"one stray value   incoherence={ic.incoherence(stray):.2f}")
```

```text
as drawn          incoherence=0.57
one stray value   incoherence=0.37
```

That is one observation in 5,000. Set `bounds` and the package rejects the stray value with an error instead.

No bounds are the correct ones. The score is always relative to the range you chose, so choose it from what the quantity could do and keep it fixed. Then find the level of incoherence that matters for your use case, as in [What counts as high?](#what-counts-as-high).

## Should I change `k`?

No, unless you are testing the estimator itself. `k` sets how sharply the entropy estimate picks out detail in real-valued observations, and it moves the score a great deal. Here is the `wide` ensemble again at four values of `k`.

```python
for k in [5, 10, 20, 100]:
    print(f"k={k:<3}  incoherence={ic.incoherence(wide, k=k):.2f}")
```

```text
k=5    incoherence=0.67
k=10   incoherence=0.57
k=20   incoherence=0.37
k=100  incoherence=0.08
```

A higher `k` only sees bunching tighter than about `1/k` of the range. Broad shapes then read as spread out, so every score falls and sits closer to its floor. It is also slower, by about 1.5 times at `k=100` and 7 times at `k=1000`, which is the limit.

No value of `k` is the correct one. The default of 10 is what the published figures used, so keep it. Never compare results made with different values.

## Should I change `cohesion_k`?

Only to match the papers. `cohesion_k` is a separate setting from `k`. It sets how finely cohesion's last step reads the pairwise divergences. By default it is 8 for each trial, so 10 trials give 80. The papers fixed it at 100, so pass `cohesion_k=100` to get their numbers back.

More trials give more pairs, and more pairs can show finer detail. A fixed value reads the same camps lower as the ensemble grows. Here are two camps of trials at both settings.

```python
rng = np.random.default_rng(1)
for count in [10, 100]:
    two = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(count)]
    for cohesion_k in [None, 100]:
        result = ic.measure(two, cohesion_k=cohesion_k)
        k, score = result.cohesion_k, result.cohesion
        print(f"{count:3} trials  cohesion_k={k:<4g}  cohesion={score:.2f}")
```

```text
 10 trials  cohesion_k=80    cohesion=0.78
 10 trials  cohesion_k=100   cohesion=0.82
100 trials  cohesion_k=800   cohesion=0.94
100 trials  cohesion_k=100   cohesion=0.83
```

- **The default tells camps from noise better.** Over 3 to 400 trials it reads noise lower than 100 does on average, and camps higher. The [k experiments](../experiments/k/README.md) tested seven rules and this one won.
- **`result.cohesion_k` says which value was used.** With unequal `weights` it counts trials by weight, so a trial with almost no weight barely moves it. The experiments did not test weights.
- **The default struggles in three places.** With about ten observations a trial, many trials read noise as camps. With many options and few trials, camps are harder to tell from noise. Trials that are each nearly certain of one option read slightly higher from about 13 trials. In all three, `cohesion_k=100` reads them better, see [the limits](../experiments/k/results/README.md#what-no-rule-for-k-fixes).
- **Never compare results made with different values,** just as for `k`.

## What should I check before I trust a number?

1. Compare it with `result.floor()`, the floor for your sample size.
2. Read cohesion only when you have many trials. Eleven is the bare minimum.
3. Never read cohesion without incoherence next to it.
4. Keep the route, the bins, `bounds`, `k` and `cohesion_k` the same across everything you compare.
