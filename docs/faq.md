# FAQ

Every example runs as written after this setup. The tests run them too, so the printed numbers
are what the package gives.

```python
import numpy as np
import incoherence as ic
```

## What counts as high?

Incoherence is a relative measure.

"p-value < 0.05" is a mythical threshold which gives people a false sense of security and can be misleading. We want to actively avoid that trap.

You need to find the incoherence and cohesion levels that are correct for your field and use case.


## How many observations does each trial need?

The incoherence measure is designed to assume incoherence, rather than consistency. So the fewer the observations, naturally the greater the differences in entropy, so the higher the incoherence.

For example:

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

Shuffle your data and measure again. Pool every observation, then deal them back out at random into trials of the same sizes. The shuffled trials share one source, so whatever incoherence is left comes from sample size alone. We call that the floor, and the method is called a permutation test.

For example:

```python
def floor(trials, rng, repeats=20):
    """Incoherence when the same observations are dealt out at random."""
    pooled = np.concatenate(trials)
    cuts = np.cumsum([len(trial) for trial in trials])[:-1]
    shuffles = [np.split(rng.permutation(pooled), cuts) for _ in range(repeats)]
    return float(np.mean([ic.incoherence(shuffle) for shuffle in shuffles]))


rng = np.random.default_rng(1)
same = [rng.normal(0, 1, 30) for _ in range(10)]
camps = [rng.normal(3 if i % 2 else -3, 1, 30) for i in range(10)]

for name, trials in [("same", same), ("camps", camps)]:
    print(f"{name:6} incoherence={ic.incoherence(trials):.2f}  floor={floor(trials, rng):.2f}")
```

```text
same   incoherence=0.10  floor=0.09
camps  incoherence=0.55  floor=0.04
```

The first ensemble sits on its floor, so its trials may not disagree at all. The second is far above its floor, so the disagreement is real.

## How many trials do you need?

The same is true as the observations (it assumes incoherence). But practically speaking aim for at least ten, and more for cohesion.

Cohesion looks at pairs of trials, and few trials make few pairs. Two trials make one pair, five make 10 and ten make 45. A handful of values always looks bunched, so cohesion reads high when there are few trials, whatever the data.

For example, here every trial wanders off on its own, which should score a low cohesion. Each row is twenty ensembles, showing the mean with the lowest and highest in brackets.

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
 2 trials  incoherence 0.29 (0.03 to 0.60)  cohesion 1.00 (0.99 to 1.00)
 3 trials  incoherence 0.38 (0.11 to 0.60)  cohesion 0.74 (0.70 to 0.92)
 5 trials  incoherence 0.45 (0.09 to 0.68)  cohesion 0.48 (0.34 to 0.89)
10 trials  incoherence 0.47 (0.16 to 0.63)  cohesion 0.29 (0.14 to 0.73)
50 trials  incoherence 0.50 (0.47 to 0.54)  cohesion 0.22 (0.17 to 0.26)
```

- **Two trials always score a cohesion near 1.** One pair gives a single value, and a single value is always bunched.
- **Ten trials are not many.** The mean cohesion is close to where it settles, but one ensemble of ten scored 0.73 and another scored 0.14.
- **Incoherence needs trials too.** It has no pull towards 1, but a few trials are a small sample of what the system can do. Two trials scored anywhere from 0.03 to 0.60.

## What does a cohesion of 1 mean?

It means every pair of trials is the same distance apart. That happens when the trials all agree, and also when they all disagree by the same amount.

Cohesion measures how bunched the pairwise divergences are. If every trial is certain of a different outcome, every pair is equally far apart. Cohesion reads that as one tight group and scores it 1.

Incoherence tells the two apart. With incoherence near 0 the trials all agree. With incoherence near 1 they share nothing. So never read cohesion without incoherence next to it.

## Should I bin my data?

Only if it is already whole numbers or categories. Cutting real-valued measurements into bins makes the answer depend on how many bins you pick.

The three routes into the package suit three kinds of data.

| Your data | Use |
| :-- | :-- |
| Real-valued measurements | `measure(trials)` |
| Whole numbers, or categories coded as numbers | `measure(trials, bins=edges)` with one bin per value |
| Counts or probabilities you already hold | `measure_counts(counts)` |

For example, the two camps below are just as far apart in every row:

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

Incoherence falls because of how it is scaled. A trial from one camp is about one bit short of the pooled entropy, however fine the bins. The largest entropy a trial could have keeps growing as bins are added, and incoherence divides by it.

If you must bin real values, fix the edges once and use the same edges for every ensemble you compare.

## How do I compare two ensembles?

Set `bounds` to the same range for both. Use the range the quantity could take.

By default each ensemble is scaled by its own lowest and highest observation. That makes the result blind to size. Two camps 6 units apart and two camps 0.6 units apart score the same.

For example:

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

## What does one outlier do?

It can drag the score down a long way, unless you set `bounds`. The default scale stretches to fit the stray value, which squeezes everything else together.

For example, here a single observation of 60 is added to the first trial of the wide ensemble:

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

That is one observation in 5,000. Fixed `bounds` would have rejected it with an error instead.

## Should I change `k`?

No, unless you are testing the estimator itself.

`k` sets how sharply the entropy estimate for real-valued observations picks out detail, and it changes the result a great deal.

For example:

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

A higher `k` only sees bunching tighter than about `1/k` of the range. Broad shapes then read as spread out, so every score falls and sits closer to its floor. A higher `k` also needs a finer grid, so it is slower, about 4 times at `k=100` and 40 times at `k=1000`. The limit is 1,000.

No value of `k` is the correct one. The default of 10 is the one the published figures used, so keep it. Never compare results made with different values.

## What should I check before I trust a number?

1. Check it against the floor for your sample size.
2. Read cohesion only when you have many trials. Ten is a bare minimum.
3. Read incoherence and cohesion together.
4. Keep the route, the bins, the `bounds` and `k` the same across everything you compare.
