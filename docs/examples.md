# Examples

The code runs as written. The tests run it too, so the printed numbers are what the package gives.

```python
import incoherence as ic
```

## The range

```python
ordered = [
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
]
disordered = [
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
]
split = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

for name, trials in [("ordered", ordered), ("disordered", disordered), ("split", split)]:
    result = ic.measure(trials, bins=2)
    print(f"{name:10} incoherence={result.incoherence:.2f}")
```

```text
ordered    incoherence=0.00
disordered incoherence=0.02
split      incoherence=0.53
```

`bins=2` tells `measure` to count the observations into two bins, one for the 0s and one for
the 1s.

The ordered trials are identical, so they score 0. The disordered trials are a jumble, yet they
score almost the same. Incoherence ignores the order inside a trial. It only asks whether the
trials agree with each other, and the disordered ones do. Each is about half 1s.

The split trials disagree. Three are nearly all 0s and three are nearly all 1s. Pool them and
you get 30 ones in 60 observations. That even mix describes none of the six.

## What is behind the number

Incoherence is built from entropy. The entropy of a trial says how hard it is to guess the next
observation. A trial of all 0s has an entropy of 0, because you always guess right. A trial that
is half 1s and half 0s has an entropy of 1, the highest two outcomes can give. This is called
Shannon entropy, and here it is measured in bits.

```python
result = ic.measure(ordered, bins=2)
print(result.entropies.round(2))
print(round(result.pooled_entropy, 2))
```

```text
[0.88 0.88 0.88 0.88 0.88 0.88]
0.88
```

`entropies` holds one value per trial. Each ordered trial has three 1s and seven 0s, which gives
`−0.3 × log2(0.3) − 0.7 × log2(0.7) = 0.88`. `pooled_entropy` treats all 60 observations as one
long trial. It is also 0.88, so pooling has changed nothing.

```python
result = ic.measure(split, bins=2)
print(result.entropies.round(2))
print(round(result.pooled_entropy, 2))
```

```text
[0.47 0.47 0.47 0.47 0.47 0.47]
1.0
```

Each split trial is easy to guess, so its entropy is low. The pool is an even mix, so its
entropy is 1. That gap between the pool and the trials is what incoherence measures.

## Total incoherence

Incoherence reaches 1 when the entropy of each trial is low, yet together they represent maximum entropy.

The table below has three trials and three outcomes. This time a row is not a list of
observations. Each column counts how often the trial landed on one outcome. The first trial only
ever lands on the first outcome, the second on the second and the third on the third.
`measure_counts` takes a table like this.

```python
result = ic.measure_counts([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
print(f"incoherence={result.incoherence:.2f}")
```

```text
incoherence=1.00
```

Every trial has an entropy of 0, because it never varies. The pool is an even mix of all three
outcomes, which is the highest entropy three outcomes can give. The gap cannot get any bigger.

## Cohesion

Incoherence says how much the trials disagree. Cohesion says whether that disagreement falls
into a few camps or is different every time.

The split trials fall into two camps. In the ensemble below every trial has its own share of 1s.

```python
scattered = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

for name, trials in [("split", split), ("scattered", scattered)]:
    result = ic.measure(trials, bins=2)
    print(f"{name:10} incoherence={result.incoherence:.2f}  cohesion={result.cohesion:.2f}")
```

```text
split      incoherence=0.53  cohesion=0.84
scattered  incoherence=0.60  cohesion=0.35
```

Both ensembles are incoherent by about the same amount. Cohesion tells them apart. A split trial
is one of two kinds, so once you know its camp you know what it looks like. The scattered trials
have no camps to fall into.

Read cohesion next to incoherence, never alone. Six trials is also few for cohesion, and real
data needs more. The [FAQ](faq.md) explains both points, along with what counts as high, whether
to bin real-valued data and when to set the options.
