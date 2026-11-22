# Examples

Two worked examples. The first measures a simulation from start to finish. The second uses the
pairwise divergences to find out which trial belongs to which group.

Both run as written after this setup. The tests run them too, so the printed numbers are what
the package gives.

```python
import numpy as np
import incoherence as ic
```

## An urn that remembers

Put one red ball and one blue ball in an urn. Draw a ball, note its colour and put it back. Do
that 1,000 times and about half the draws are red, every time you run it.

Now change one rule. Each time you draw a ball, put it back with one more of the same colour. An
early run of red makes red more likely, which makes red more likely still. Each run settles on
its own mix of colours and stays there. This is called a Pólya urn, and it is a simple model of
a system where early luck gets locked in.

Each trial below is one run of 1,000 draws, recorded as 1 for red and 0 for blue. `bins=2` counts
the two colours.

```python
def urn(rng, draws, remember):
    """Draw from an urn that starts with one red ball and one blue ball."""
    red, blue = 1, 1
    seen = np.empty(draws)
    for i in range(draws):
        seen[i] = rng.random() < red / (red + blue)
        if remember:
            red += seen[i]
            blue += 1 - seen[i]
    return seen


rng = np.random.default_rng(4)
plain = [urn(rng, 1000, remember=False) for _ in range(20)]
polya = [urn(rng, 1000, remember=True) for _ in range(20)]

for name, trials in [("plain", plain), ("polya", polya)]:
    result = ic.measure(trials, bins=2)
    red = np.mean([trial.mean() for trial in trials])
    print(f"{name}  red={red:.2f}  incoherence={result.incoherence:.2f}", end="  ")
    print(f"cohesion={result.cohesion:.2f}")
```

```text
plain  red=0.50  incoherence=0.00  cohesion=1.00
polya  red=0.49  incoherence=0.28  cohesion=0.37
```

Averaged over the runs, both urns draw red about half the time. The average says they are the
same system. Incoherence says they are not. Every run of the plain urn looks like every other
run. The runs of the Pólya urn disagree, and the average of 0.49 describes almost none of them.

The low cohesion adds that the runs do not fall into camps. Each one lands on its own mix.

```python
shares = np.sort([trial.mean() for trial in polya])
print("share of red in each run, lowest to highest")
print(" ".join(f"{share:.2f}" for share in shares[:10]))
print(" ".join(f"{share:.2f}" for share in shares[10:]))
```

```text
share of red in each run, lowest to highest
0.04 0.16 0.23 0.26 0.29 0.33 0.33 0.37 0.43 0.48
0.56 0.58 0.59 0.61 0.62 0.65 0.73 0.81 0.84 0.95
```

## Finding the camps

Cohesion tells you that the trials fall into groups. It does not tell you which trial is in
which group. The divergences behind it do.

`result.divergences` holds one value for every pair of trials. It is 0 when two trials have the
same distribution and grows as they differ. `scipy.spatial.distance.squareform` lays the values
out as a table with one row and one column per trial.

```python
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

rng = np.random.default_rng(1)
camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
result = ic.measure(camps)

print(f"cohesion={result.cohesion:.2f}")
for row in squareform(result.divergences)[:4, :4]:
    print(" ".join(f"{value:.2f}" for value in row))
```

```text
cohesion=0.82
0.00 0.53 0.00 0.54
0.53 0.00 0.51 0.00
0.00 0.51 0.00 0.53
0.54 0.00 0.53 0.00
```

The table shows the first four trials. Trials 1 and 3 are 0.00 apart, so they behave alike.
Trials 1 and 2 are 0.53 apart, so they do not.

To sort every trial into a group, join the two closest trials, then the next closest, and carry
on until the number of groups you asked for is left. This is called hierarchical clustering, and
`scipy` does it from the divergences as they are.

```python
groups = fcluster(linkage(result.divergences, "average"), t=2, criterion="maxclust")
print(groups)
```

```text
[1 2 1 2 1 2 1 2 1 2]
```

The odd trials were drawn around −3 and the even ones around +3, and the groups match.

Check cohesion before you trust the groups. Asking for two groups always returns two, even when
the trials are scattered and no real groups exist. A cohesion of 0.82 says the groups here are
real.
