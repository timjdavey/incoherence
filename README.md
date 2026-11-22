# incoherence

Run a complex system twice from the same start and it can end up somewhere quite different. When
that happens, the average over many runs may describe none of them. This package measures how
far that has gone.

**Incoherence** says how much repeated trials of a system disagree with each other. **Cohesion**
says whether that disagreement falls into a few shared patterns or is different every time. Both
run from 0 to 1.

The measures come from two papers. The [primers](https://timjdavey.com/incoherence/) explain the
ideas without equations and are the best place to start.

- Incoherence: [primer](https://timjdavey.com/incoherence/) and [paper](https://doi.org/10.3390/e26080683)
- Cohesion: [primer](https://timjdavey.com/cohesion/) and [paper](https://doi.org/10.3390/e25121605)

## Install

The package is not on PyPI yet, so install it from GitHub. It needs Python 3.12 or later.

```sh
uv add git+https://github.com/timjdavey/incoherence
# or
pip install git+https://github.com/timjdavey/incoherence
```

## A quick example

A trial is one run of your experiment or simulation. An ensemble is all of your trials together.
You pass the observations from each trial and get the measures back.

The example builds three systems of ten trials each. In the first every trial behaves the same.
In the second the trials split into two camps. In the third every trial wanders off on its own.

```python
import numpy as np
import incoherence as ic

rng = np.random.default_rng(1)

same = [rng.normal(0, 1, 500) for _ in range(10)]
camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]
scatter = [rng.normal(rng.uniform(-4, 4), rng.uniform(0.3, 2), 500) for _ in range(10)]

for name, trials in [("same", same), ("camps", camps), ("scatter", scatter)]:
    result = ic.measure(trials)
    print(f"{name:8} incoherence={result.incoherence:.2f}  cohesion={result.cohesion:.2f}")
```

```text
same     incoherence=0.03  cohesion=1.00
camps    incoherence=0.57  cohesion=0.83
scatter  incoherence=0.45  cohesion=0.32
```

| Incoherence | Cohesion | What it means |
| :-- | :-- | :-- |
| Low | High | Every trial looks the same, so the average is a fair summary of any one of them. |
| High | High | The trials disagree, but they settle into a few clear scenarios. |
| High | Low | Every trial does its own thing, and the average describes none of them. |

## Three kinds of data

**Real-valued observations.** Pass one array per trial. Trials can hold different numbers of
observations.

```python
result = ic.measure(trials)
```

**Observations you want counted into bins.** Add `bins`. It takes anything that
`numpy.histogram_bin_edges` takes, such as a number of bins, a list of edges or a rule like
`"auto"`. Use this for whole numbers and categories coded as numbers.

```python
result = ic.measure(trials, bins=[0, 1, 2, 3])
```

**A table you already have.** Pass counts or probabilities with one row per trial and one column
per state.

```python
result = ic.measure_counts([[90, 10, 0], [85, 15, 0], [5, 15, 80]])
```

Binned and unbinned data use different estimates of entropy, so the same observations give
different numbers by each route. Pick one route and compare like with like.

## What you get back

Both functions return a `Result` with these attributes. Each one is worked out the first time you
read it, so reading only `incoherence` never pays for `cohesion`.

| Attribute | What it holds |
| :-- | :-- |
| `incoherence` | How much the trials disagree, from 0 to 1. |
| `cohesion` | How strongly the trials gather into shared patterns, from 0 to 1. |
| `entropies` | The entropy of each trial. |
| `pooled_entropy` | The entropy of all the trials pooled into one distribution. |
| `max_entropy` | The largest entropy a trial could have. |
| `divergences` | The divergence between every pair of trials, each from 0 to 1. |

If you only want one number, `ic.incoherence(trials)` and `ic.cohesion(trials)` are shortcuts.
They take the same arguments as `measure`.

## Options

- **`weights`** gives one weight per trial. By default every trial counts the same, however many
  observations it holds.
- **`bounds`** is the lowest and highest value an observation could take. By default it is the
  lowest and highest value seen across all trials. Set it when you compare several ensembles, so
  that they share one scale.
- **`k`** sets how sharply the entropy estimate for real-valued observations picks out detail.
  The default of 10 matches the code behind the papers.

## More documentation

- [FAQ](https://github.com/timjdavey/incoherence/blob/main/docs/faq.md)
  answers what counts as high, how many trials and observations you need, when to set `bounds`
  and whether to bin your data.
- [Examples](https://github.com/timjdavey/incoherence/blob/main/docs/examples.md)
  measures a simulation from start to finish, then uses `divergences` to find which trial
  belongs to which group.
- [Changelog](https://github.com/timjdavey/incoherence/blob/main/CHANGELOG.md)
  lists what changed in each version.

## How it is calculated

Each trial has an entropy `H_k`, which is low when its observations are bunched and high when
they are spread out. Pooling all the trials into one distribution gives the pooled entropy `H̃`.
If the trials agree, pooling changes nothing and `H̃` equals every `H_k`. If they disagree, the
pooled distribution is more spread out than any single trial. Incoherence measures that gap:

```text
I = sqrt( Σ w_k (H̃ − H_k)² / (H̃ · H_max) )
```

Cohesion looks at every pair of trials. For each pair it pools the two and measures the same gap,
which gives one divergence per pair. If the trials fall into a few groups, those divergences
bunch around a few values. Cohesion is high when they do:

```text
d_ab = H̃_ab − (H_a + H_b) / 2
C = (1 − H(D))²        where D is the set of every d_ab
```

Binned data and tables use Shannon entropy in bits. Real-valued observations use the density
estimate described in the papers. The code for each equation is in
[`_measures.py`](https://github.com/timjdavey/incoherence/blob/main/src/incoherence/_measures.py)
and names the equation it implements.

## Differences from the papers

The package was rebuilt from the equations, and it improves on the original design in a few
places.
[Improvements on the papers](https://github.com/timjdavey/incoherence/blob/main/docs/improvements.md)
lists each one with its effect on results. Three things matter when you compare your numbers with
the published ones.

- **Incoherence matches** the code behind the papers to three decimal places, when every trial
  has the same number of observations.
- **Cohesion reads about 5% higher.** The package works it out on a finer grid than the papers
  used, which makes it more accurate.
- **Incoherence of binned data in the Cohesion paper is on a larger scale.** That paper's
  equation 1 has no `H_max` term. The package uses equation 9 of the Incoherence paper, which
  replaced it. Figure 2 of the Cohesion paper prints 0.31 and 0.36 where the package gives 0.16
  and 0.18.

The simulations and notebooks behind the papers live in
[incoherence-papers](https://github.com/timjdavey/incoherence-papers). Corrections to the text of
the papers are kept there too, in `corrections.md`.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # install the package and the development tools
uv run pytest            # run the tests
uv run ruff format .     # format
uv run ruff check .      # lint
uv run pyright           # check types
```

## Citing

If this work is useful to you, please cite the papers. GitHub's "Cite this repository" button
gives the details, which come from
[`CITATION.cff`](https://github.com/timjdavey/incoherence/blob/main/CITATION.cff).

## Licence

[MIT](https://github.com/timjdavey/incoherence/blob/main/LICENSE)
