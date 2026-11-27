# incoherence

[![CI](https://github.com/timjdavey/incoherence/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/timjdavey/incoherence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://github.com/timjdavey/incoherence/blob/main/pyproject.toml)
[![MIT licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/timjdavey/incoherence/blob/main/LICENSE)

**Incoherence** says how much repeated trials of a system disagree with each other. [See explainer](https://timjdavey.com/incoherence/) and [paper](https://doi.org/10.3390/e26080683)

**Cohesion** says whether that disagreement falls into a pattern. [See explainer](https://timjdavey.com/cohesion/) and [paper](https://doi.org/10.3390/e25121605)

<table>
  <tr>
    <td width="33%"><img src="https://raw.githubusercontent.com/timjdavey/incoherence-papers/371249c97d2ac95a277cd47be35ca99d2581a0a0/notebooks/cohesion/low-inc.png" alt="Trials whose distributions all lie on top of each other"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/timjdavey/incoherence-papers/371249c97d2ac95a277cd47be35ca99d2581a0a0/notebooks/cohesion/low-coh.png" alt="Trials whose distributions each sit in a different place"></td>
    <td width="33%"><img src="https://raw.githubusercontent.com/timjdavey/incoherence-papers/371249c97d2ac95a277cd47be35ca99d2581a0a0/notebooks/cohesion/high-coh.png" alt="Trials that split into two tight clusters"></td>
  </tr>
  <tr>
    <td align="center"><b>Low incoherence</b><br><sub>Every trial looks the same, so the average is a fair summary of any one of them.</sub></td>
    <td align="center"><b>High incoherence, low cohesion</b><br><sub>Every trial does its own thing, and the average describes none of them.</sub></td>
    <td align="center"><b>High incoherence, high cohesion</b><br><sub>The trials disagree, but they organise themselves into a few clear camps.</sub></td>
  </tr>
</table>

## Install

The package is not on PyPI yet, so install it from GitHub. It needs Python 3.12 or later.

```sh
uv add git+https://github.com/timjdavey/incoherence
# or
pip install git+https://github.com/timjdavey/incoherence
```

## Use

Pass the observations from each trial and read the measures off the result.

```python
import numpy as np
import incoherence as ic

rng = np.random.default_rng(1)
camps = [rng.normal(3 if i % 2 else -3, 1, 500) for i in range(10)]  # ten trials in two camps

result = ic.measure(camps)
print(f"incoherence={result.incoherence:.2f}  cohesion={result.cohesion:.2f}")
```

```text
incoherence=0.57  cohesion=0.78
```

The trials disagree (incoherence of 0.57), but that disagreement has a clear pattern (cohesion of
0.78), because they form two camps. The FAQ explains
[what counts as high](https://github.com/timjdavey/incoherence/blob/main/docs/faq.md#what-counts-as-high).

## Documentation

- [Examples](https://github.com/timjdavey/incoherence/blob/main/docs/examples.md)
  walks through small ensembles of 1s and 0s, the entropies behind their scores and cohesion.
- [FAQ](https://github.com/timjdavey/incoherence/blob/main/docs/faq.md)
  answers what counts as high, how many trials and observations you need, whether to bin your
  data and when to set `bounds`, `k` or `cohesion_k`.
- [Improvements on the papers](https://github.com/timjdavey/incoherence/blob/main/docs/improvements.md)
  lists how results differ from the code behind the papers, and by how much.

Every option and every value you can read back is described in the docstrings. Run
`help(ic.measure)` and `help(ic.Result)`, or read
[`_result.py`](https://github.com/timjdavey/incoherence/blob/main/src/incoherence/_result.py).

## Cite

If this work is useful to you, please cite the papers.

```bibtex
@article{davey2024incoherence,
  author  = {Davey, Timothy},
  title   = {Incoherence: A Generalized Measure of Complexity to Quantify Ensemble Divergence in Multi-Trial Experiments and Simulations},
  journal = {Entropy},
  volume  = {26},
  number  = {8},
  pages   = {683},
  year    = {2024},
  doi     = {10.3390/e26080683}
}

@article{davey2023cohesion,
  author  = {Davey, Timothy},
  title   = {Cohesion: A Measure of Organisation and Epistemic Uncertainty of Incoherent Ensembles},
  journal = {Entropy},
  volume  = {25},
  number  = {12},
  pages   = {1605},
  year    = {2023},
  doi     = {10.3390/e25121605}
}
```

## Project

- [Changelog](https://github.com/timjdavey/incoherence/blob/main/CHANGELOG.md)
  lists what changed in each version.
- [Contributing](https://github.com/timjdavey/incoherence/blob/main/CONTRIBUTING.md)
  covers setup, tests and checks.
- The code is under the [MIT licence](https://github.com/timjdavey/incoherence/blob/main/LICENSE).
