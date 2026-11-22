# incoherence

**Incoherence** says how much repeated trials of a system disagree with each other. [See explainer](https://timjdavey.com/incoherence/) and [paper](https://doi.org/10.3390/e26080683)

**Cohesion** says whether that disagreement falls into a pattern. [See explainer](https://timjdavey.com/cohesion/) and [paper](https://doi.org/10.3390/e25121605)


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
incoherence=0.57  cohesion=0.82
```

## Documentation

- [Examples](https://github.com/timjdavey/incoherence/blob/main/docs/examples.md)
  walks through small ensembles of 1s and 0s, the entropies behind their scores and cohesion.
- [FAQ](https://github.com/timjdavey/incoherence/blob/main/docs/faq.md)
  answers what counts as high, how many trials and observations you need, whether to bin your
  data and when to set `bounds` or `k`.
- [Improvements on the papers](https://github.com/timjdavey/incoherence/blob/main/docs/improvements.md)
  lists how results differ from the code behind the papers, and by how much.
- [Changelog](https://github.com/timjdavey/incoherence/blob/main/CHANGELOG.md)
  lists what changed in each version.
- [Contributing](https://github.com/timjdavey/incoherence/blob/main/CONTRIBUTING.md)
  covers setup, tests and checks.

Every option and every value you can read back is described in the docstrings. Run
`help(ic.measure)` and `help(ic.Result)`, or read
[`_result.py`](https://github.com/timjdavey/incoherence/blob/main/src/incoherence/_result.py).

## Citing and licence

If this work is useful to you, please cite the papers. GitHub's "Cite this repository" button
gives the details. The code is under the
[MIT licence](https://github.com/timjdavey/incoherence/blob/main/LICENSE).
