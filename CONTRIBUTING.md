# Contributing

The project uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # install the package and the development tools
uv run pytest            # run the tests
uv run ruff format .     # format
uv run ruff check .      # lint
uv run pyright           # check types
```

## What CI checks

CI runs the same four checks on every pull request, on Python 3.12, 3.13 and 3.14. The formatter
runs in check mode, so unformatted code fails. One more job uses Python 3.12 with the oldest
dependency versions the package allows. To run that job yourself:

```sh
uv sync --python 3.12 --resolution lowest-direct
uv run --no-sync pytest
git restore uv.lock && uv sync
```

The first line rewrites `uv.lock`. The last line puts it back and discards any uncommitted change
to that file, so commit your own lockfile changes first.

## If your change moves a result

The package implements the measures in two published papers, so a number that moves needs a
written reason. If your change moves any value:

1. Add it to [Improvements on the papers](docs/improvements.md). Say what changed, why, by how
   much and which published figures move.
2. Add it to the [changelog](CHANGELOG.md).
3. Leave the tolerances in `tests/test_paper.py` alone. Those tests hold the package to the papers,
   so a failure there means the change needs explaining.

## Docs

The tests run every `python` block in the README and `docs/`, in order, with one namespace per
page. When a `text` block follows after one blank line, the output must match it exactly.

Numbers anywhere else are not checked. That covers prose and tables, including almost all of the
improvements page. If you change one, rerun it by hand or add a test, as
`test_grid_table_in_the_improvements_page` does.

Docs use British spelling.

## Where tests go

| File | What belongs there |
| :-- | :-- |
| `tests/test_paper.py` | Worked values and stated properties from the papers, naming the section or figure |
| `tests/test_properties.py` | Conditions a measure must meet on any data, using Hypothesis |
| `tests/test_measures.py` | The equations, checked against calculations written out separately |
| `tests/test_inputs.py` | How the package reads and checks what it is given |

Outside Hypothesis, random data must come from a seeded `np.random.default_rng`.

## Scope

Runtime dependencies stay at numpy and scipy. Plotting, comparisons with other statistics and
entropy in more than one dimension were left out on purpose. They are still in `ensemblepy`, in
[incoherence-papers](https://github.com/timjdavey/incoherence-papers).

## Interface and style

The public interface is `__all__` in `src/incoherence/__init__.py`. While the version starts with
0, a minor release may change it, and the change goes in the changelog.

`incoherence._measures.cohesion(divergences, points=501)` counts as public too, because the
improvements page tells readers to use it to get the published values back.

Docstrings follow the numpy convention. Ruff enforces this outside `tests/`.

## Reporting a bug

[Open an issue](https://github.com/timjdavey/incoherence/issues) with the package version, a short
example with a seed, and the values you expected and got.
