# Contributing

The project uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync                  # install the package and the development tools
uv run pytest            # run the tests
uv run ruff format .     # format
uv run ruff check .      # lint
uv run pyright           # check types
```

CI runs the same four checks on every pull request.

The tests also run every code block in the README and the docs, and compare what each one
prints with the page. If you change a number in the docs, the tests will tell you whether the
package agrees.
