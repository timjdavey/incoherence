# Changelog

Version numbers follow [semantic versioning](https://semver.org). While the version starts with
0, a minor release may change the interface.

## 0.1.0 (unreleased)

First release. A fresh build of the measures in the Incoherence and Cohesion papers, replacing
`ensemblepy`, the library that made their figures.

[Improvements on the papers](docs/improvements.md) lists how results differ from `ensemblepy` and
by how much.

- **Cohesion's `k` grows with trials.** It is 8 for each trial, where the papers fixed it at 100.
  The [k experiments](experiments/k/README.md) found this tells camps from noise better.
- **`cohesion_k` sets it yourself.** Pass `cohesion_k=100` to `measure`, `measure_counts` or
  `cohesion` to match the papers. `Result.cohesion_k` says which value was used.
