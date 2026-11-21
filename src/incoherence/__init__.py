"""Incoherence and cohesion for ensembles of trials.

Incoherence measures how much repeated trials of a system disagree with each
other. Cohesion measures whether that disagreement falls into a few shared
patterns. Both are defined in:

- Davey, T. (2024). Incoherence. Entropy, 26(8), 683.
- Davey, T. (2023). Cohesion. Entropy, 25(12), 1605.
"""

from ._result import Result, cohesion, incoherence, measure, measure_counts

__all__ = ["Result", "cohesion", "incoherence", "measure", "measure_counts"]
