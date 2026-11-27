"""The result of measuring an ensemble, which works out each measure when first read."""

from functools import cached_property

import numpy as np

from . import _measures, _shuffle
from ._entropy import Array
from ._measures import Entropy

#: Default number of shuffles behind :meth:`Result.floor`.
DEFAULT_REPEATS = 20


class Result:
    """The measures for one ensemble of trials.

    Returned by :func:`measure` and :func:`measure_counts`. You do not build
    one yourself. Each measure is worked out the first time it is read and
    then kept, so reading only ``incoherence`` never pays for ``cohesion``.
    """

    def __init__(
        self,
        rows: Array,
        weights: Array,
        entropy: Entropy,
        max_entropy: float,
        deal: _shuffle.Deal | None = None,
        cohesion_k: float | None = None,
    ) -> None:
        self._rows = rows
        self._weights = weights
        self._entropy = entropy
        self._max_entropy = max_entropy
        self._deal = deal
        self._cohesion_k = cohesion_k

    def __repr__(self) -> str:
        """Return a short description."""
        return (
            f"{type(self).__name__}(trials={len(self._rows)}, incoherence={self.incoherence:.3f})"
        )

    @cached_property
    def entropies(self) -> Array:
        """Entropy of each trial. Bits for discrete data, a value in [0, 1] for continuous."""
        return _read_only(self._entropy(self._rows))

    @cached_property
    def pooled_entropy(self) -> float:
        """Entropy of all the trials pooled into one distribution."""
        return float(self._entropy(_measures.pooled(self._rows, self._weights)))

    @property
    def max_entropy(self) -> float:
        """The largest entropy a trial could have."""
        return self._max_entropy

    @cached_property
    def incoherence(self) -> float:
        """How much the trials disagree with each other, from 0 to 1.

        It is 0 when every trial has the same distribution. For a table of
        counts, it reaches 1 when each trial is certain of one state and the
        pooled trials cover every state evenly. Unused states lower it, so two
        trials certain of different states out of four score 0.71.
        """
        return _measures.incoherence(
            self.entropies, self.pooled_entropy, self.max_entropy, self._weights
        )

    @cached_property
    def divergences(self) -> Array:
        """Jensen-Shannon divergence between every pair of trials, each from 0 to 1.

        The order matches ``numpy.triu_indices(trials, 1)``, so
        ``scipy.spatial.distance.squareform`` turns it into a square matrix.
        """
        return _read_only(_measures.pairwise_divergences(self._rows, self._entropy, self._weights))

    @property
    def cohesion_k(self) -> float:
        """The ``k`` behind ``cohesion``.

        Unless you set ``cohesion_k``, it is 8 for each trial. With unequal
        weights, trials are counted as ``1 / sum(weights ** 2)`` of the weights
        scaled to sum to 1, so a trial with almost no weight barely counts.
        """
        if self._cohesion_k is not None:
            return self._cohesion_k
        return _measures.cohesion_k(_measures.effective_trials(self._weights))

    @cached_property
    def cohesion(self) -> float:
        """How strongly the trials gather into a few shared patterns, from 0 to 1.

        Read it next to ``incoherence``. Trials that all disagree by the same
        amount also score 1, because their pairwise divergences bunch around
        one value. Two trials always score near 1, because they make one pair.
        With unequal weights, a pair counts in proportion to both its weights.
        """
        return _measures.cohesion(
            self.divergences, self.cohesion_k, weights=_measures.pair_weights(self._weights)
        )

    def floor(
        self, repeats: int = DEFAULT_REPEATS, rng: np.random.Generator | int | None = None
    ) -> float:
        """Incoherence these observations would give if every trial came from one source.

        Every observation is pooled, then dealt back out at random into trials
        of the same sizes. The shuffled trials share one source, so whatever
        incoherence is left comes from the sample size alone. An ``incoherence``
        close to its floor may not be a real disagreement.

        The shuffles use the same settings as the measurement. Each one costs
        about as much as measuring ``incoherence`` once.

        Parameters
        ----------
        repeats
            How many times to shuffle. The result is the mean over them.
        rng
            A seed or a ``numpy.random.Generator``. Leave as ``None`` for a
            fresh shuffle on every call, which moves the result a little.

        Raises
        ------
        ValueError
            If the result came from a table of probabilities, because the
            floor depends on how many observations each trial holds.
        """
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        if self._deal is None:
            raise ValueError(
                "floor needs whole-number counts that total less than a billion, "
                "because it deals the observations out again at random"
            )
        generator = np.random.default_rng(rng)
        scores = [
            Result(
                self._deal(generator),
                self._weights,
                self._entropy,
                self._max_entropy,
                cohesion_k=self._cohesion_k,
            ).incoherence
            for _ in range(repeats)
        ]
        return float(np.mean(scores))


def _read_only(values: Array) -> Array:
    """Lock an array, so editing a kept result cannot change later measures."""
    values.setflags(write=False)
    return values
