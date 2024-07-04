from .bins import binint, binspace, binobs, binseries
from .collection import Collection
from .continuous import Continuous
from .correlation import Correlation, digitize
from .densityvar import dimspace, density_variance, minmax_variance, densities
from .discrete import Discrete
from .divergences import js_divergence, radial_divergences
from .entropy import shannon_entropy, ensemble_entropies, point_pmf, pooled_entropy
from .plots import plot_series, plot_discrete
from .stats import LEGEND, measures
from .wrapper import incoherence, cohesion
