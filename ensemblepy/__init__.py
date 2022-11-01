from .bins import binint, binspace, binobs, binseries
from .collection import Collection
from .continuous import Continuous
from .correlation import Correlation, digitize
from .densityvar import dimspace, minmax_variance, densities, density_variance
from .discrete import Discrete
from .divergences import js_divergence, kl_divergences
from .entropy import ensemble_entropies, point_pmf, pooled_entropy
from .scan import Scan
from .series import Series
from .stats import LEGEND, measures
from .wrapper import incoherence, cohesion