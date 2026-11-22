# Improvements on the papers

The original `ensemblepy` in [incoherence-papers](https://github.com/timjdavey/incoherence-papers) was developed over time. It wasn't designed from exploration and research in mind, so lacked speed and simplicity. Here we rebuild it as a dedicated, reusable package.

None of the changes below effect the published results. However, the changes in some defaults would cause minor differences for reproducibility.

| # | Improvement | Effect on your numbers |
| :-- | :-- | :-- |
| 1 | A finer grid for cohesion | Cohesion reads about 5% higher than the papers. |
| 2 | An exact top of the entropy scale | Fourth decimal place. No published figure moves. |
| 3 | Every trial counts the same | None when trials are the same size. Two panels of one published figure move. |
| 4 | Bad input raises an error | None on clean data. |
| 5 | No data visualisation | None. |

## 1. A finer grid for cohesion

**What changed.** The last step of cohesion measures how bunched the pairwise divergences are. The
Cohesion paper does this on 500 reference points, which the code behind it ran as 501.

This was for speed with large data analysis. However, here we have increased the default to 2,001 for accuracy.

**Why.** That step uses `k = 100`, so each divergence only reaches about 0.01 either side of
itself. The 501 grid has a point every 0.004. A divergence that falls between two grid points
looks more spread out than one that sits on a point. The top of the scale was set with a value
that sits exactly on a point, so real divergences can never quite reach it.

The clearest case is trials that are almost the same. Their pairwise divergences are all near one
value, so cohesion should be 1. On the 501 grid it depends on where that value lands.

| Every pairwise divergence equals | 501 points | 2,001 points |
| :-- | :-- | :-- |
| 0.004, on a grid point | 1.000 | 1.000 |
| 0.0025, between two points | 0.927 | 0.995 |
| 0.3333, between two points | 0.932 | 0.996 |

**How much.** Cohesion on the 501 grid reads about 5% low across the whole range. Across 1,500
synthetic sets of divergences, the fully converged value was between 1.040 and 1.061 times the
501 value, and the two ranked the sets in almost exactly the same order (rank agreement 0.9999).
The largest move in any one value was 0.08.

The published figures move like this. The 501 column is the code behind the paper, rerun.

| Cohesion paper | Published | 501 points | 2,001 points |
| :-- | :-- | :-- | :-- |
| Figure 2, Current | 0.93 | 0.940 | 0.992 |
| Figure 2, Option A | 0.25 | 0.245 | 0.258 |
| Figure 2, Option B | 0.79 | 0.791 | 0.836 |
| Figures 5 and 6, two patterns | 0.8 | 0.825 | 0.869 |
| Figures 5 and 6, same two patterns with small variation | 0.8 | 0.797 | 0.839 |
| Figures 5 and 6, five patterns | 0.5 | 0.524 | 0.551 |
| Figures 5 and 6, no patterns | 0.1 | 0.141 | 0.148 |

Figure 2 is not seeded in the original notebook, so its rows are the mean of ten runs.

The last two rows show the grid alone. Those two panels use trials of different sizes, so the
package itself gives 0.414 and 0.140. Improvement 3 explains why.

The three test ensembles in `tests/test_paper.py` move like this.

| Test ensemble | 501 points | 2,001 points |
| :-- | :-- | :-- |
| Ten matching trials | 0.951 | 0.996 |
| Two camps | 0.801 | 0.831 |
| Scattered trials | 0.302 | 0.318 |

**What stays the same.** The order of every comparison in the paper. The published values are
correct for the method the paper describes, so this is not a correction to the paper. A grid
finer than 2,001 points moves cohesion by less than 0.004 more.

**What it costs.** Very little. The last step of cohesion takes under 0.001 seconds for 100 trials
and about 0.04 seconds for 1,000. Incoherence at the default `k = 10` still uses 501 points so
that it matches the published figures.

**The same rule for a higher `k`.** The 501 grid is fine enough for any `k` up to 25. Above that
the package adds grid points in step with `k`, so that about ten always sit across the reach of
one observation. At `k = 100` that is 2,001 points, the same as cohesion. On 501 points
incoherence at `k = 100` read about 5% low, and at `k = 1000` it could miss an observation
altogether. No published value moves, because `ensemblepy` always measured incoherence at
`k = 10` and had no option to change it.

**To get the published values back**, pass the divergences to the internal function with the old
grid: `incoherence._measures.cohesion(divergences, points=501)`.

## 2. An exact top of the entropy scale

**What changed.** The entropy of real-valued observations is scaled between two constants. `V0`
is the variance of the densities when observations are spread evenly, which is the highest
entropy. `V1` is the variance when every observation is the same value, which is the lowest. The
code behind the papers found `V0` by drawing 50,000 random numbers once and saving the answer.
The package works `V0` out from the exact integral.

**Why.** The exact value does not depend on a random draw, and it works for any `k`.

**How much.** The exact `V0` differs from the saved one by between 0.02% and 0.18%. That moves
results in the fourth decimal place. For the two camps in those test ensembles, the code behind
the papers gives an incoherence of 0.5712 and the package gives 0.5713. `V1` matches to every
digit.

**Which published figures move.** None. Rerunning the code behind the papers with the exact `V0`
moves Figures 7 and 8 of the Incoherence paper by at most 0.0002, and Figures 2, 5 and 6 of the
Cohesion paper by at most 0.0003. The papers report those values to one or two decimal places.

The `V0` printed in section 2.2 of the Incoherence paper is the saved value, 0.004768391. The
exact value for the same settings is 0.004769250.

## 3. Every trial counts the same

**What changed.** Equation 5 of the Incoherence paper pools the trials by taking a plain average
of their distributions, so every trial counts the same. The code behind the papers did this for
binned data. For real-valued data it pooled by joining all the observations into one list, which
gives a trial with more observations more say. It then averaged the squared differences with
equal weights, so the two halves of the equation disagreed. It did the same for each pair of
trials when working out cohesion. The package uses equal weights everywhere.

**Why.** It is what the equation says. A trial is one run of the system, whether it holds 50
observations or 5,000.

**How much.** Nothing when every trial has the same number of observations. When they differ, the
package will not match the code behind the papers for real-valued data.

**Which published figures move.** Two panels of Figures 5 and 6 in the Cohesion paper, and
nothing else. Every other figure in both papers that uses real-valued data has trials of the same
size. In Figures 5 and 6 each burst is a trial, and the notebook draws the number of observations
in each pattern at random.

| Figures 5 and 6 | Observations per trial | Published | Code behind the paper | Package on 501 points | Package |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Two patterns | 10 | 0.8 | 0.825 | 0.825 | 0.869 |
| Same two patterns with small variation | 10 | 0.8 | 0.797 | 0.797 | 0.839 |
| Five patterns | 3 to 14 | 0.5 | 0.524 | 0.394 | 0.414 |
| No patterns | 3 to 17 | 0.1 | 0.141 | 0.133 | 0.140 |

The 501 column takes out the finer grid, so the gap between it and the code is this change
alone. Both patterns in the top two panels happen to hold 10 observations, so those panels do not
move. The order of the four panels stays the same.

**To make some trials count for more**, pass `weights`. Passing each trial's size as its weight
does not give the old values back, because the old code weighted the two halves differently. It
gives 0.475 and 0.167 for the bottom two panels.

With unequal weights, each pair of trials counts towards cohesion in proportion to the weights of
both trials, so a trial with almost no weight has almost no effect. Equal weights, the default,
give exactly the same cohesion as before.

## 4. Better input sanitation

**What changed.** The code behind the papers often silently dropped missing values and empty trials.
The package raises an error for both. It also raises an error for an observation that falls
outside the bin edges you give, which used to be dropped without a word.

**Why.** A dropped trial changes both measures, and you would not know it had gone.

**How much.** Nothing on clean data.

## 5. Left out

Plotting, comparisons with other statistics and entropy in more than one dimension were not
carried over. They are still in `ensemblepy`, in
[incoherence-papers](https://github.com/timjdavey/incoherence-papers).
