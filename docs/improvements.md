# Improvements on the papers

The original `ensemblepy` in [incoherence-papers](https://github.com/timjdavey/incoherence-papers) was developed over time. It wasn't designed from exploration and research in mind, so lacked speed and simplicity. Here we rebuild it as a dedicated, reusable package.

None of the changes below affect the published conclusions. Some defaults differ, so rerunning the papers gives different numbers. The table says by how much.

| # | Improvement | Effect on your numbers |
| :-- | :-- | :-- |
| 1 | Cohesion's `k` grows with trials | Small from 10 trials. Figure 2 of the Cohesion paper, with 5 trials, moves by up to 0.23. |
| 2 | A finer grid for cohesion | Takes out a bias that read cohesion about 5% low. |
| 3 | An exact top of the entropy scale | Fourth decimal place. No published figure moves. |
| 4 | Every trial counts the same | None when trials are the same size. Two panels of one published figure move. |
| 5 | Bad input raises an error | None on clean data. |
| 6 | No data visualisation | None. |

## 1. Cohesion's k grows with trials

**What changed.** Cohesion's last step spreads each pairwise divergence into a narrow bump, and
`k` sets how narrow. The papers fixed it at 100. The package sets it to 8 for each trial, so 10
trials give 80 and 400 give 3,200.

**Why.** More trials give more pairs, and more pairs can show finer detail. A fixed `k` can only be
right at one number of trials. The [k experiments](../experiments/k/README.md) tested seven rules
from 3 to 400 trials, and `8n` read noise lowest and camps nearly highest. Fixed 100 came last.

**How much.** Little from 10 trials, and a lot with fewer. The package column is the default, so
it also holds improvements 2 and 4.

| Ensemble | Trials | Published | Code behind the paper | Package |
| :-- | :-- | :-- | :-- | :-- |
| Figure 2, Current | 5 | 0.93 | 0.940 | 0.993 |
| Figure 2, Option B | 5 | 0.79 | 0.791 | 0.615 |
| Figure 2, Option A | 5 | 0.25 | 0.245 | 0.028 |
| Figures 5 and 6, two patterns | 40 | 0.8 | 0.825 | 0.874 |
| Figures 5 and 6, same two patterns with small variation | 40 | 0.8 | 0.797 | 0.827 |
| Figures 5 and 6, five patterns | 40 | 0.5 | 0.524 | 0.426 |
| Figures 5 and 6, no patterns | 40 | 0.1 | 0.141 | 0.135 |
| Test ensemble, ten matching trials | 10 | | 0.951 | 0.996 |
| Test ensemble, two camps | 10 | | 0.801 | 0.794 |
| Test ensemble, scattered trials | 10 | | 0.302 | 0.313 |

Figure 2 is not seeded in the original notebook, so its rows are the mean of ten runs. The test
ensembles are the three in `tests/test_paper.py`.

**What stays the same.** The order of every comparison in the papers. Figure 2 has only 5 trials,
so its `k` falls to 40 and Options A and B both read lower.

**To get the published values back**, pass `cohesion_k=100` for the papers' `k` on the finer grid.
To match the code behind the papers exactly, for trials of the same size, pass the divergences to
the internal function with the old grid: `incoherence._measures.cohesion(divergences, 100, points=501)`.

## 2. A finer grid for cohesion

**What changed.** Cohesion's last step measures how bunched the pairwise divergences are, on a grid
of reference points. The Cohesion paper used 500 points, which the code behind it ran as 501. The
package puts ten points across the reach of each divergence, which is `1 / k`. So 10 trials use
1,601 points and 400 trials use 64,001.

**Why.** At the papers' `k = 100` each divergence only reaches about 0.01 either side of itself, and
the 501 grid has a point every 0.004. A divergence that falls between two grid points looks more
spread out than one that sits on a point. The clearest case is trials that are almost the same.
Their divergences are all near one value, so cohesion should be 1. On the 501 grid it depends on
where that value lands.

| Every pairwise divergence equals | 501 points | 2,001 points |
| :-- | :-- | :-- |
| 0.004, on a grid point | 1.000 | 1.000 |
| 0.0025, between two points | 0.927 | 0.995 |
| 0.3333, between two points | 0.932 | 0.996 |

**How much.** The 501 grid read cohesion about 5% low across the whole range. Across 1,500
synthetic sets of divergences, the converged value was between 1.040 and 1.061 times the 501
value, and the two ranked the sets in almost exactly the same order (rank agreement 0.9999). A grid
four times finer than the package's moves cohesion by less than 0.004 more, at every `k` from 24
to 12,800. The [grid check](../experiments/k/results/confirm.txt) has the numbers.

**What it costs.** The last step takes under 0.001 seconds for 100 trials and about 0.1 seconds for
1,600, against 0.05 seconds at `k = 100`. The [cost check](../experiments/k/results/cost.txt)
has the numbers.

**Incoherence uses the same rule.** At its default `k = 10` that is still 501 points, so it matches
the published figures. Above `k = 25` the grid grows in step with `k`, where 501 points would read
about 5% low at `k = 100` and could miss an observation at `k = 1000`.

## 3. An exact top of the entropy scale

**What changed.** The entropy of real-valued observations is scaled between two constants. `V0`
is the variance of the densities when observations are spread evenly, which is the highest
entropy. `V1` is the variance when every observation is the same value, which is the lowest. The
code behind the papers found `V0` by drawing 50,000 random numbers once and saving the answer.
The package works `V0` out from the exact integral.

**Why.** The exact value does not depend on a random draw, and it works for any `k`.

**How much.** The exact `V0` differs from the saved one by between 0.02% and 0.18%. For the two
camps test ensemble, incoherence moves from 0.5712 to 0.5713. `V1` matches to every digit.

**Which published figures move.** None. Figures 7 and 8 of the Incoherence paper move by at most
0.0002, and Figures 2, 5 and 6 of the Cohesion paper by at most 0.0003. The papers report those
values to one or two decimal places.

The `V0` printed in section 2.2 of the Incoherence paper is the saved value, 0.004768391. The
exact value for the same settings is 0.004769250.

## 4. Every trial counts the same

**What changed.** Equation 5 of the Incoherence paper pools the trials by taking a plain average
of their distributions, so every trial counts the same. The code behind the papers did this for
binned data. For real-valued data it joined all the observations into one list, which gives a
trial with more observations more say, and then averaged the squared differences with equal
weights. So the two halves of the equation disagreed. The package uses equal weights everywhere.

**Why.** It is what the equation says. A trial is one run of the system, whether it holds 50
observations or 5,000.

**Which published figures move.** Only the five patterns and no patterns panels of Figures 5 and 6
in the Cohesion paper, whose trials hold between 3 and 17 observations. Every other figure that
uses real-valued data has trials of the same size. [Improvement 1](#1-cohesions-k-grows-with-trials)
shows their values.

**To make some trials count for more**, pass `weights`. Each pair of trials then counts towards
cohesion in proportion to the weights of both, so a trial with almost no weight has almost no
effect. Passing each trial's size does not give the old values back, because the old code
weighted the two halves differently.

## 5. Bad input raises an error

**What changed.** The code behind the papers often silently dropped missing values and empty trials.
The package raises an error for both. It also raises an error for an observation that falls
outside the bin edges you give, which used to be dropped without a word.

**Why.** A dropped trial changes both measures, and you would not know it had gone.

## 6. Left out

Plotting, comparisons with other statistics and entropy in more than one dimension were not
carried over. They are still in `ensemblepy`, in
[incoherence-papers](https://github.com/timjdavey/incoherence-papers).
