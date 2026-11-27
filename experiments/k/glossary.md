# Glossary

The one word we use for each idea in these pages.

| Word | What it means |
| :-- | :-- |
| Trial | One distribution, or one set of observations, in the ensemble |
| Draw | One seeded ensemble from a population. A cell uses 200 draws, or 2,000 in phase 2 |
| Population | A recipe for making draws, such as "four tight camps over four options" |
| Cloud | Every trial scattered at random, with no camps. The noise case |
| Camp | A group of trials that sit close together. Exact camps are identical within the camp |
| One source | Every trial drawn from the same distribution, so they differ only by sampling |
| Stray trial | One trial that sits away from every camp. It has its own population |
| Outlier | An unlucky high score. Principle 1 covers it. Not the same as a stray trial |
| Rule | A way of setting k, such as fixed 100 or `8n` |
| Floor | A lowest k, so the rule becomes `max(floor, rule)` |
| Ceiling | A highest k, so the rule becomes `min(ceiling, rule)` |
| Join | How a floor or ceiling meets the growth, such as a hard `max` or an added constant |
| Sampling noise | How far apart two trials from one source sit, only because each has finite observations |
| Map | Every draw scored at a sweep of k, so any rule can be read off as a path through it |
| Noise error | What noise reads, since it should read 0. It is the cloud's mean |
| Camp error | What tight camps fall short of 1 by, since they should read 1. It is 1 less the mean of the 2 and 4 tight camps |
| Weight | How many times the noise error counts against the camp error. We use 2 |
| Weighted error | The noise error times the weight, plus the camp error, divided by the weight plus 1. So 0 is perfect and 1 is as wrong as a rule can be |
| Best k | The k on the map with the lowest weighted error, at one number of trials |
| Band | Every k within a tie of the best k. A rule is good where its path stays inside the band |
| Balance | The k where the error from too low a k equals the error from too high a k |
| Kind of noise | A cloud with one or two parts moved, such as its spread, its options or its observations |
| Tie | Two rules whose difference could be chance. See [how ties are called](design.md#how-ties-are-called) |
| Looseness | How far a camp's trials sit from its centre. A trial is `1 − looseness` of the centre plus `looseness` of its own mix, so 0 is an exact camp, 0.02 is a tight camp and 1 is a cloud |
| Spread | How widely a cloud scatters its trials. It is the concentration of the Dirichlet that draws them. 1 is uniform over every mix, below 1 pushes trials to the corners and above 1 pulls them to the middle |
| Corner | A trial that puts more than 90% on one option |
| Rate | How often a draw from one population outscores a draw from another, over every pairing. A dead heat counts as half |
| Gap | The camps' 5th percentile less the cloud's 95th |
| Winner | The rule phase 2 picks, which is the one with the lowest weighted error over the grid. Phases 3 and 4 score it against fixed 100 |
