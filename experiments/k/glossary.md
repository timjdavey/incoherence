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
| Best k | The k on the map that best keeps camps and noise apart, at one number of trials |
| Band | Every k within a tie of the best k. A rule is good where its path stays inside the band |
| Balance | The k where the error from too low a k equals the error from too high a k |
| Pass bar | A limit a rule must meet before anything else about it counts |
| Tie | Two rules whose difference could be chance. See [how ties are called](design.md#how-ties-are-called) |
