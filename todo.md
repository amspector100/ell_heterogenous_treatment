# Modeling

1. Model plots?
2. 

# Slide outline

## Goal/background
1 slide on background of discoveries (content)
1 slide on previous methods
(a) Something like "Not enough data/noise to do more sophisticated methods"
	We want to show that these methods are viable
(b) More sophisticated methods are more trustworthy - our criterion are pretty
unarbitrary, we didn't make many choices
(c) We want to find new results!

## Data
1-2 slides on what project upgrade is

## Methods
1-2 slides motivating Kosuke's paper
1-2 slides on post-selective inference

## Results
1-2 slides on modeling choices, fit
1-2 slides on actual results and interpretation

# Math-y todos

1. Cluster robust errors for student-level data


# Some reslts

1. l1 = 1, l2 = 5, response = Yr05_print_knowledge
gives 0.0011 pval
2.


# Post-Slides Reslts
1. A lasso for hierarchical interactions
2. Multiple testing selection on selected LASSO

- Differentiate substantive result from original paper
- Kosuke says try to publish, survey literature and clearly
define contribution/what gap you're filling
- There's not much out there for heterogenous
treatment effects in clustered trials, there's
big literature on dynamic treatment regimes
- Would want to weight by the size of the cluster
- Tree type of methods, data that built the tree structure
- Data that you used, even the tree that you use to
estimate the 

Big picture from Kosuke: you need to appeal not only
to methodologists but to people who care about the
substantive contribution of the work as well.


Contribution should be twofold:
1. Substantive contribution
Go through other analyses and explain
why our contribution is different.
IN PARTICULAR Layzer 2011.

(I think the claim we should make is that
the intervention is largely not helpful 
for groups above a certain level, and 
provide preliminary evidence that it can
be harmful for certain high-performing teachers.
This is a step beyond what the substantively lit
already suspects.)

2. Methodological contribution

Go through other analyses and explain problems;
IN PARTICULAR Layzer 2011. 
- Subgroup analysis (run ~50 linear regressions)
- Find results for only some categorical variables,
e.g. lower educational level but not above
- Non-causal analysis on teachers

Go through literature and acknowledge need:
1. Peope saying there's a need
2. Meta-analyses are a bad method


# Thoughts on why our analysis is cool

1. Because we analyze all possible interaction terms and automatically select this one, we can conclude that this is the **causal reason** there are heterogenous treatment effects. 

E.g., other analyses showed that there's some difference in treatment effects between subgroups. This particular analysis 

Note that the linear model is not even 
identifiable in this case, because there are more covariates than data points. 

# Future work
1. Identify