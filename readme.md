# Overview

# Analysis thoughts

Initial model: hierarchical
https://www.acf.hhs.gov/sites/default/files/opre/upgrade_miami_dade.pdf
see page 42

# Data thoughts

## DS0001 

Child-level data

## DS0002

2003 classroom/center level data

## DS0003

2004 classroom/center level data

## DS0004

2005 classroom/center level data

## DS0005

Follow up data!


# Try implementing them separately
Implement normal random forest for treatment/control
separately, and then compute the difference, see
if you pick up something. 

Stefan Wager, Stanford: https://arxiv.org/abs/1902.07409

CI for L1 methods: https://economics.mit.edu/files/12538
Have to split at the cluster level

Post-selective inference at the cluster level? 
https://arxiv.org/abs/1205.5050 

What's the right way to present a result conditional
on the Xis?
1. Can always sort by the size of the effects
2. That's why the sparsity methods are helpful:
look at the variables which are actually selected.
In public policy context, it's important to explain
to policy makers to say "these kind of students will
be helped by this program."

Cincia Rudin: MIT, now at Duke: they're doing decision trees, trying to make them interpretable. If you're this
type of person, and then this type