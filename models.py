import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.regression import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score

from selectinf.algorithms.lasso import lasso
import regreg.api as rr

import matplotlib.pyplot as plt


def calc_cv(X, y, l1, l2):
    """ Cross val score for a given l1, l2 """
    alpha = [
        l1 if 'interaction' in c else l2 for c in X.columns
    ]
    wrapped_class = LassoWrapper(sm.OLS, alpha)
    cvscore = cross_val_score(
        wrapped_class, X, y,
        scoring = 'neg_mean_squared_error',
        cv = 5
    )
    return cvscore

def full_grid_search(X, y, grid_size = 10):

	# Parameter grid
	n = X.shape[0]
	l1s = np.logspace(-1.5, 1, grid_size, base = 4)
	# l1s = np.flip(l1s) Doesn't change outcomes - the model
	# is truly picking up signal :)
	l2s = np.logspace(-1.5, 1, grid_size, base = 4)

	# Grid search
	best_l1 = 20
	best_l2 = 20
	best_cvscore = np.inf
	for l1 in l1s:
		for l2 in l2s:

			# MSE - we divide by n to account for differences
			# in the regularization constants in the regreg 
			# vs statsmodels packages
			neg_mse = np.mean(calc_cv(X, y, l1/n, l2/n))
			
			# Adjust for sparsity on whole dataset
			model = linear_model.OLS(y, X)
			alpha = [
				l1/n if 'interaction' in c else l2/n for c in X.columns
			]
			result = model.fit_regularized(
			    alpha = alpha, L1_wt = 1, 
			)
			num_nonzero = sum(result.params != 0)

			# Coeff used to be as in Imai/Ratkovic 2013
			# but then post-selective intervals aren't correct
			# so now we keep it at 1
			coeff = 1#/((1 - num_nonzero/n)**2) 
			cvscore =  -1* coeff * neg_mse

			if cvscore < best_cvscore:
				print(neg_mse, coeff, num_nonzero, l1, l2)
				best_cvscore = cvscore
				best_l1 = l1
				best_l2 = l2

    # Return 
	return best_l1, best_l2


# see https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
class LassoWrapper(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper for statsmodels regularized regression """
    def __init__(self, model_class, alpha):
        self.model_class = model_class
        self.alpha = alpha
    def fit(self, X, y, **kwargs):
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit_regularized(
            alpha = self.alpha, L1_wt = 1
        )
    def predict(self, X):
        return self.results_.predict(X)
    
# Post-selective inference
class PostSelectiveLasso():

	def __init__(self, X, y, l1 = None, l2 = None):

		# Create n, p, loss
		self.n = X.shape[0]
		self.p = X.shape[1]
		self.columns = X.columns
		self.loss = rr.glm.gaussian(X.values, y.values)

		# Penalty
		penalty = rr.l1norm(X.shape[1], lagrange = 1)

		# Lasso class
		if l1 is None or l2 is None:
			l1, l2 = full_grid_search(X, y)

		alpha = np.array(
			[l1 if 'interaction' in c else l2 for c in X.columns]
		)

		self.selectinf_lasso = lasso(
			loglike = self.loss,
			feature_weights = alpha,

		)

	def fit(self, **kwargs):

		self.selectinf_lasso.fit(**kwargs)
		output = self.selectinf_lasso.summary(compute_intervals = True)
		self.colname_dic = {}
		for i, col in enumerate(self.columns):
			self.colname_dic[i] = col

		output.index = output.index.map(self.colname_dic)

		return output



