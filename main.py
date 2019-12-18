import os
import numpy as np
import pandas as pd

from statsmodels.regression import linear_model
from data import pull_classroom_data, pull_student_data
from models import PostSelectiveLasso, full_grid_search


if __name__ == '__main__':

	# Seed of 186 (we picked this bc we are in Stat 186)
	np.random.seed(186)

	# List of analysis to do.
	# We do not consider phonological awareness or
	# literacy opportunities because data on those  
	# categories is not collected every year
	responses = ['Yr04_print_knowledge',
				 'Yr04_literacy_resources',
				 'Yr04_oral_language',
				 'Yr04_print_motivation',]

	refit = True
	bootstrap = True
	num_bootstrap_samples = 50

	# Iterate through analysis types
	for response in responses:
		print(f'\n -----------{response}---------- ')
		if not refit:
			print(pd.read_csv(f'results/{response}.csv'))

		elif bootstrap:

			X, y = pull_classroom_data(response = response)
			n = X.shape[0]

			# Caching
			old_data_path = f'results/old/{response}_bootstrap_coeffs.csv'
			bootstrapped_data_path = f'results/bootstrap/{response}_bootstrap_coeffs.csv'

			# Check for fully cached data
			if os.path.exists(bootstrapped_data_path):
				coeffs = pd.read_csv(bootstrapped_data_path)
				to_drop = [c for c in coeffs.columns if 'Unnamed' in c]
				coeffs = coeffs.drop(to_drop, axis = 'columns') 
				precomputed = coeffs['seed'].unique()
			else:
				coeffs = pd.DataFrame()
				precomputed = []

			# Check for legacy cached data
			if os.path.exists(old_data_path):
				old_coeffs = pd.read_csv(old_data_path)
				to_drop = [c for c in old_coeffs.columns if 'Unnamed' in c]
				old_coeffs = old_coeffs.drop(to_drop, axis = 'columns') 
				old_precomputed = old_coeffs['seed'].unique()
			else:
				old_coeffs = pd.DataFrame()
				old_precomputed = []

			for b in range(num_bootstrap_samples):

				if b in precomputed:
					continue
				else:
					print(f'Working on computing seed {b}')

				# Bootstrap resamples
				np.random.seed(b)
				sample_inds = np.random.randint(0, high = n, size = n)
				re_X = X.iloc[sample_inds]
				re_y = y.iloc[sample_inds]

				if b in old_precomputed:
					result_params = old_coeffs.loc[old_coeffs['seed'] == b].iloc[0]
					inds = [c for c in result_params.index if c != 'seed']
					result_params = result_params.loc[inds]

				else:
					# Gridsearch
					l1, l2 = full_grid_search(X, y, grid_size = 5)

					# Find nonzero coefficients
					model = linear_model.OLS(re_y, re_X)
					alpha = [
						l1/n if 'interaction' in c else l2/n for c in re_X.columns
					]
					result = model.fit_regularized(
					    alpha = alpha, L1_wt = 1, 
					)
					result_params = result.params

				# Find selected, nonselected variables
				selected = result_params[result_params != 0].index
				nonselected = result_params[result_params == 0].index

				# Run OLS on selected variables only
				selected_model = linear_model.OLS(re_y, re_X[selected])
				selective_coeffs = selected_model.fit().params

				# Cache and combine
				new_coeffs = pd.concat([result_params[nonselected],
									    selective_coeffs])

				new_coeffs['seed'] = int(b)
				coeffs = coeffs.append(new_coeffs, ignore_index = True)
				coeffs.to_csv(bootstrapped_data_path)


		# For each response, cross validate to get l1/l2
		else:
			X, y = pull_classroom_data(response = response)
			l1, l2 = full_grid_search(X, y)

			# Now apply postselective lasso using
			# selectinf package
			k = PostSelectiveLasso(X, y, l1 = l1, l2 = l2)
			selective_lasso_output = k.fit()
			columns = [c for c in selective_lasso_output.columns if 'trunc' not in c]
			selective_lasso_output[columns].to_csv(f'results/{response}.csv')