import os
import numpy as np
import pandas as pd

from statsmodels.regression import linear_model
from data import pull_classroom_data, pull_student_data
from models import PostSelectiveLasso, full_grid_search

import sys
import argparse


def str2bool(v):
	""" Helper function, converts strings to boolean vals""" 
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def cte_boostrap(response):
	""" Bootstraps coefficients for a response """

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


def main(args):

	# Description: runs experiments
	description = 'Simulates power, fdr of various knockoff methods'
	parser = argparse.ArgumentParser(description = description)


	# Add arguments
	parser.add_argument('--refit', dest = 'refit',
					type=str,
					help='If true, will refit responses (default: False)',
					default = 'False')

	parser.add_argument('--latex', dest = 'latex',
					type=str,
					help='If true, will print datasets in latexed format (default: False)',
					default = 'False')

	parser.add_argument('--bootstrap', dest = 'bootstrap',
					type=str,
					help='If true, do bootstrap lasso feature statistic (default: False)',
					default = 'False')

	parser.add_argument('--response', dest = 'response',
					type=str,
					help='Which response to fit. (Default: "all")',
					default = 'all')

	parser.add_argument('--fitprop', dest = 'fitprop',
					type=float,
					help='How much data to use to fit the model. (Default: 1)',
					default = 1)


	# Parse args, including some boolean flags
	args = parser.parse_args()
	args.refit = str2bool(args.refit)
	refit = args.refit
	args.bootstrap = str2bool(args.bootstrap)
	bootstrap = args.bootstrap
	args.latex = str2bool(args.latex)

	print(f"Parsed args are {args}")


	# List of analysis to do.
	# We do not consider phonological awareness or
	# literacy opportunities because data on those  
	# categories is not collected every year
	response = str(args.response).lower()
	if response == 'all':
		responses = ['Yr04_print_knowledge',
					 'Yr04_literacy_resources',
					 'Yr04_oral_language',
					 'Yr04_print_motivation',]
	elif 'oral' in response:
		responses = ['Yr04_oral_language']
	elif 'literacy' in response:
		responses = ['Yr04_literacy_resources']
	elif 'print' in response and 'knowledge' in response:
		responses = ['Yr04_print_knowledge']
	elif 'print' in response and 'motivation' in response:
		responses = ['Yr04_print_motivation']
	else:
		raise ValueError(f'Response {response} unrecognized')


	# Seed of 186 (we picked this bc we are in Stat 186)
	np.random.seed(186)

	num_bootstrap_samples = 50

	# Iterate through analysis types
	for response in responses:
		print(f'\n -----------{response}---------- ')
		if not refit:
			data = pd.read_csv(f'results/{response}.csv')
			if not args.latex:
				print(data)
			else:

				# Make pretty for latex
				rename_vars = {}
				for ind in data['Unnamed: 0']:
					name = str(ind)
					name = name.replace('Ba03_', '2003_')
					if 'desc_lang2' in name:
						name = 'Pct Spanish Speaking'
						if 'interaction' in ind:
							name += ' (interaction)'

					if 'AnyTreat' in name:
						name = 'Treatment'


					if 'TeacherEdu' in name:
						if '4_' in name:
							name = 'Teacher Edu (College)'
						if '3_' in name:
							name = 'Teacher Edu (Some College)'
						if '2_' in name:
							name = 'Teacher Edu (HS+CDA)'


						if 'interaction' in ind:
							name += ' (interaction)'

					rename_vars[ind] = name

				# Rename
				data['Unnamed: 0'] = data['Unnamed: 0'].apply(lambda x: rename_vars[x])
				data = data.drop(['lasso', 'variable'], axis='columns')
				data = data.rename(
					columns={'Unnamed: 0':"Variable", "onestep":"point",
							 'upper_confidence':'upper', 'lower_confidence':'lower'}
				)
				data = data.set_index('Variable')
				data = np.around(data, 3)

				# Save
				data.to_latex(f'results/tex/{response}-latex.tex')

		# For each response, cross validate to get l1/l2
		else:
			X, y = pull_classroom_data(response = response)


			if args.fitprop == 1:
				outpath = f'results/{response}.csv'
				l1, l2 = full_grid_search(X, y)

				# Now apply postselective lasso using
				# selectinf package
				k = PostSelectiveLasso(X, y, l1 = l1, l2 = l2)
				selective_lasso_output = k.fit()
				columns = [c for c in selective_lasso_output.columns if 'trunc' not in c]
				selective_lasso_output[columns].to_csv(outpath)

			else:

				fitprop = args.fitprop
				if fitprop == -1:
					fitprops = [0.73, 0.76, 0.8, 0.85]
				else:
					fitprops = [fitprop]

				# Select random subset of the data for PAPE estimation
				for fitp in fitprops:
					n = X.shape[0]
					num_obs = int(X.shape[0]*fitp)
					rand_inds = np.arange(0, n, 1)

					# Set seed, shuffle
					np.random.seed(186)
					np.random.shuffle(rand_inds)
					rand_inds = rand_inds[0:num_obs]
					Xsub = X.loc[rand_inds]
					ysub = y.loc[rand_inds]

					# Output path
					outpath = f'results/fitprop/{response}_fitprop{fitp}.csv'



					l1, l2 = full_grid_search(Xsub, ysub)

					# Now apply postselective lasso using
					# selectinf package
					k = PostSelectiveLasso(Xsub, ysub, l1 = l1, l2 = l2)
					selective_lasso_output = k.fit()
					columns = [c for c in selective_lasso_output.columns if 'trunc' not in c]
					selective_lasso_output[columns].to_csv(outpath)

		if bootstrap:

			cte_bootstrap(response = response)



if __name__ == '__main__':

	sys.exit(main(sys.argv))