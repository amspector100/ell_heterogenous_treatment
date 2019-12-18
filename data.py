import numpy as np
import pandas as pd

# Helper
def isfloatable(x):
    try:
        x = float(x)
        return True
    except:
        return False

def read_data(i):
    """ Reads data from folders 1 through 5, just for convenicence"""
    path = f'raw/DS000{i}/31061-000{i}-Data.tsv'
    data = pd.read_csv(path, sep = '\t')
    return data

def pull_classroom_data(treatment_var = 'Ba03_AnyTreat',
						cts_features = ['Ba03_Arnett_PosPunDet',
										'Ba03_desc_lang2'],
						to_binarize = ['Ba03_TeacherEduc'],
						cov_binarize = ['Ba03_Block'],
						response = 'Yr04_oral_language'):
	""" Creates classroom level data """

	# Pull raw data
	d1 = read_data(1)
	d2 = read_data(2)
	d2['Ba03_TeacherEduc'] = d2['Ba03_TeacherEduc'].replace(' ', '5')
	d3 = read_data(3)
	d4 = read_data(4)

	# Join 
	class_data = pd.merge(d2, d3, on = 'Center_ID')
	class_data = pd.merge(class_data, d4, on = 'Center_ID')

	# Add previous version of response to cts features
	response_var = response.split('_')[1:]
	response_var.insert(0, 'Ba03')
	response_var = ('_').join(response_var)
	if response_var not in cts_features and response_var in class_data.columns:
		cts_features = cts_features + [response_var]

	# Ignore the truly irrelevant columns
	all_cols = [treatment_var, response] + to_binarize + cts_features 
	all_cols += cov_binarize
	class_data = class_data[all_cols]

	# Standardize and add intercept
	cts_vars = cts_features + [response]
	class_data[cts_vars] -= class_data[cts_vars].mean()
	class_data[cts_vars] = class_data[cts_vars]/class_data[cts_vars].std()
	class_data['intercept'] = 1

	# Binarize with no interaction effects
	for var in cov_binarize:
	    dummy_vars = pd.get_dummies(class_data[var], drop_first = True)
	    dummy_vars.columns = [str(c) + '_' + var for c in dummy_vars.columns]
	    class_data = pd.concat([dummy_vars, class_data], axis = 'columns')
	    class_data = class_data.drop(var, axis = 'columns')
	    
	# Binarize and create interaction effects
	for var in to_binarize:
	    # Construct dummy. interaction vars
	    dummy_vars = pd.get_dummies(class_data[var], drop_first = True)
	    dummy_vars.columns = [str(c) + '_' + var for c in dummy_vars.columns]
	    interaction_vars = dummy_vars.multiply(
	       class_data[treatment_var], axis = 0
	    )
	    interaction_vars.columns = [
	       'interaction_' + c for c in interaction_vars.columns
	    ]
	    
	    class_data = pd.concat([interaction_vars, dummy_vars, class_data], axis = 'columns')
	    class_data = class_data.drop(var, axis = 'columns')
	    
	# Interaction effects for continuous features
	for var in cts_features:
	    new_var = 'interaction_' + var
	    class_data[new_var] = class_data[var].multiply(
	        class_data[treatment_var], axis = 0
	    )

	# Assemble model params
	X = class_data.drop(response, axis = 'columns')
	y = class_data[response]

	# Return
	return X, y

def pull_student_data(treatment_var = 'Ba03_AnyTreat', 
					  student_features = ['Chld_SexMale', 'Chld_age', 'Chld_LangHome'],
					  response = 'Chld_B_Stndzd',
					  center_features = ['Ba03_Block', 'Ba03_Y23PRE_CT_1',
					  					 'Ba03_AnyTreat'],
					  cov_binarize = ['Ba03_Block'],
					  ):


	# Pull raw data
	d1 = read_data(1)
	d2 = read_data(2)
	d2['Ba03_TeacherEduc'] = d2['Ba03_TeacherEduc'].replace(' ', '0')
	d3 = read_data(3)
	d4 = read_data(4)

	# Need center id to merge
	if 'center_id' not in student_features:
		student_features.append('center_id') 
	data = d1[student_features + [response]]

	# Wrangle child language feature
	data['Lang1'] = (data['Chld_LangHome'] == 1).astype('int')
	data['Lang2'] = (data['Chld_LangHome'] == 2).astype('int')
	data = data.drop('Chld_LangHome', axis = 'columns')

	# Add center features
	if 'Center_ID' not in center_features:
		center_features.append('Center_ID')

	center_data = d2[center_features]
	data = data.merge(center_data, how = 'left', 
	                  left_on = 'center_id', 
	                  right_on = 'Center_ID')
	data = data.drop(['Center_ID'], axis = 'columns')

	# Interaction features
	int_features = student_features + ['Ba03_Y23PRE_CT_1', 'Lang1', 'Lang2']
	int_features.remove('Chld_LangHome')
	int_features.remove('center_id')

	# Binarize with no interaction effects
	for var in cov_binarize:
	    dummy_vars = pd.get_dummies(data[var], drop_first = True)
	    dummy_vars.columns = [str(c) + '_' + var for c in dummy_vars.columns]
	    data = pd.concat([dummy_vars, data], axis = 'columns')
	    data = data.drop(var, axis = 'columns')
	    
	# Interaction effects for all other features
	for var in int_features:
	    new_var = 'interaction_' + var
	    data[new_var] = data[var].astype('float32').multiply(
	        data[treatment_var], axis = 0
	    )

	# Assemble model params
	data = data.loc[data[response].apply(isfloatable)]
	data = data.replace(' ', np.nan)
	center_ids = data['center_id']
	data = data.drop('center_id', axis = 'columns')
	y = data[response].astype('float32')
	y = (y - y.mean())/y.std()
	X = data.drop(response, axis = 'columns').astype('float32')

	# Return
	return X, y