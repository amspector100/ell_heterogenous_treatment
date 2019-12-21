""" Basically, does all analysis after models have been fit """

import numpy as np
import pandas as pd
import scipy
from data import pull_classroom_data, pull_student_data, read_data

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from plotnine import *


responses = ['Yr04_print_knowledge',
            'Yr04_literacy_resources',
            'Yr04_oral_language',
            'Yr04_print_motivation',]

def combine_all_statistics(fitprop = 1):
    """ Pulls fit LASSO models """
        
    all_cols = set()
    vanilla_results_dic = {}
    results_dic = {}

    for response in responses:

        # Pull/clean results
        if fitprop == 1:
            path = f'results/{response}.csv'
        else:
            path = f'results/fitprop/{response}_fitprop{fitprop}.csv'
        results = pd.read_csv(path)

        # Rename variable names
        results = results.rename(columns = {'Unnamed: 0': 'varname'})
        results = results.set_index('varname')
        results['response'] = response
        vanilla_results_dic[response] = results.copy().transpose()

        # Rename interaction
        response_2003 = 'Ba03_' + ('_').join(response.split('_')[1:])
        interaction_response = 'interaction_' + response_2003
        results = results.rename(index = {interaction_response:'2003_response_interaction',
                                          response_2003:'2003_response'})


        # Transpose and store
        results = results.transpose()
        results_dic[response] = results

        # List of all variables selected
        all_cols = all_cols.union(set(results.columns.tolist()))
        all_statistics = results.index.tolist()
        
    return all_statistics, all_cols, vanilla_results_dic, results_dic




def create_results_heatmaps(all_statistics, results_dic, fitprop = 1):
    """ Creates heatmaps in slides """
    
    # Assemble final dataframe
    all_results = {x:pd.DataFrame() for x in all_statistics}

    for response in responses:

        # Add dummy columns with NaNs
        results = results_dic[response].copy()
        not_included = list(all_cols - set(results.columns.tolist()))
        results[not_included] = pd.DataFrame(index = results.index,
                                             columns = not_included,
                                             data = np.nan)
        results = results[[c for c in results.columns if 'Block' not in c]]

        # Join all together
        for statistic in all_statistics:
            to_add = results.loc[statistic]
            to_add.name = response
            all_results[statistic] = all_results[statistic].append(to_add, ignore_index = False)

    # Coerce
    pvals = all_results['pval'].transpose()
    pvals['mean'] = pvals.mean(axis = 1)
    pvals['num_na'] = (1-pvals.notnull()).mean(axis = 1)
    pvals = pvals.sort_values(by = ['num_na', 'mean'],ascending = True)
    pvals = pvals.drop(['num_na', 'mean'], axis = 'columns')
    pvals = pvals.transpose()
    pvals.columns = pvals.columns.map(lambda x: x.replace('Ba03', '').strip('_'))
    pvals.index = pvals.index.map(lambda x: x.replace('Yr04_', ''))
    pvals = pvals.apply(lambda x: round(x, 3))

    # Round
    num_features = pvals.shape[1]
    if fitprop == 1:

        for i in range((num_features // 5)+1):

            subset = pvals.iloc[:, 5*i:5*(i+1)]

            # Plot
            fig, ax = plt.subplots(figsize = (20, 10))
            sns.heatmap(subset, cmap = 'Blues_r', ax = ax, annot = True)
            plt.xticks(rotation=0)

            ax.set(xlabel = 'Covariates and Interactions',
                   ylabel = 'Responses')

            # Save and plot
            plt.savefig(f'figures/pvals/heatmap{i}.JPG')
            plt.show()

    return all_results, pvals


def post_process(all_X):
    """ Post-processes CATEs/bootstrapped CATES """
    
    teacher_cols = [c for c in all_X.columns if 'TeacherEdu' in c]
    all_X['1_Ba03_TeacherEdu'] = 1 - all_X[teacher_cols].max(axis = 1)
    teacher_cols = [c for c in all_X.columns if 'TeacherEdu' in c]
    teacher_types = all_X[teacher_cols].idxmax(axis=1).apply(lambda x: x.split('_')[0])
    all_X['teacher_education'] = teacher_types

    edu_type_dict = {'1':'(1) HS',
                     '2':'(2) HS + CDA',
                     '3':'(3) Some College',
                     '4':'(4) College', 
                     '5':'(5) Missing'}
    all_X['teacher_education'] = all_X['teacher_education'].map(edu_type_dict)
    prev_cols = ['Ba03_' + response.replace('Yr04_', '') for response in responses]
    all_X['2003_response'] = all_X[prev_cols].max(axis = 1)
    return all_X


def analyze_ITR(all_X, budget = 100, competitor = False):
    """
    If competitor is True, calculate PAPD between treatment rule
    and simple one from Layzer et al.
    Else, calculate PAPE.
    """
    
    # See: 
    # https://imai.fas.harvard.edu/research/files/indtreat.pdf
    
    # Copy data, figure out how many analyses we're doing simaltaneously
    new_data = all_X.copy()
    new_data = new_data.rename(columns = {'Ba03_AnyTreat':'treat'})
    num_responses = new_data['response'].unique().shape[0]
    
    # Calculate max percent treated
    n = all_X.shape[0]/num_responses
    n1 = new_data['treat'].sum()/num_responses
    pcF = min(1, (num_responses*budget)/new_data.shape[0])
    
    # Threshholds (these are basically the treatment rules)
    threshholds = new_data.groupby(['response'])['point'].quantile(1-pcF)
    threshholds = threshholds.apply(lambda x: max(0, x)) # Never apply the treatment when negatie
    threshholds.name = 'threshhold'
    new_data = pd.merge(new_data, threshholds, on = 'response', how = 'left')

    # Calculate ITR, estimated hat pcF
    new_data['ITR'] = (new_data['threshhold'] <= new_data['point'])
    hat_pcfs = new_data.groupby(['response'])['ITR'].mean()
    hat_pcfs.name = 'pcf'
    new_data = pd.merge(new_data, hat_pcfs, on = 'response', how = 'left')
    
    # Competitor rule based on % spanish speaking
    if competitor:
        
        # Calculate g, then f - g. 
        # First need to perterb proportion to break ties
        new_data['Ba03_desc_lang2'] += 0.001*np.random.uniform(size = all_X.shape[0])
        
        comp_threshhold = new_data['Ba03_desc_lang2'].quantile(1-pcF)
        new_data['compITR'] = (new_data['Ba03_desc_lang2'] >= comp_threshhold).astype('int')
        new_data['diffTR'] = new_data['ITR'] - new_data['compITR']
    
        # Calculate estimator
        new_data['YTdiff'] = new_data['diffTR']*new_data['treat']*new_data['2004_response']
        new_data['Y1T1diff'] = (1-new_data['ITR'])*(1-new_data['treat'])*new_data['2004_response']
        
        # Calculate the two terms
        term1 = new_data.groupby(['response'])['YTdiff'].sum()/n1
        term2 = new_data.groupby(['response'])['Y1T1diff'].sum()/(n-n1)
        
        # This is called the PAPE for consistency but it's really the PAPD
        PAPE = term1 - term2
        
        # Prepare for marginal variance computations
        new_data['Ystar'] = new_data['diffTR']*new_data['2004_response']
        
    
    # Else, no competitor rule: just calculate PAPE
    else:
        # Calculate PAPE estimate - start by coercing columns
        new_data['YTF'] = new_data['ITR']*new_data['treat']*new_data['2004_response']
        new_data['Y1T1F'] = (1-new_data['ITR'])*(1-new_data['treat'])*new_data['2004_response']
        new_data['YT'] = new_data['treat']*new_data['2004_response']
        new_data['Y1T'] = (1-new_data['treat'])*new_data['2004_response']


        # Calculate each term
        term1 = new_data.groupby(['response'])['YTF'].sum()/n1
        term2 = new_data.groupby(['response'])['Y1T1F'].sum()/(n-n1)
        term3 = (hat_pcfs*new_data.groupby(['response'])['YT'].sum())/n1
        term4 = ((1-hat_pcfs)*new_data.groupby(['response'])['Y1T'].sum())/(n-n1)

        # Calculate PAPE
        PAPE = n*(term1 + term2 - term3 - term4)/(n-1)
    
        # Calculate Variance of PAPE
        new_data['Ystar'] = (new_data['ITR'] - new_data['pcf'])*new_data['2004_response']
        
    # Calculate marginal variances - Ystar is different in competitor/non competitor case
    marg_vars = new_data.groupby(['response', 'treat'])['Ystar'].std().unstack()**2
    marg_vars[0] = marg_vars[0]/(n-n1)
    marg_vars[1] = marg_vars[1]/(n1)
    marg_vars = marg_vars.sum(axis = 1)
    
    # Calculate covariance term: depends on whether there is/isn't a budget constraint
    # and also on whether there's a competitor method
    if competitor:
        
        # Budget constraint and PAPD
        Kvalsf = new_data.groupby(['ITR', 'treat', 'response'])['2004_response'].mean().unstack(1)
        Kvalsf = (Kvalsf[1] - Kvalsf[0]).unstack(0)
        Kvalsg = new_data.groupby(['compITR', 'treat', 'response'])['2004_response'].mean().unstack(1)
        Kvalsg = (Kvalsg[1] - Kvalsg[0]).unstack(0)
        
        frac1 = budget*(n - budget)/(n**2*(n-1))
        frac2 = budget*max(budget, n - budget)/(n**2*(n-1))
                
        covterm = frac1*(Kvalsg[1]**2 + Kvalsf[1]**2) + 2*frac2*(Kvalsf[1]*Kvalsg[1])
        
    elif budget >= 155:
        
        # Covariance, no budget constraint
        frac = 1/(n**2)
        hattau = new_data.groupby(['response', 'treat'])['2004_response'].mean().unstack()
        hattau = hattau[1] - hattau[0]
        term3 = PAPE - n*hat_pcfs*(1-hat_pcfs)*(hattau**2) + 2*(n-1)*(2*hat_pcfs - 1)*PAPE*hattau
        covterm = frac*term3        
        
    else:
        
        # Covariance, with budget constraint
        frac = budget*(n - budget)/(n**2*(n-1))
        Kvals = new_data.groupby(['ITR', 'treat', 'response'])['2004_response'].mean().unstack(1)
        Kvals = (Kvals[1] - Kvals[0]).unstack(0)
        term3 = (2*hat_pcfs-1)*(Kvals[True]**2) - 2*hat_pcfs*Kvals[True]*Kvals[False]
        covterm = frac*term3
    
        # BIAS BIAS - ASSUMES LISCHPITZ CONTINUITY
        # This is not implemented in the R version :(
#         eps = 0.005
#         denom_est = new_data.loc[np.abs(new_data['threshhold'] - new_data['point']) < eps]
#         denom_est = denom_est.groupby(['response'])['point'].mean()
#         gamma = eps/(denom_est)
#         print('The gamma is {gamma}')
        
#         term1 = scipy.special.betainc(budget, n - budget + 1, hat_pcfs + gamma)
#         term2 = scipy.special.betainc(budget, n - budget + 1, hat_pcfs - gamma)
#         bias_prob = 1 - term1 + term2
#         print('the bias prob is', bias_prob)
        
    # Variance, SD, pvals
    hatvar = (marg_vars + covterm)*(n/(n-1))**2
    hatsd = np.sqrt(hatvar)
    pvals = 1-scipy.stats.norm.cdf(np.abs(PAPE/hatsd))+ scipy.stats.norm.cdf(-1*np.abs(PAPE/hatsd))
    
    return PAPE, hatsd, pvals, new_data

def analyze_residuals(vanilla_results_dic):

    # Calculate hat Y
    all_resids = pd.DataFrame()

    for response in responses:
        
        # Get model coefficients
        fit = vanilla_results_dic[response]
        model_coef = fit.loc['onestep']

        # Calculate hat Y
        # Calculate
        X, y = pull_classroom_data(response = response)

        # Residuals
        subset1 = X[model_coef.index]
        hatY = np.dot(subset1.values, model_coef.values)
        X['hatY'] = hatY
        X['resid'] = y - X['hatY']
        
        # Renaming
        X['2004_response'] = y
        X['2003_response'] = X[response.replace('Yr04', 'Ba03')]
        X['response'] = response.replace('Yr04_', '')
        
        # Append
        all_resids = pd.concat([all_resids, X], axis = 0)

    # Plot to assess normality
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8))
    all_axes = axes[1].tolist() + axes[0].tolist()
    nresp = all_resids['response'].unique().tolist()


    for r, ax in zip(nresp, all_axes):
        sns.distplot(all_resids.loc[all_resids['response']==r, 'resid'], ax = ax, color = 'navy')
        ax.set(title = f'{r}')

    # Save
    plt.subplots_adjust(hspace = 0.3)
    plt.suptitle(r'Residuals for Fitted Models $\hat \tau(x)$')
    plt.savefig('figures/resids.JPG')
    plt.show()

    # Plot to assess homoskedacicity
    all_resids_new = all_resids.sort_values(['response', '2003_response']).copy()
    all_resids_new['2003_response'] += 0.00001*np.random.uniform(size = all_resids_new.shape[0])
    all_resids_new['2003_response_quintile'] = 0

    for j, q in enumerate([0, 0.2, 0.4, 0.6, 0.8]):
        outs = all_resids_new.groupby(['response'])['2003_response'].quantile(q)
        outs.name = f'2003_quantile{q}'
        all_resids_new = pd.merge(all_resids_new, outs, on = 'response', how = 'left')
        all_resids_new['2003_response_quintile'] += all_resids_new['2003_response'] >= all_resids_new[f'2003_quantile{q}']

    fig, ax = plt.subplots(figsize = (8, 8))
    sds = all_resids_new.groupby(['2003_response_quintile', 'response'])['resid'].std()
    sds.unstack(0).plot(kind = 'bar', cmap = 'Accent', ax = ax)
    ax.set(title = 'Standard Errors for Each Response, Grouped by Baseline Quintile')
    plt.xticks(rotation = 30)
    plt.savefig('figures/homosked.JPG')


if __name__ == '__main__':

    # Pull results
    out = combine_all_statistics(fitprop = 1)
    all_statistics, all_cols, vanilla_results_dic, results_dic = out
    all_results, pvals = create_results_heatmaps(all_statistics, results_dic, fitprop = 1)

    # Print models for selected interaction terms ---

    # Coerce and sort
    points = all_results['onestep']
    points.columns = points.columns.map(lambda x: x.replace('Ba03', '').strip('_'))
    points.index = points.index.map(lambda x: x.replace('Yr04_', ''))
    points = points[pvals.columns]
    points = points.apply(lambda x: round(x, 3))
                
    num_features = points.shape[1]

    for i in range((num_features // 5)+1):

        subset = points.iloc[:, 5*i:5*(i+1)]
        print(subset.to_latex())

    # Calculate tau(x) for each x in dataset
    all_X = pd.DataFrame()
    for response in responses:
        path = f'results/bootstrap/{response}_bootstrap_coeffs.csv'
        true_path = f'results/{response}.csv'

        # Pull true coefficients
        true_coeffs = vanilla_results_dic[response].loc['onestep'].fillna(0)
        true_coeffs = true_coeffs[[c for c in true_coeffs.index if 'interaction' in c or 'Treat' in c]]
        true_coeffs.index = true_coeffs.index.map(lambda x: x.replace('interaction_', ''))

        # Drop useless stuff
        bootstrapped_results = pd.read_csv(path).drop(['Unnamed: 0', 'seed'], axis = 'columns')

        # Only focus on interactions
        interactions = [c for c in bootstrapped_results.columns if 'interaction' in c or 'Treat' in c]
        bootstrapped_results = bootstrapped_results[interactions]
        bootstrapped_results.columns = [c.replace('interaction_', '') for c in bootstrapped_results.columns]

        # Calculate
        X, y = pull_classroom_data(response = response)

        def calc_treatment_effect(X, true_coeffs, bootstrapped_results):

            # Means
            subset1 = X[true_coeffs.index]
            subset1['Ba03_AnyTreat'] = 1
            hattaus = np.dot(subset1.values, true_coeffs.values)
            hattaus = pd.Series(hattaus, index = X.index)
            X['point'] = hattaus


            # Confidence intervals
            subset = X[bootstrapped_results.columns]
            subset['Ba03_AnyTreat'] = 1 
            taus = np.dot(subset.values, bootstrapped_results.values.T)
            taus = pd.DataFrame(taus, index = X.index)
            X['lower'] = taus.quantile(0.025, axis = 1)
            X['upper'] = taus.quantile(0.975, axis = 1)
            X['pval'] = (1+(X['point'] < taus).sum(axis = 1))/(taus.shape[1])

            return X

        # Treatment effects
        X = calc_treatment_effect(X, true_coeffs, bootstrapped_results)
        X = X.drop([c for c in X.columns if 'interaction' in c], axis = 'columns')
        X['response'] = response.replace('Yr04_', '')
        X['2004_response'] = y
        all_X = pd.concat([all_X, X], axis = 0)
        
    all_X = post_process(all_X)

    # Plot individualized treatment effects
    # First the marginal distributions
    g = (
        ggplot(all_X, aes(x = 'point', fill = 'response',
                            color = 'response'))
        + geom_histogram(alpha = 0.7)#, show_legend = False)
        + facet_wrap('~response')
        + labs(title = 'Distributions of Treatment Effects',
               x = 'Estimated Treatment Effect')
    )
    g.save('figures/hattaudists.JPG')

    # Then the CDFs/ranked hattaus
    all_X_modified = all_X.copy()
    hattau_ranks = all_X_modified.groupby(['response'])['point'].rank()#.unstack()
    all_X_modified['rank'] = hattau_ranks

    id_vars = [c for c in all_X_modified.columns if c not in ['point', 'upper', 'lower']]
    all_X_modified = all_X_modified.melt(id_vars = id_vars, var_name = 'estimand')

    g = (
        ggplot(all_X_modified, aes(x = 'rank',y = 'value', fill = 'estimand',
                            color = 'estimand'))
        + geom_point()#show_legend = False)
        + facet_wrap('~response')
        + labs(title = r'Distribution of $\hat \tau(x)$ in the Dataset',
               x = 'Rank', y = 'Estimated Treatment Effect')
        + theme(legend_position = None)
    )
    g.save('figures/hattaudist.JPG')


    # And then the 2003 response vs. CATE distributions
    g = (
        ggplot(all_X, aes(x = '2003_response',y = 'point', fill = 'response',
                            color = 'response'))
        + geom_point()#show_legend = False)
        + facet_wrap('~response')
        + labs(title = 'Effect of Previous Response on Estimated Treatment Effect',
               x = 'Response in 2003', y = 'Estimated Treatment Effect (Sd)')
        + theme(legend_position = None)
    )
    g.save('figures/hattauresponse.JPG')


    # Next, DERIVE ITRs, PAPE, PAPD -------------------

    # Recreate allX, but the models are only fit on split data
    fitprop = 0.65
    out2 = combine_all_statistics(fitprop = fitprop)
    all_statistics2, all_cols2, vanilla_results_dic2, results_dic2 = out2
    all_X2 = pd.DataFrame()
    for response in responses:
        
        # Pull true coefficients
        true_coeffs = vanilla_results_dic2[response].loc['onestep'].fillna(0)
        true_coeffs = true_coeffs[[c for c in true_coeffs.index if 'interaction' in c or 'Treat' in c]]
        true_coeffs.index = true_coeffs.index.map(lambda x: x.replace('interaction_', ''))

        # Calculate for ALL data - later, we'll only include
        # data which wasn't used to fit the models
        X, y = pull_classroom_data(
            response = response,
            cts_features = [
                'Ba03_Arnett_PosPunDet', 'Ba03_desc_lang2', 'Ba03_desc_lang5',
                'Yr05_pct_adlt_eng',
            ],
        )

        def calc_treatment_effect(X, true_coeffs, bootstrapped_results):

            # Means
            subset1 = X[true_coeffs.index]
            if 'Ba03_AnyTreat' in subset1.columns:
                subset1['Ba03_AnyTreat'] = 1
            print(f'Num selected vars is {len(subset1.columns)} for {response}')
                
            hattaus = np.dot(subset1.values, true_coeffs.values)
            hattaus = pd.Series(hattaus, index = X.index)
            X['point'] = hattaus

            return X

        # Treatment effects
        X = calc_treatment_effect(X, true_coeffs, bootstrapped_results)
        X = X.drop([c for c in X.columns if 'interaction' in c], axis = 'columns')
        X['response'] = response.replace('Yr04_', '')
        X['2004_response'] = y
        all_X2 = pd.concat([all_X2, X], axis = 0)

    
    all_X2 = post_process(all_X2)

    # Figure out which centers we haven't already viewed
    n = X.shape[0]
    num_obs = int(X.shape[0]*fitprop)
    rand_inds = np.arange(0, n, 1)
    np.random.seed(186)
    np.random.shuffle(rand_inds)
    to_select = rand_inds[num_obs:]
    all_X2 = all_X2.loc[to_select]

    # Finally, PAPE 
    PAPE, hatsd, pvals, _ = analyze_ITR(all_X = all_X2, budget = 28, competitor = False)
    PAPD, hatsdD, pvalsD, _ = analyze_ITR(all_X = all_X2, budget = 28, competitor = True)

    # Combine all of this into one dataframe
    output_df = pd.DataFrame()
    output_df['PAPE'] = PAPE
    output_df['PAPEalower'] = PAPE - 1.96*hatsd
    output_df['PAPEupper'] = PAPE + 1.96*hatsd
    output_df['sd (PAPE)'] = hatsd
    output_df['pval (PAPE)'] = pvals
    output_df['PAPD'] = PAPD
    output_df['PAPDlower'] = PAPD - 1.96*hatsdD
    output_df['PAPDupper'] = PAPD + 1.96*hatsdD
    output_df['sd (PAPD)'] = hatsdD
    output_df['pval (PAPD)'] = pvalsD
    output_df = np.around(output_df, 3)


    # Print (two responses have useless models)
    print(output_df.loc[['literacy_resources', 'print_knowledge']])#.to_latex())
    
    # Fnally, analyze residuals ---------
    analyze_residuals(vanilla_results_dic)