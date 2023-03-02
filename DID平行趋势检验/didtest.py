import pandas as pd
import numpy as np
from scipy.stats import norm
from plotnine import ggplot, aes, geom_bar, scale_x_continuous, xlab, ylab
from copy import deepcopy
from numpy.random import seed, choice


def calculator(df, columns):
    weighted_sum = (df[columns[0]] * df[columns[1]]).sum() / df[columns[0]].sum()
    return weighted_sum


def compute_implied_density(start_year, end_year, data):
    compare_pre_post = data[((start_year <= data.year) & (data.year <= end_year))]
    tmp_df = pd.pivot_table(compare_pre_post, index=['statenum'], values=['treated'], \
                            aggfunc=np.max).reset_index()
    tmp_df = tmp_df.rename(columns={'treated': 'treated_in_period'})
    compare_pre_post = compare_pre_post.merge(tmp_df, on='statenum', how='left')
    compare_pre_post = compare_pre_post[(compare_pre_post.year == start_year) | (compare_pre_post.year == end_year)]
    long_summary_table = compare_pre_post.groupby(['year', 'wagebins', 'treated_in_period']). \
        apply(lambda x: calculator(x, ['population', 'overallcountpc'])).reset_index()
    long_summary_table = long_summary_table.rename(columns={0: 'employment_per_capita'})

    wide_summary_table = pd.pivot_table(long_summary_table, index=['wagebins'], columns=['year', 'treated_in_period'],
                                        values=['employment_per_capita'])

    rename_the_df = []
    # 遍历原多维列名称
    for x in wide_summary_table.columns:
        rename_the_df.append(str(x[1]) + '_' + str(x[2]))
    # 赋值
    wide_summary_table.columns = rename_the_df
    wide_summary_table.reset_index(inplace=True)
    wide_summary_table['implied_density_post'] = wide_summary_table[str(start_year) + '_1'] - wide_summary_table[
        str(start_year) + '_0'] + wide_summary_table[str(end_year) + '_0']
    return wide_summary_table


def implied_density_plot(start_year, end_year, data, minwagebin=500, maxwagebin=3000):
    cid = compute_implied_density(start_year, end_year, data)
    cid = cid[(minwagebin <= cid.wagebins) & (cid.wagebins < maxwagebin)]
    cid['wage'] = cid['wagebins'] / 100
    tmp = deepcopy(cid['implied_density_post'].values.astype(str))
    tmp[cid['implied_density_post'] < 0] = 'Negative'
    tmp[cid['implied_density_post'] >= 0] = 'non-Negative'
    cid['Implied Employment'] = tmp
    cid.index = range(len(cid))
    p = ggplot(cid, aes(x='wage', y='implied_density_post', fill='Implied Employment')) + geom_bar(
        stat="identity") + xlab("Wage Bin") + ylab("Employment-to-pop") + scale_x_continuous(breaks=np.arange(5, 5, 31))
    return p


def bootstrap_draw(seed_set, data):
    seed(seed_set)
    states = np.unique(data['statenum'])
    randx = choice(np.arange(len(states)), len(states), replace=True)
    bootstrap_states = states[randx]
    tmp = np.hstack((np.arange(len(states)).reshape(-1, 1), bootstrap_states.reshape(-1, 1)))
    statesDF = pd.DataFrame(tmp, columns=['statenum_bootstrap', 'bootstrap_state'])
    bootstrap_DF = statesDF.merge(data, how='left', left_on=['bootstrap_state'], right_on=['statenum'])
    del bootstrap_DF['statenum']
    bootstrap_DF = bootstrap_DF.rename(columns={'bootstrap_state': 'statenum'})
    return bootstrap_DF


def lf_moment_inequality_test(muhat, Sigmahat, numSims=1000, seed_set=0):
    seed(seed_set)
    sims = np.random.multivariate_normal(0 * muhat, Sigmahat, numSims)
    sims_max = np.max(sims, axis=1)

    p_value = np.mean(sims_max > np.max(muhat / np.sqrt(np.diag(Sigmahat))))
    return p_value


def compute_bootstrap_pvalue(start_year, end_year, data, numBootstrapDraws):
    implied_density_post_df = compute_implied_density(start_year, end_year, data)
    for i in range(numBootstrapDraws):
        tmp = compute_implied_density(start_year, end_year, bootstrap_draw(i, data))
        tmp['seed'] = i
        if i == 0:
            bootStrapResults = tmp
        else:
            bootStrapResults = pd.concat((bootStrapResults, tmp), axis=0)
    bootStrapResults.index = range(len(bootStrapResults))
    calc_data = pd.pivot_table(bootStrapResults[['wagebins', 'implied_density_post', 'seed']],
                               index='seed', columns='wagebins', values='implied_density_post').values.T
    sigma = np.cov(calc_data)
    p_value = lf_moment_inequality_test(muhat=-implied_density_post_df['implied_density_post'], Sigmahat=sigma)
    return p_value

try:
    mw_by_year_df = pd.read_parquet('simdata.parquet')
except:
    mw_by_year_df = pd.read_csv('simdata.csv')

table_2007_2015 = compute_implied_density(start_year=2007, end_year=2015, data=mw_by_year_df)

plot_2007_2015 = implied_density_plot(start_year=2007, end_year=2015, data=mw_by_year_df)
print(plot_2007_2015)

pval_2007_2015 = compute_bootstrap_pvalue(start_year=2007, end_year=2015, data=mw_by_year_df, numBootstrapDraws=10)
print('p_value:',pval_2007_2015)
