import json

import pandas as pd
import dataparser as dp
import numpy as np
from scipy import stats
from collections import Counter, defaultdict, UserDict
from glob import glob
import argparse

"""This script has a single purpose, add a Total block to the UQV reference tables"""

parser = argparse.ArgumentParser(description='add a Total block to the UQV reference tables',
                                 usage='',
                                 epilog='')

parser.add_argument('-c', '--corpus', type=str, default=None, help='corpus to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--oracle', action='store_true')
parser.add_argument('-t', '--table', choices=['main', 'single'])

POST_PREDICTORS = ['CLARITY', 'NQC', 'WIG', 'SMV', 'RSD', 'QF']
UEF_PREDICTORS = ['UEF/{}'.format(p) for p in POST_PREDICTORS]
UEF_PREDICTORS.remove('UEF/RSD')
UEF_PREDICTORS.remove('UEF/SMV')
PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']
PRE_RET_PREDICTORS = [s.upper() for s in PRE_RET_PREDICTORS]
PREDICTORS = PRE_RET_PREDICTORS + POST_PREDICTORS + UEF_PREDICTORS
MACROS_DICT = {'CLARITY': '\\clarity', 'NQC': '\\nqc', 'WIG': '\\wig', 'SMV': '\\smv', 'RSD': '\\rsd', 'QF': '\\qf',
               'UEF': '\\UEF', 'PRERET/MAXIDF': '\\maxIDF', 'PRERET/AVGIDF': '\\avgIDF',
               'PRERET/MAXVARTFIDF': '\\maxVarTFIDF', 'PRERET/AVGVARTFIDF': '\\avgVarTFIDF',
               'PRERET/MAXSCQTFIDF': '\\maxSCQ', 'PRERET/AVGSCQTFIDF': '\\avgSCQ', 'UEF/CLARITY': '\\uef{\\clarity}',
               'UEF/NQC': '\\uef{\\nqc}', 'UEF/SMV': '\\uef{\\smv}', 'UEF/QF': '\\uef{\\qf}', 'UEF/WIG': '\\uef{\\wig}',
               'RBO': '\\RBOTab', 'Uniform': '\onlyPredTab', 'TopDocs': '\\OverlapTab',
               'SimilarityOnly': '\\onlySimTab', 'Title': '\\titleQuery', 'RBO-F': '\\RBOFuseTab',
               'Jaccard': '\\JacTab', 'GEO': '\\GeoTab'}

PREDICTORS_PATH = {'PRERET/AVGIDF': 'preret/AvgIDF', 'PRERET/AVGSCQTFIDF': 'preret/AvgSCQTFIDF',
                   'PRERET/AVGVARTFIDF': 'preret/AvgVarTFIDF',
                   'PRERET/MAXIDF': 'preret/MaxIDF', 'PRERET/MAXSCQTFIDF': 'preret/MaxSCQTFIDF',
                   'PRERET/MAXVARTFIDF': 'preret/MaxVarTFIDF', 'TopDocs': 'sim', 'Uniform': 'uni', 'Jaccard': 'jac',
                   'RBO-F': 'rbof'}


def print_ref_table_from_pkl(df_file):
    df = pd.read_pickle(df_file)

    for _predictor, _df in df.groupby('Predictor'):
        print(_predictor)
        _df = _df.set_index(['Predictor', 'Quantile']).applymap('${}$'.format)
        print(_df.reset_index())
    exit()
    quants_df = df.loc[df['Quantile'] != 'All']
    quants_df = quants_df.set_index(['Predictor', 'Quantile']).replace('-', np.nan).astype(float)

    counters_per_column = defaultdict(Counter)
    for predictor, df in quants_df.groupby(level=0):
        for col in df.columns[1:]:
            max_quant = df[col].idxmax()
            if max_quant is np.nan:
                continue
            counters_per_column[col][max_quant[1]] += 1

    _df = pd.DataFrame.from_dict(counters_per_column, dtype=int).fillna(0)
    _df = _df.reindex(['Med', 'Top', 'Low', 'Low-0'], fill_value=0).astype(int)
    _df = _df.applymap('${}$'.format)
    _df.insert(loc=0, column='group', value='-')
    _df.reset_index(inplace=True)
    _df.insert(0, 'sum', 'Total')
    table = _df.to_latex(escape=False, index=False, index_names=False)
    print(table.replace('Total', ''))


def load_ref_tables_and_sum(corpus, kind):
    table_files = glob(f'/home/olegzendel/{corpus}_*_queries_{kind}_results_DF.pkl')
    assert len(table_files) > 0, f'{corpus}_*_queries_full_results_DF.pkl, are missing'
    for df_file in sorted(table_files):
        file_name = dp.ensure_file(df_file)
        queries_group = file_name.split('_')[1]
        print(f'Table from {queries_group} queries: ')
        sum_ref_table_quant_columns(df_file, queries_group)


def sum_ref_table_quant_columns(df_file, queries_group):
    df = pd.read_pickle(df_file)

    quants_df = df.loc[df['Quantile'] != 'All']
    quants_df = quants_df.set_index(['Predictor', 'Quantile']).replace('-', np.nan).astype(float)

    counters_per_column = defaultdict(Counter)
    for predictor, df in quants_df.groupby(level=0):
        for col in df.columns[1:]:
            max_quant = df[col].idxmax()
            if max_quant is np.nan:
                continue
            counters_per_column[col][max_quant[1]] += 1

    _df = pd.DataFrame.from_dict(counters_per_column, dtype=int).fillna(0)
    _df = _df.reindex(['Med', 'Top', 'Low', 'Low-0'], fill_value=0).astype(int)
    _df = _df.applymap('${}$'.format)
    _df.insert(loc=0, column='group', value='-')
    _df.reset_index(inplace=True)
    _df.insert(0, 'sum', 'Total')
    table = _df.to_latex(escape=False, index=False, index_names=False)
    print(table.replace('Total', '').replace('index', queries_group))


def compare_main_tables():
    rb_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    test_rb_df_file = dp.ensure_file(f'~/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    cw_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    test_cw_df_file = dp.ensure_file(f'~/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    rb_df = pd.read_pickle(rb_df_file)
    test_rb_df = pd.read_pickle(test_rb_df_file)
    cw_df = pd.read_pickle(cw_df_file)
    test_cw_df = pd.read_pickle(test_cw_df_file)

    rb_df = rb_df.loc[rb_df['Quantile'] == 'All'].set_index('Predictor').drop(['Quantile', 'Jaccard', 'GEO', 'RBO-F'],
                                                                              axis=1)
    test_rb_df = test_rb_df.loc[test_rb_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)
    cw_df = cw_df.loc[cw_df['Quantile'] == 'All'].set_index('Predictor').drop(['Quantile', 'Jaccard', 'GEO', 'RBO-F'],
                                                                              axis=1)
    test_cw_df = test_cw_df.loc[test_cw_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)

    print('CW')
    new = test_cw_df.replace('-', 0).applymap(float)
    cur = cw_df.replace('-', 0).applymap(float)
    print((new - cur).applymap(lambda x: True if x < 0 else False))
    print(new - cur)

    print('ROBUST')
    new = test_rb_df.replace('-', 0).applymap(float)
    cur = rb_df.replace('-', 0).applymap(float)
    print((new - cur).applymap(lambda x: True if x < 0 else False))
    print((new - cur))


def _main_table_w_kendall_improvement():
    rb_p_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    rb_k_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_kendall_queries_full_results_DF.pkl')
    cw_p_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    cw_k_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_kendall_queries_full_results_DF.pkl')
    rb_p_df = pd.read_pickle(rb_p_df_file)
    rb_k_df = pd.read_pickle(rb_k_df_file)
    cw_p_df = pd.read_pickle(cw_p_df_file)
    cw_k_df = pd.read_pickle(cw_k_df_file)

    rb_p_df = rb_p_df.loc[rb_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)
    rb_k_df = rb_k_df.loc[rb_k_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)
    cw_p_df = cw_p_df.loc[cw_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)
    cw_k_df = cw_k_df.loc[cw_k_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F'], axis=1)

    rb_p_imp = rb_p_df.max(1).sub(rb_p_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None)
    rb_p_imp = rb_p_imp.div(rb_p_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None) * 100

    rb_k_imp = rb_k_df.max(1).sub(rb_k_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None)
    rb_k_imp = rb_k_imp.div(rb_k_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None) * 100

    cw_p_imp = cw_p_df.max(1).sub(cw_p_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None)
    cw_p_imp = cw_p_imp.div(cw_p_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None) * 100

    cw_k_imp = cw_k_df.max(1).sub(cw_k_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None)
    cw_k_imp = cw_k_imp.div(cw_k_df['Title'].replace('-', np.nan).astype(float, errors='ignore'), fill_value=None) * 100

    rb_p_imp = rb_p_imp.apply('{:.1f}\\%'.format)
    rb_k_imp = rb_k_imp.apply('{:.1f}\\%'.format)
    cw_p_imp = cw_p_imp.apply('{:.1f}\\%'.format)
    cw_k_imp = cw_k_imp.apply('{:.1f}\\%'.format)

    _rb_p_df = mark_max_per_row(rb_p_df).assign(improvement=rb_p_imp)
    _rb_k_df = mark_max_per_row(rb_k_df).assign(improvement=rb_k_imp)
    _cw_p_df = mark_max_per_row(cw_p_df).assign(improvement=cw_p_imp)
    _cw_k_df = mark_max_per_row(cw_k_df).assign(improvement=cw_k_imp)

    rb_df = pd.concat({'P $\\rho$': _rb_p_df, 'K $\\tau$': _rb_k_df}, axis=1).reindex(
        ['P $\\rho$', 'K $\\tau$'], axis=1, level=0).swaplevel(axis=1)
    cw_df = pd.concat({'P $\\rho$': _cw_p_df, 'K $\\tau$': _cw_k_df}, axis=1).reindex(
        ['P $\\rho$', 'K $\\tau$'], axis=1, level=0).swaplevel(axis=1)

    df = pd.concat({'\\robust': rb_df, '\\clueTwelve': cw_df}, axis=1).reindex(['\\robust', '\\clueTwelve'], axis=1,
                                                                               level=0)

    df = df.reindex(PREDICTORS + ['SimilarityOnly'])
    df = df.applymap(lambda x: f'${x}$')
    df = df.reindex(['Title', 'Uniform', 'TopDocs', 'RBO', 'improvement'], axis=1, level=1)
    df = df.rename(MACROS_DICT, axis=1, level=1).rename(MACROS_DICT).rename({'\\titleQuery': '\\baseline'}, axis=1,
                                                                            level=1)
    # print(df.to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    print(df['\\robust'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    print(df['\\clueTwelve'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))


def t_test(sr_a, sr_b, alpha=0.05):
    """Two Tailed paired samples t test"""
    pval = stats.ttest_rel(sr_a, sr_b).pvalue[0]
    return pval < alpha


def check_significance_to_base(corpus, predictor, similarity, quant):
    baseline_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/{predictor}/evaluation/')
    baseline_file = dp.ensure_file(f'{baseline_dir}/simple_results_vector_for_2_folds_30_repetitions_title.json')
    with open(baseline_file) as json_data:
        data = json.load(json_data)
    baseline_sr = pd.DataFrame.from_dict(data, orient='index', columns=['correlation'], dtype=float)

    candidate_dir = dp.ensure_dir(
        f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/title/{quant}_vars/general'
        f'/{similarity}/{predictor}/evaluation/')

    candidate_file = dp.ensure_file(f'{candidate_dir}/simple_results_vector_for_2_folds_30_repetitions_title.json')
    with open(candidate_file) as json_data:
        data = json.load(json_data)
    candidate_sr = pd.DataFrame.from_dict(data, orient='index', columns=['correlation'], dtype=float)
    return t_test(baseline_sr, candidate_sr)


def check_significance_to_quants(corpus, predictor, similarity, quant_a, quant_b, alpha=0.05):
    _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/title/'
    baseline_dir = dp.ensure_dir(f'{_base_dir}/{quant_a}_vars/general/{similarity}/{predictor}/evaluation/')
    baseline_file = dp.ensure_file(f'{baseline_dir}/simple_results_vector_for_2_folds_30_repetitions_title.json')
    with open(baseline_file) as json_data:
        data = json.load(json_data)
    baseline_sr = pd.DataFrame.from_dict(data, orient='index', columns=['correlation'], dtype=float)

    candidate_dir = dp.ensure_dir(f'{_base_dir}/{quant_b}_vars/general/{similarity}/{predictor}/evaluation/')

    candidate_file = dp.ensure_file(f'{candidate_dir}/simple_results_vector_for_2_folds_30_repetitions_title.json')
    with open(candidate_file) as json_data:
        data = json.load(json_data)
    candidate_sr = pd.DataFrame.from_dict(data, orient='index', columns=['correlation'], dtype=float)
    return t_test(baseline_sr, candidate_sr, alpha)


def mark_quant_significance(_df: pd.DataFrame, corpus, q_a, q_b):
    indices = defaultdict(str)
    if q_a == 'all' or q_b == 'all':
        sig_sign = '\\statSigAllQuant'
    else:
        sig_sign = '\\statSigVarQuant'
    predictors = _df.index.unique()
    similarities = _df.columns.unique()
    for pred in predictors:
        if pred == 'SimilarityOnly':
            continue
        for simi in similarities:
            if simi == 'Title':
                continue
            predictor = PREDICTORS_PATH.get(pred, pred.lower())
            similarity = PREDICTORS_PATH.get(simi, simi.lower())
            if check_significance_to_quants(corpus, predictor, similarity, q_a, q_b, alpha=0.05 / 3):
                indices[(pred, simi)] = f'{sig_sign}'
                # df.loc[pred, simi] = f'{df.loc[pred, simi]}^{sig_sign}'
    return indices


def mark_significance_to_base(_df: pd.DataFrame, corpus, quant):
    try:
        quantile_sr = _df['Quantile']
        _df = _df.drop('Quantile', axis=1)
        quantile = True
    except KeyError:
        quantile = False
    df = _df.applymap(float_to_str)
    for pred in df.index:
        if pred == 'SimilarityOnly':
            continue
        for simi in df.columns:
            if simi == 'Title':
                continue
            predictor = PREDICTORS_PATH.get(pred, pred.lower())
            similarity = PREDICTORS_PATH.get(simi, simi.lower())
            if check_significance_to_base(corpus, predictor, similarity, quant):
                df.loc[pred, simi] = f'{df.loc[pred, simi]}\\sig'
    if quantile:
        df.insert(loc=0, column='Quantile', value=quantile_sr)
    return df


def mark_significance_bold_prow(_df: pd.DataFrame, corpus, quant):
    try:
        quantile_sr = _df['Quantile']
        _df = _df.drop('Quantile', axis=1)
        quantile = True
    except KeyError:
        quantile = False
    row_indices = _df.replace('-', np.nan).applymap(float).idxmax(1)
    df = _df.applymap(float_to_str)
    for pred in df.index:
        if pred == 'SimilarityOnly':
            continue
        for simi in df.columns:
            if simi == 'Title':
                continue
            predictor = PREDICTORS_PATH.get(pred, pred.lower())
            similarity = PREDICTORS_PATH.get(simi, simi.lower())
            if check_significance_to_base(corpus, predictor, similarity, quant):
                df.loc[pred, simi] = f'{df.loc[pred, simi]}^{{*}}'
    for i in zip(row_indices.index, row_indices.values):
        df.loc[i] = float_to_bold(df.loc[i])
    try:
        df.insert(loc=0, column='Quantile', value=quantile_sr)
    except NameError:
        pass
    return df


def mark_significance_under_pcol(_df: pd.DataFrame, corpus, quant):
    try:
        quantile_sr = _df['Quantile']
        _df = _df.drop('Quantile', axis=1)
        quantile = True
    except KeyError:
        quantile = False
    col_indices = _df.replace('-', np.nan).applymap(float).idxmax(0)
    df = _df.applymap(float_to_str)
    for pred in df.index:
        if pred == 'SimilarityOnly':
            continue
        for simi in df.columns:
            if simi == 'Title':
                continue
            predictor = PREDICTORS_PATH.get(pred, pred.lower())
            similarity = PREDICTORS_PATH.get(simi, simi.lower())
            if check_significance_to_base(corpus, predictor, similarity, quant):
                df.loc[pred, simi] = f'{df.loc[pred, simi]}^{{*}}'
    for i in zip(col_indices.values, col_indices.index):
        df.loc[i] = float_to_underline(df.loc[i])
    try:
        df.insert(loc=0, column='Quantile', value=quantile_sr)
    except NameError:
        pass
    return df


def print_main_table():
    rb_p_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    cw_p_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    rb_p_df = pd.read_pickle(rb_p_df_file)
    cw_p_df = pd.read_pickle(cw_p_df_file)

    rb_p_df = rb_p_df.loc[rb_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)
    cw_p_df = cw_p_df.loc[cw_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)

    _rb_p_df = mark_significance_under_pcol(rb_p_df, 'ROBUST', 'all')

    _cw_p_df = mark_significance_under_pcol(cw_p_df, 'ClueWeb12B', 'all')

    df = pd.concat({'\\robust': _rb_p_df, '\\clueTwelve': _cw_p_df}, axis=1) \
        .reindex(['\\robust', '\\clueTwelve'], axis=1, level=0)

    df = df.reindex(PREDICTORS)
    df = df.applymap(lambda x: f'${x}$')
    df = df.reindex(['Title', 'TopDocs', 'RBO'], axis=1, level=1)
    df = df.rename(MACROS_DICT, axis=1, level=1).rename(MACROS_DICT).rename({'\\titleQuery': '\\baseline'}, axis=1,
                                                                            level=1)
    print(df.to_latex(escape=False, multicolumn_format='c'))
    # print(df['\\robust'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    # print(df['\\clueTwelve'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))


def print_main_table_w_kendall():
    rb_p_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    rb_k_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_kendall_queries_full_results_DF.pkl')
    cw_p_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    cw_k_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_kendall_queries_full_results_DF.pkl')
    rb_p_df = pd.read_pickle(rb_p_df_file)
    rb_k_df = pd.read_pickle(rb_k_df_file)
    cw_p_df = pd.read_pickle(cw_p_df_file)
    cw_k_df = pd.read_pickle(cw_k_df_file)

    rb_p_df = rb_p_df.loc[rb_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)
    rb_k_df = rb_k_df.loc[rb_k_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)
    cw_p_df = cw_p_df.loc[cw_p_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)
    cw_k_df = cw_k_df.loc[cw_k_df['Quantile'] == 'All'].set_index('Predictor').drop(
        ['Quantile', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)

    # rb_df = rb_df.loc[rb_df['Quantile'] == 'All'].set_index('Predictor').drop(
    #     ['Quantile', 'Uniform', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)
    # cw_df = cw_df.loc[cw_df['Quantile'] == 'All'].set_index('Predictor').drop(
    #     ['Quantile', 'Uniform', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop('SimilarityOnly', axis=0)

    # _rb_p_df = mark_significance(rb_p_df, 'ROBUST', 'all')
    # _rb_k_df = mark_significance(rb_k_df, 'ROBUST', 'all')
    #
    # _cw_p_df = mark_significance(cw_p_df, 'ClueWeb12B', 'all')
    # _cw_k_df = mark_significance(cw_k_df, 'ClueWeb12B', 'all')

    _rb_p_df = mark_significance_bold_prow(rb_p_df, 'ROBUST', 'all')
    _rb_k_df = mark_significance_bold_prow(rb_k_df, 'ROBUST', 'all')

    _cw_p_df = mark_significance_bold_prow(cw_p_df, 'ClueWeb12B', 'all')
    _cw_k_df = mark_significance_bold_prow(cw_k_df, 'ClueWeb12B', 'all')

    _rb_df = pd.concat({'$P$ $r$': _rb_p_df, '$K$ $\\tau$': _rb_k_df}, axis=1) \
        .reindex(['$P$ $r$', '$K$ $\\tau$'], axis=1, level=0).swaplevel(axis=1)
    _cw_df = pd.concat({'$P$ $r$': _cw_p_df, '$K$ $\\tau$': _cw_k_df}, axis=1) \
        .reindex(['$P$ $r$', '$K$ $\\tau$'], axis=1, level=0).swaplevel(axis=1)

    df = pd.concat({'\\robust': _rb_df, '\\clueTwelve': _cw_df}, axis=1) \
        .reindex(['\\robust', '\\clueTwelve'], axis=1, level=0)

    df = df.reindex(PREDICTORS)
    df = df.applymap(lambda x: f'${x}$')
    df = df.reindex(['Title', 'TopDocs', 'RBO'], axis=1, level=1)
    df = df.rename(MACROS_DICT, axis=1, level=1).rename(MACROS_DICT).rename({'\\titleQuery': '\\baseline'}, axis=1,
                                                                            level=1)
    print(df.to_latex(escape=False, multicolumn_format='c'))
    # print(df['\\robust'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    # print(df['\\clueTwelve'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))


def mark_bold_row_und_col(_df):
    col_indices = _df.idxmax(0)
    row_indices = _df.idxmax(1)
    df = _df.applymap(float_to_str)
    for i in zip(row_indices.index, row_indices.values):
        df.loc[i] = float_to_bold(df.loc[i])
    for i in zip(col_indices.values, col_indices.index):
        df.loc[i] = float_to_underline(df.loc[i])
    return df


def mark_max_per_df(_df):
    col_id = _df.max(0).idxmax()
    row_id = _df.max(1).idxmax()
    df = _df.applymap(float_to_str)
    df.loc[row_id, col_id] = float_to_bold(df.loc[row_id, col_id])
    return df


def mark_max_per_row(df):
    """marking the maximum in a Series or DataFrame"""
    _df = df.apply(lambda x: [float_to_bold(i) if i == max(x) else float_to_str(i) for i in x], axis=1,
                   result_type='expand')
    _df.columns = df.columns
    return _df


def mark_max_per_column(df):
    """marking the maximum in a Series or DataFrame"""
    _df = df.apply(lambda x: [float_to_underline(i) if i == max(x) else float_to_str(i) for i in x], axis=0,
                   result_type='expand')
    _df.columns = df.columns
    return _df


def float_to_underline(x):
    return f'\\underline{{{float_to_str(x)}}}'


def float_to_bold(x):
    return f'\\mathbf{{{float_to_str(x)}}}'


def float_to_str(x):
    if type(x) is float:
        return f'{x:.3f}'.lstrip('0')
    else:
        try:
            x.isnumeric()
            return x.lstrip('0')
        except AttributeError:
            return x


def print_single_table():
    rb_df_file = dp.ensure_file(f'~/ROBUST_single_queries_full_results_DF.pkl')
    cw_df_file = dp.ensure_file(f'~/ClueWeb12B_single_queries_full_results_DF.pkl')
    rb_df = pd.read_pickle(rb_df_file)
    cw_df = pd.read_pickle(cw_df_file)
    rb_df = rb_df.drop('kendall', axis=1).rename({'pearson': '\\robust'}, axis=1)
    rb_df = rb_df.pivot(index='Predictor', columns='Function')
    cw_df = cw_df.drop('kendall', axis=1).rename({'pearson': '\\clueTwelve'}, axis=1)
    cw_df = cw_df.pivot(index='Predictor', columns='Function')

    _cw_df = mark_bold_row_und_col(cw_df)
    _rb_df = mark_bold_row_und_col(rb_df)

    df = pd.merge(right=_cw_df, left=_rb_df, on='Predictor')

    df = df.reindex(PREDICTORS)
    df = df.applymap(lambda x: f'${x}$')
    df = df.reindex(['title', 'top', 'medh', 'low'], axis=1, level=1).rename(
        {'title': '\\titleQuery', 'top': '\\maxAP', 'medh': '\\medAP', 'low': '\\minAP'}, axis=1).rename(MACROS_DICT)
    print(df.to_latex(escape=False))


def append_sign_df(df, indices_dict, indx_quant):
    for (row_index, col_index), sign in indices_dict.items():
        df.loc[(row_index, indx_quant), col_index] = f'{df.loc[(row_index, indx_quant), col_index]}{sign}'


def print_quant_table():
    rb_p_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    cw_p_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    rb_df = pd.read_pickle(rb_p_df_file)
    cw_df = pd.read_pickle(cw_p_df_file)

    rb_df = rb_df.set_index(['Predictor']).drop(['Uniform', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop(
        'SimilarityOnly', axis=0)
    rb_df = rb_df.loc[['PRERET/MAXIDF', 'PRERET/AVGSCQTFIDF', 'WIG', 'UEF/CLARITY']]

    cw_df = cw_df.set_index(['Predictor']).drop(['Uniform', 'Jaccard', 'GEO', 'RBO-F', 'TopDocs'], axis=1).drop(
        'SimilarityOnly', axis=0)
    cw_df = cw_df.loc[['PRERET/MAXIDF', 'PRERET/AVGSCQTFIDF', 'WIG', 'UEF/CLARITY']]

    _all_rb_df = mark_significance_to_base(rb_df.loc[rb_df['Quantile'] == 'All'], 'ROBUST', 'all')
    _low_rb_df = mark_significance_to_base(rb_df.loc[rb_df['Quantile'] == 'Low-0'], 'ROBUST', 'low')
    _high_rb_df = mark_significance_to_base(rb_df.loc[rb_df['Quantile'] == 'High'], 'ROBUST', 'high')

    _all_cw_df = mark_significance_to_base(cw_df.loc[cw_df['Quantile'] == 'All'], 'ROBUST', 'all')
    _low_cw_df = mark_significance_to_base(cw_df.loc[cw_df['Quantile'] == 'Low-0'], 'ROBUST', 'low')
    _high_cw_df = mark_significance_to_base(cw_df.loc[cw_df['Quantile'] == 'High'], 'ROBUST', 'high')

    rb_all_lo_sigs = mark_quant_significance(rb_df.drop('Quantile', axis=1), 'ROBUST', 'all', 'low')
    rb_all_hi_sigs = mark_quant_significance(rb_df.drop('Quantile', axis=1), 'ROBUST', 'all', 'high')
    rb_lo_hi_sigs = mark_quant_significance(rb_df.drop('Quantile', axis=1), 'ROBUST', 'low', 'high')

    cw_all_lo_sigs = mark_quant_significance(cw_df.drop('Quantile', axis=1), 'ROBUST', 'all', 'low')
    cw_all_hi_sigs = mark_quant_significance(cw_df.drop('Quantile', axis=1), 'ROBUST', 'all', 'high')
    cw_lo_hi_sigs = mark_quant_significance(cw_df.drop('Quantile', axis=1), 'ROBUST', 'low', 'high')

    _rb_df = pd.concat([_all_rb_df, _high_rb_df, _low_rb_df], axis=0).reset_index().set_index(['Predictor', 'Quantile'])
    _cw_df = pd.concat([_all_cw_df, _high_cw_df, _low_cw_df], axis=0).reset_index().set_index(['Predictor', 'Quantile'])

    append_sign_df(_rb_df, rb_all_lo_sigs, 'Low-0')
    append_sign_df(_rb_df, rb_all_hi_sigs, 'High')
    append_sign_df(_rb_df, rb_lo_hi_sigs, 'High')

    append_sign_df(_cw_df, cw_all_lo_sigs, 'Low-0')
    append_sign_df(_cw_df, cw_all_hi_sigs, 'High')
    append_sign_df(_cw_df, cw_lo_hi_sigs, 'High')

    df = pd.concat({'\\robust': _rb_df, '\\clueTwelve': _cw_df}, axis=1).reindex(['\\robust', '\\clueTwelve'], axis=1,
                                                                                 level=0)

    df = df.applymap(lambda x: f'${x}$')
    df = df.reindex(['Title', 'TopDocs', 'RBO'], axis=1, level=1)
    df = df.rename(MACROS_DICT, axis=1, level=1).rename(MACROS_DICT).rename({'\\titleQuery': '\\baseline'}, axis=1,
                                                                            level=1)

    for predictor, _df in df.reset_index().groupby('Predictor'):
        table = _df.to_latex(escape=False, index=False, header=False, index_names=False)
        table = table.replace(predictor, '{}')
        table = table.replace('\\toprule\n', '')
        table = table.replace('\\begin{tabular}{llllll}', f'\\multirow{{3}}{{*}}{{{predictor}}}')
        table = table.replace('\\bottomrule', '\\midrule')
        table = table.replace('Low-0', ' Low')
        table = table.replace('\end{tabular}\n', '')

        print(table, end='')

    # print(table)
    # print(df['\\robust'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    # print(df['\\clueTwelve'].to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))


def print_quality_table():
    rb_title_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    rb_min_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_MinAP_pearson_queries_full_results_DF.pkl')
    rb_max_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_MaxAP_pearson_queries_full_results_DF.pkl')
    rb_med_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_MedHiAP_pearson_queries_full_results_DF.pkl')

    cw_title_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    cw_min_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_MinAP_pearson_queries_full_results_DF.pkl')
    cw_max_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_MaxAP_pearson_queries_full_results_DF.pkl')
    cw_med_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_MedHiAP_pearson_queries_full_results_DF.pkl')

    rb_df = pd.read_pickle(rb_p_df_file)
    cw_df = pd.read_pickle(cw_p_df_file)


def print_similarities_table():
    rb_df_file = dp.ensure_file(f'~/cur_tables/ROBUST_Title_pearson_queries_full_results_DF.pkl')
    cw_df_file = dp.ensure_file(f'~/cur_tables/ClueWeb12B_Title_pearson_queries_full_results_DF.pkl')
    rb_df = pd.read_pickle(rb_df_file)
    cw_df = pd.read_pickle(cw_df_file)

    rb_df = rb_df.loc[rb_df['Quantile'] == 'All'].set_index('Predictor').loc[
        ['PRERET/MAXIDF', 'PRERET/AVGSCQTFIDF', 'WIG', 'UEF/CLARITY', 'SimilarityOnly']].drop('Quantile', axis=1)
    cw_df = cw_df.loc[cw_df['Quantile'] == 'All'].set_index('Predictor').loc[
        ['PRERET/MAXIDF', 'PRERET/AVGSCQTFIDF', 'WIG', 'UEF/CLARITY', 'SimilarityOnly']].drop('Quantile', axis=1)

    _rb_df = mark_significance_bold_prow(rb_df, 'ROBUST', 'all')
    _cw_df = mark_significance_bold_prow(cw_df, 'ROBUST', 'all')

    df = pd.concat({'\\robust': _rb_df, '\\clueTwelve': _cw_df}, axis=1).reindex(['\\robust', '\\clueTwelve'], axis=1,
                                                                                 level=0)

    df = df.reindex(['PRERET/MAXIDF', 'PRERET/AVGSCQTFIDF', 'WIG', 'UEF/CLARITY', 'SimilarityOnly'], axis=0)
    df = df.reindex(['Title', 'Jaccard', 'TopDocs', 'RBO', 'RBO-F', 'GEO', 'Uniform'], axis=1, level=1)
    df = df.applymap(lambda x: f'${x}$')
    df = df.rename(MACROS_DICT, axis=1, level=1).rename(MACROS_DICT).rename({'\\titleQuery': '\\baseline'}, axis=1,
                                                                            level=1)
    # print(df.to_latex(escape=False, multicolumn_format='c'))
    print(df['\\robust'].T.to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))
    print(df['\\clueTwelve'].T.to_latex(escape=False, multicolumn_format='c').replace('nan', '-'))


def main(args):
    corpus = args.corpus
    oracle = args.oracle
    table_type = args.table

    # table_type = 'main'
    table_type = 'quant'
    # table_type = 'qual'
    # table_type = 'similar'
    corpus = 'ROBUST'
    predictor = 'wig'
    # compare_main_tables()
    # exit()

    # check_significance(corpus, predictor)

    if oracle:
        kind = 'oracle'
    else:
        kind = 'full'
    if table_type == 'main':
        print_main_table()
        # load_ref_tables_and_sum(corpus, kind)
    elif table_type == 'single':
        print_single_table()
    elif table_type == 'quant':
        print_quant_table()
    elif table_type == 'qual':
        print_quality_table()
    elif table_type == 'similar':
        print_similarities_table()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
