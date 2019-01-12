import pandas as pd
import dataparser as dp
import numpy as np
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
PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']
PRE_RET_PREDICTORS = [s.upper() for s in PRE_RET_PREDICTORS]
PREDICTORS = PRE_RET_PREDICTORS + POST_PREDICTORS + UEF_PREDICTORS
MACROS_DICT = {'CLARITY': '\\clarity', 'NQC': '\\nqc', 'WIG': '\\wig', 'SMV': '\\smv', 'RSD': '\\rsd', 'QF': '\\qf',
               'UEF': '\\UEF', 'PRERET/MAXIDF': '\\maxIDF', 'PRERET/AVGIDF': '\\avgIDF',
               'PRERET/MAXVARTFIDF': '\\maxVarTFIDF', 'PRERET/AVGVARTFIDF': '\\avgVarTFIDF',
               'PRERET/MAXSCQTFIDF': '\\maxSCQ', 'PRERET/AVGSCQTFIDF': '\\avgSCQ', 'UEF/CLARITY': '\\uef{\\clarity}',
               'UEF/NQC': '\\uef{\\nqc}', 'UEF/SMV': '\\uef{\\smv}', 'UEF/QF': '\\uef{\\qf}', 'UEF/WIG': '\\uef{\\wig}'}


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


def print_main_table(corpus):
    table_files = glob(f'/home/olegzendel/{corpus}_Title_queries_full_results_DF.pkl')
    rb_df_file = dp.ensure_file(f'~/ROBUST_Title_queries_full_results_DF.pkl')
    cw_df_file = dp.ensure_file(f'~/ClueWeb12B_Title_queries_full_results_DF.pkl')
    rb_df = pd.read_pickle(rb_df_file)
    cw_df = pd.read_pickle(cw_df_file)
    rb_df = rb_df.loc[rb_df['Quantile'] == 'All'].set_index('Predictor').drop('Quantile', axis=1)
    cw_df = cw_df.loc[cw_df['Quantile'] == 'All'].set_index('Predictor').drop('Quantile', axis=1)
    print(mark_max_per_row(rb_df).applymap('${}$'.format).to_latex(escape=False))
    print(mark_max_per_row(cw_df).applymap('${}$'.format).to_latex(escape=False))
    # print(rb_df.max(1))
    # print(rb_df)
    # print(cw_df.max(1))
    # print(cw_df)


def mark_bold_row_und_col(_df):
    col_indices = _df.idxmax(0)
    row_indices = _df.idxmax(1)
    df = _df.applymap(float_to_str)
    for i in zip(row_indices.index, row_indices.values):
        df.loc[i] = float_to_bold(df.loc[i])
    for i in zip(col_indices.values, col_indices.index):
        df.loc[i] = float_to_underline(df.loc[i])
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


def main(args):
    corpus = args.corpus
    oracle = args.oracle
    table_type = args.table

    table_type = 'main'
    corpus = 'ROBUST'

    if oracle:
        kind = 'oracle'
    else:
        kind = 'full'
    if table_type == 'main':
        print_main_table(corpus)
        # load_ref_tables_and_sum(corpus, kind)
    else:
        print_single_table()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
