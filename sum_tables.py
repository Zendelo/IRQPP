import pandas as pd
import dataparser as dp
import numpy as np
from collections import Counter, defaultdict, UserDict
from glob import glob


def sum_ref_table_quant_columns(df_file):
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
    _df = _df.reindex(['Med', 'Top', 'Low'], fill_value=0).astype(int)
    _df = _df.applymap('${}$'.format)
    _df.insert(loc=0, column='group', value='-')
    _df.reset_index(inplace=True)
    _df.insert(0, 'sum', 'Total')
    table = _df.to_latex(escape=False, index=False, index_names=False)
    print(table.replace('Total', ''))


def main():
    df_file = dp.ensure_file('~/ROBUST_Title_queries_full_results_DF.pkl')
    corpus = 'ClueWeb12B'
    table_files = glob(f'{corpus}_*_queries_full_results_DF.pkl')
    for df_file in table_files:
        print(dp.ensure_file(df_file))
        sum_ref_table_quant_columns(df_file)


if __name__ == '__main__':
    main()
