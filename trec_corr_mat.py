import pandas as pd
from glob import glob

working_dir = '/home/olegzendel/track_runs_corrected_and_eval'
ap_aggregations_dir = '/home/olegzendel/QppUqvProj/Results/ROBUST/test/aggregated'


def generate_aggregated_ap_files():
    eval_files = glob(f'{working_dir}/eval_inputs/*')
    _list = []
    for file in eval_files:
        _df = pd.read_csv(file, delim_whitespace=True, header=None, names=['measure', 'qid', 'map'])
        _list.append(_df.loc[(_df['measure'] == 'map') & (_df['qid'] != 'all')][['qid', 'map']].set_index('qid'))
    df = pd.concat(_list, axis=1)
    df.mean(1).to_csv(f'{working_dir}/trec-avg', sep=" ", header=False, index=True,
                      float_format='%f')
    df.max(1).to_csv(f'{working_dir}/trec-max', sep=" ", header=False, index=True,
                     float_format='%f')
    df.min(1).to_csv(f'{working_dir}/trec-min', sep=" ", header=False, index=True,
                     float_format='%f')
    df.median(1).to_csv(f'{working_dir}/trec-med', sep=" ", header=False, index=True,
                        float_format='%f')
    df.std(1).to_csv(f'{working_dir}/trec-std', sep=" ", header=False, index=True,
                     float_format='%f')


def df_from_files(files, files_type):
    _list = []
    col_names = []
    for file in files:
        _agg = file.split('-')[-1]
        if _agg == 'sum':
            continue
        col_name = f'{_agg}-{files_type}'
        col_names.append(col_name)
        _list.append(
            pd.read_csv(file, delim_whitespace=True, header=None, names=['qid', col_name],
                        index_col='qid'))
    return pd.concat(_list, axis=1), sorted(col_names)


def add_format(s):
    s = '${:.3f}$'.format(s)
    return s


def calc_correlations_df():
    ap_files = glob(f'{ap_aggregations_dir}/map1000-*')
    trec_ap_files = glob(f'{working_dir}/trec-*')
    ap_df, ap_headers = df_from_files(ap_files, 'AP')
    trec_df, trec_headers = df_from_files(trec_ap_files, 'Trec')
    df = pd.merge(ap_df, trec_df, left_index=True, right_index=True)
    df = df.corr().loc[trec_headers, ap_headers]
    formatters = [add_format] * len(df.columns)
    print(df.to_latex(float_format='%f', formatters=formatters, escape=False))


if __name__ == '__main__':
    # generate_aggregated_ap_files()
    calc_correlations_df()
