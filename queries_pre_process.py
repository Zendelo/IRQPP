import argparse
from statistics import median_high, median_low

import pandas as pd

import dataparser as dt

# TODO: add logging and qrels file generation for UQV

parser = argparse.ArgumentParser(description='Script for query files pre-processing',
                                 epilog='Use this script with Caution')

parser.add_argument('-t', '--queries', default=None, metavar='queries.txt', help='path to UQV queries txt file')
parser.add_argument('--remove', default=None, metavar='queries.txt',
                    help='path to queries txt file that will be removed from the final file NON UQV ONLY')
parser.add_argument('--top', action='store_true', help='Return only the best performing queries of each topic')
parser.add_argument('--low', action='store_true', help='Return only the worst performing queries of each topic')
parser.add_argument('--medh', action='store_true', help='Return only the high median performing queries of each topic')
parser.add_argument('--medl', action='store_true', help='Return only the low median performing queries of each topic')
parser.add_argument('--quant', default=None, choices=['low', 'top', 'med'],
                    help='Return a quantile of the variants for each topic')
parser.add_argument('--ap', default=None, metavar='QLmap1000', help='path to queries AP results file')


def add_original_queries():
    pass


def convert_vid_to_qid(df: pd.DataFrame):
    _df = df.set_index('qid')
    _df.rename(index=lambda x: f'{x.split("-")[0]}', inplace=True)
    return _df.reset_index()


def filter_quant_variants(qdf: pd.DataFrame, apdb: dt.ResultsReader, q):
    """This function returns a df with QID: TEXT of the queries inside a quantile"""
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        if 0 in q:
            # For the low quantile, 0 AP variants are removed
            _df = _df[_df['ap'] > 0]
        q_vals = _df.quantile(q=q)
        _qvars = _df.loc[(_df['ap'] >= q_vals['ap'].min()) & (_df['ap'] <= q_vals['ap'].max())]
        _list.extend(_qvars.index.tolist())
    _res_df = qdf.loc[qdf['qid'].isin(_list)]
    return _res_df


def filter_top_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        top_var = _apdf.loc[q_vars].idxmax()
        _list.append(top_var[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_low_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        # remove 0 ap variants
        _df = _df[_df['ap'] > 0]
        low_var = _df.idxmin()
        _list.append(low_var[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_med_h_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        _med = median_high(_df['ap'])
        med_var = _df.loc[_df['ap'] == _med]
        _list.append(med_var.index[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_med_l_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        _med = median_low(_df['ap'])
        med_var = _df.loc[_df['ap'] == _med]
        _list.append(med_var.index[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def remove_duplicates(qdb: dt.QueriesTextParser):
    _list = []
    for topic, q_vars in qdb.query_vars.items():
        _list.append(qdb.queries_df.loc[qdb.queries_df['qid'].isin(q_vars)].drop_duplicates('text'))
    return pd.concat(_list)


def alternate_remove_duplicates(qdb: dt.QueriesTextParser):
    """Different commands, same result"""
    _dup_list = []
    for topic, q_vars in qdb.query_vars.items():
        _dup_list.extend(qdb.queries_df.loc[qdb.queries_df['qid'].isin(q_vars)].duplicated('text'))
    return qdb.queries_df[~qdb.queries_df['qid'].isin(qdb.queries_df.loc[_dup_list]['qid'])]


def remove_q1_from_q2(qdf_1: pd.DataFrame, qdf_2: pd.DataFrame):
    """This function will remove from qdf_2 the queries that exist in qdf_1 """
    return qdf_2.loc[~qdf_2['text'].isin(qdf_1['text'])]


def save_txt_queries(q_df: pd.DataFrame):
    q_df.to_csv('queries_new.txt', sep=":", header=False, index=False)


def main(args):
    queries_txt_file = args.queries
    queries_to_remove = args.remove
    ap_file = args.ap
    top_queries = args.top
    low_queries = args.low
    medh_queries = args.medh
    medl_queries = args.medl
    quant_variants = args.quant

    # quant_variants = 'low'
    # ap_file = '../../QppUqvProj/Results/ROBUST/test/raw/QLmap1000'
    # queries_txt_file = '../../QppUqvProj/data/ROBUST/queries_ROBUST_UQV_full.txt'

    if queries_txt_file:
        qdb = dt.QueriesTextParser(queries_txt_file, 'uqv')
        queries_df = remove_duplicates(qdb)
        if queries_to_remove:
            qdb_rm = dt.QueriesTextParser(queries_to_remove)
            queries_df = remove_q1_from_q2(qdb_rm.queries_df, queries_df)
        if ap_file:
            apdb = dt.ResultsReader(ap_file, 'ap')
            if top_queries:
                queries_df = filter_top_queries(queries_df, apdb)
            elif low_queries:
                queries_df = filter_low_queries(queries_df, apdb)
            elif medh_queries:
                queries_df = filter_med_h_queries(queries_df, apdb)
            elif medl_queries:
                queries_df = filter_med_l_queries(queries_df, apdb)
            elif quant_variants:
                if quant_variants == 'low':
                    queries_df = filter_quant_variants(queries_df, apdb, [0, 0.3])
                elif quant_variants == 'med':
                    queries_df = filter_quant_variants(queries_df, apdb, [0.33, 0.6])
                elif quant_variants == 'top':
                    queries_df = filter_quant_variants(queries_df, apdb, [0.66, 1])

        # # In order to convert the vid (variants ID) to qid, uncomment next line
        # queries_df = convert_vid_to_qid(queries_df)
        save_txt_queries(queries_df)
        query_xml = dt.QueriesXMLParser(queries_df)
        query_xml.print_queries_xml()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
