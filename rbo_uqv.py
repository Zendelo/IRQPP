#! /usr/bin/env python

import argparse
from qpputils import dataparser as dp
from RBO import rbo_dict

parser = argparse.ArgumentParser(description='WIG predictor',
                                 usage='Input CE(q|d) scores and queries files',
                                 epilog='Prints the WIG predictor scores')

parser.add_argument('results', metavar='UQV_results_file', help='The raw QL results file')
parser.add_argument('full_queries_file', help='Full UQV queries file')
parser.add_argument('queries_to_predict', help='The queries file that should be predicted')

NUMBER_OF_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]


def queries_to_predict(full_queries_file, predict_queries):
    """This function will return a list of ID's of the queries in the UQV results, the list might be empty
    It matches the queries by text"""
    qdb_f = dp.QueriesTextParser(full_queries_file, 'uqv')
    fqdf = qdb_f.queries_df
    qdb_p = dp.QueriesTextParser(predict_queries)
    pqdf = qdb_p.queries_df
    return fqdf.loc[fqdf['text'].isin(pqdf['text'])]['qid']


def split_prediction_queries(res_df, q_to_p):
    _df = res_df.reset_index()
    return _df.loc[_df['qid'].isin(q_to_p)], _df.loc[~_df['qid'].isin(q_to_p)]


def calc_rbo(pred_res_df, vars_res_df, top):
    _prdf = pred_res_df['docID', 'docScore'].head(top)
    _prdf.reset_index(drop=True, inplace=True)
    _prdf.set_index('docID', inplace=True)
    _prdict = _prdf.to_dict()['docScore']

    _vrdf = vars_res_df['docID', 'docScore'].head(top)
    _vrdf.reset_index(drop=True, inplace=True)
    _vrdf.set_index('docID', inplace=True)
    _vrdict = _vrdf.to_dict()['docScore']

    rbo_dict(_prdf, _vrdf)


def main(args):
    results_file = args.results
    predict_queries_file = args.queries_to_predict
    full_queries_file = args.full_queries_file

    results_obj = dp.ResultsReader(results_file, 'trec')
    res_df = results_obj.data_df
    q2p = queries_to_predict(full_queries_file, predict_queries_file)
    pred_res_df, vars_res_df = split_prediction_queries(res_df, q2p)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
