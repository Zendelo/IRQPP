#!/usr/bin/env python

import argparse
from collections import defaultdict
from statistics import median_high, median_low

import pandas as pd

from dataparser import ResultsReader

parser = argparse.ArgumentParser(description='UQV single query calculations script',
                                 usage='python3.6 singleUQV.py RAW_AP_FILE RAW_PREDICTIONS_FILE function',
                                 epilog='Calculate new UQV scores')

parser.add_argument('map', metavar='RAW_AP_FILE', help='path to raw ap scores file')
parser.add_argument('predictions', metavar='RAW_PREDICTIONS_FILE', help='path to raw predictions file')
parser.add_argument('-f', type=str, default='all', choices=['max', 'medh', 'medl', 'min'],
                    help='Single pick function')


# TODO: Move the script to work with dataparser Classes
# TODO: Fix this warning:
# See the documentation here:
# https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
#   _df = self.raw_pred.loc[self.selected_queries[func]]
# /home/olegzendel/repos/IRQPP/singleUQV.py:100: FutureWarning:
# Passing list-likes to .loc or [] with any missing label will raise
# KeyError in the future, you can use .reindex() as an alternative.


class Single:
    def __init__(self, raw_map_df: pd.DataFrame, raw_pred_df: pd.DataFrame, pick_func):
        self.raw_map = raw_map_df
        self.raw_pred = raw_pred_df
        # The function to be used to pick the queries (by 'max', 'min', etc... ap score)
        self.func = pick_func
        # qid_var_dict is {qid:[qid-x, qid-y ...] } var_qid_dict is {qid-x:qid} }
        self.qid_var_dict, self.var_qid_dict = self.__generate_qid_cont()
        self.selected_queries = self.__pick_single_queries()

    def __generate_qid_cont(self):
        qid_vars_dict = defaultdict(list)
        vars_qid_dict = defaultdict(str)
        for rawqid in self.raw_map.index:
            qid = rawqid.split('-')[0]
            qid_vars_dict[qid].append(rawqid)
            vars_qid_dict[rawqid] = qid
        return qid_vars_dict, vars_qid_dict

    def __pick_single_queries(self):
        picked_queries = defaultdict(list)
        for qid, qvars in self.qid_var_dict.items():
            _df = self.raw_map.loc[qvars]
            picked_queries['max'].append(_df.idxmax()[0])
            picked_queries['min'].append(_df.idxmin()[0])
            medl = median_low(_df['ap'])
            medh = median_high(_df['ap'])
            picked_queries['medl'].append(_df[_df['ap'] == medl].head(1).index[0])
            picked_queries['medh'].append(_df[_df['ap'] == medh].head(1).index[0])
            picked_queries['qid'].append(qid)
        return picked_queries

    def print_score(self, func):
        _df = self.raw_pred.loc[self.selected_queries[func]]
        _df.insert(0, 'qid', self.selected_queries['qid'])
        print(_df.to_string(index=False, index_names=False, header=False))


def main(args: parser):
    map_file = args.map
    predictions_file = args.predictions
    pick_func = args.f

    ap_scores = ResultsReader(map_file, 'ap')
    prediction_scores = ResultsReader(predictions_file, 'predictions')
    single_scores = Single(ap_scores.data_df, prediction_scores.data_df, pick_func)

    if pick_func == 'all':
        for f in ['max', 'medh', 'medl', 'min']:
            print('\nlist for {}:\n'.format(f))
            single_scores.print_score(f)
    else:
        single_scores.print_score(pick_func)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
