#!/usr/bin/env python

import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='UQV aggregation script',
                                 usage='Create new UQV scores',
                                 epilog='ROBUST version')

parser.add_argument('-a', '--map', default=None, help='path to ap scores file')
parser.add_argument('-p', '--predictions', default=None, help='path to prediction scores file')
parser.add_argument('-f', '--function', default='avg', choices=['max', 'std', 'min', 'avg'], help='Aggregate function')


class DataReader:
    def __init__(self, data_file: str, file_type):
        """
        :param data_file: results res
        :param file_type: 'result' for predictor results res or 'ap' for ap results res
        """
        self.file_type = file_type
        self.data = data_file
        self.__number_of_col = self.__check_number_of_col()
        if self.file_type == 'result':
            assert self.__number_of_col == 2 or self.__number_of_col == 4, 'Wrong File format'
            self.data_df = self.__read_results_data_2() if self.__number_of_col == 2 else self.__read_results_data_4()
        elif self.file_type == 'ap':
            self.data_df = self.__read_ap_data_2()

    def __check_number_of_col(self):
        with open(self.data) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            first_row = next(reader)
            num_cols = len(first_row)
        return int(num_cols)

    def __read_results_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'score'],
                                dtype={'qid': str, 'score': np.float64})
        return data_df

    def __read_ap_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid AP'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'ap'],
                                dtype={'qid': str, 'ap': np.float64})
        return data_df

    def __read_results_data_4(self):
        """Assuming data is a res with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': str, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        data_df = data_df.filter(['qid', 'score'], axis=1)
        return data_df


class Aggregate:
    def __init__(self, data_df: pd.DataFrame, agg_func):
        self.data = data_df
        self.agg_func = agg_func
        self.agg_scores_dict = self._aggregate_scores()
        self.final_score_dict = self._calc_scores()

    def _aggregate_scores(self):
        _agg = defaultdict(list)
        for qid, _score in self.data.itertuples():
            qid = qid.split('-')[0]
            _agg[qid].append(_score)
        return _agg

    def _calc_scores(self):
        _final_scores = defaultdict(float)
        for qid, scores in self.agg_scores_dict.items():
            if self.agg_func.lower() == 'max':
                score = np.max(scores)
            elif self.agg_func.lower() == 'min':
                score = np.min(scores)
            elif self.agg_func.lower() == 'avg':
                score = np.mean(scores)
            elif self.agg_func.lower() == 'std':
                score = np.std(scores)
            elif self.agg_func.lower() == 'med':
                score = np.median(scores)
            else:
                assert False, 'Unknown aggregate function {}'.format(self.agg_func)
            _final_scores[qid] = score
        return _final_scores

    def print_score(self):
        for qid, score in self.final_score_dict.items():
            print('{} {}'.format(qid, score))


def main(args: parser):
    map_file = args.map
    predictions_file = args.predictions
    agg_func = args.function

    assert not (map_file is None and predictions_file is None), 'No file was given'

    if map_file is not None:
        ap_scores = DataReader(map_file, 'ap')
        aggregation = Aggregate(ap_scores.data_df, agg_func)
        aggregation.print_score()
    else:
        prediction_scores = DataReader(predictions_file, 'result')
        agg_prediction = Aggregate(prediction_scores.data_df, agg_func)
        agg_prediction.print_score()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
