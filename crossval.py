#!/usr/bin/env python

import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import argparse

parser = argparse.ArgumentParser(description='Cross Validation script',
                                 usage='Use CV to optimize correlation',
                                 epilog='The files must have 2 columns, first for index and second for the values')

parser.add_argument('--predictor', metavar='predictor_file_path',
                    default='SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana', help='path to predictor executable file')
parser.add_argument('--parameters', metavar='parameters_file_path', default='clarity/clarityParam.xml',
                    help='path to predictor parameters file')
parser.add_argument('--testing', metavar='running_parameter', default='-documents=', choices=['-documents'],
                    help='The parameter to optimize')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='data/ROBUST/queries.xml',
                    help='path to queries xml file')
parser.add_argument('-l', '--labeled', default='baseline/QLmap1000', help='path to labeled list file')
parser.add_argument('-r', '--repeats', default=30, help='number of repeats')
parser.add_argument('-k', '--splits', default=2, help='number of k-fold')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")


class DataReader:
    def __init__(self, data: str):
        self.data = data
        self.__number_of_col = self.__check_number_of_col()
        assert self.__number_of_col == 2 or self.__number_of_col == 4, 'Wrong File format'
        self.data_df = self.__read_data_2() if self.__number_of_col == 2 else self.__read_data_4()

    def __check_number_of_col(self):
        with open(self.data) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            first_row = next(reader)
            num_cols = len(first_row)
        return int(num_cols)

    def __read_data_2(self):
        """Assuming data is a file with 2 columns, 'Qid Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0, names=['qid', 'score'],
                                dtype={'qid': int, 'score': np.float64})
        return data_df

    def __read_data_4(self):
        """Assuming data is a file with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': int, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        return data_df


class CrossValidation:
    def __init__(self, dataset, labeled, k=2, rep=1):
        self.k = k
        self.rep = rep
        self.data = dataset.data_df
        self.labels = labeled.data_df
        self.index = self.data.index

    def add_results(self, res):
        row = pd.DataFrame(res)
        row = row.set_index('set_id')
        if hasattr(self, 'results_df'):
            self.results_df.append(row)
        else:
            self.results_df = pd.DataFrame(row)

    @staticmethod
    def _df2file(fname, df):
        df.to_csv(fname, sep=' ', header=False)

    @staticmethod
    def _index2file(fname, index):
        index.to_series().to_csv(fname, index=False)

    def calc_k_folds(self):
        """TODO: need to save the K-folds for future use with other predictors
        And the results, Also save the vector of the results (Spearman/Pearson...)"""
        rkf = RepeatedKFold(n_splits=self.k, n_repeats=self.rep)
        count = 1
        results = {'set_id': [], 'train': [], 'test': []}
        for train, test in rkf.split(self.index):
            train_index, test_index = self.index[train], self.index[test]
            self._index2file('qid_train_set_{}'.format(count), train_index)
            self._index2file('qid_test_set_{}'.format(count), test_index)
            train_df, test_df = self.data.loc[train_index], self.data.loc[test_index]
            # print("\nTRAIN:", train_df.shape, "\nTEST:", test_df.shape)
            train_result = calc_cor_df(train_df, self.labels)
            test_result = calc_cor_df(test_df, self.labels)
            results['set_id'].append(count)
            results['train'].append(train_result)
            results['test'].append(test_result)
            print(train_result)
            count += 1
        print(results)
        self.add_results(results)
        print(self.results_df)


def calc_cor_files(first_file, second_file, test):
    first_df = pd.read_table(first_file, delim_whitespace=True, header=None, index_col=0, names=['x'])
    second_df = pd.read_table(second_file, delim_whitespace=True, header=None, index_col=0, names=['y'])
    return calc_cor_df(first_df, second_df, test)


def calc_cor_df(first_df, second_df, test='pearson'):
    merged_df = pd.merge(first_df, second_df, left_index=True, right_index=True, how='inner', validate='1:1')
    return merged_df['score_x'].corr(merged_df['score_y'], method=test)


def main(args):
    predictor_exe = args.predictor
    parameters_xml = args.parameters
    test_parameter = args.testing
    labeled_file = args.labeled
    queries = args.queries
    correlation_measure = args.measure
    repeats = args.repeats
    splits = args.splits
    data = DataReader('newfile')
    gt = DataReader(labeled_file)
    y = CrossValidation(data, gt)
    y.calc_k_folds()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
