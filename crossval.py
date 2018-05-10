#!/usr/bin/env python

import argparse
import csv
import glob
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

parser = argparse.ArgumentParser(description='Cross Validation script',
                                 usage='Use CV to optimize correlation',
                                 epilog='The files must have 2 columns, first for index and second for the values')

parser.add_argument('-p', '--predictions', metavar='predictions_dir',
                    default='./', help='path to prediction results files directory')
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
parser.add_argument("-g", "--generate", help="generate new CrossValidation sets",
                    action="store_true")
parser.add_argument("--load", help="load existing CrossValidation JSON file",
                    default='2_folds_30_repetitions.json')
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")


class DataReader:
    def __init__(self, data: str, file_type):
        """
        :param data: results file
        :param file_type: 'result' for predictor results file or 'ap' for ap results file
        """
        self.file_type = file_type
        self.data = data
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
        """Assuming data is a file with 2 columns, 'Qid Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'score'],
                                dtype={'qid': int, 'score': np.float64})
        return data_df

    def __read_ap_data_2(self):
        """Assuming data is a file with 2 columns, 'Qid AP'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'ap'],
                                dtype={'qid': int, 'ap': np.float64})
        return data_df

    def __read_results_data_4(self):
        """Assuming data is a file with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': int, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        return data_df


class CrossValidation:
    def __init__(self, file_to_load=None, results_file=None, k=2, rep=1, load=True):
        self.k = k
        self.rep = rep
        self.file_name = file_to_load
        self.full_set = pd.DataFrame()
        self.corrs_df = pd.DataFrame()
        assert (not load and results_file is not None) or (load and file_to_load is not None), 'Specify file to load'
        if load:
            self.__load_k_folds()
        else:
            self.data = DataReader(results_file, 'results')
            self.index = self.data.data_df.index
            self.file_name = self._generate_k_folds()
            self.__load_k_folds()

    def _generate_k_folds(self):
        """ Generates a k-folds json file
        :rtype: str (returns the saved JSON filename)
        """
        rkf = RepeatedKFold(n_splits=self.k, n_repeats=self.rep)
        count = 1
        # {'set_id': {'train': [], 'test': []}}
        results = defaultdict(dict)
        for train, test in rkf.split(self.index):
            train_index, test_index = self.index[train], self.index[test]

            results[count] = {'train': train_index, 'test': test_index}
            count += 1
        pd.DataFrame(results).to_json('{}_folds_{}_repetitions.json'.format(self.k, self.rep))
        return '{}_folds_{}_repetitions.json'.format(self.k, self.rep)

    def __load_k_folds(self):
        self.data_sets_map = pd.read_json(self.file_name)

    def calc_correlations(self):
        sets = self.data_sets_map.columns
        corr_results = defaultdict(dict)
        for set_number in sets:
            train_quries = self.data_sets_map[set_number]['train']
            test_quries = self.data_sets_map[set_number]['test']
            train_set = self.full_set.loc[train_quries]
            test_set = self.full_set.loc[test_quries]
            corr_results[set_number] = {'train': self.calc_corr_df(train_set), 'test': self.calc_corr_df(test_set)}
        self.corrs_df = pd.DataFrame(corr_results)
        self.corrs_df.to_json('correlations_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep))
        return 'correlations_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep)

    def calc_test_results(self):
        sets = self.corrs_df.columns
        test_results = []
        for set_id in sets:
            max_train_param = max(self.corrs_df[set_id]['train'].items(), key=operator.itemgetter(1))[0]
            print(max_train_param)
            test_result = self.corrs_df[set_id]['test'][max_train_param]
            print(test_result)
            test_results.append(test_result)
        print('The average result for clarity is: {0:0.4f}'.format(np.average(test_results)))

    def create_full_set(self, dir, ap_file):
        all_files = glob.glob(dir + "/*docs.res")
        list_ = []
        for file_ in all_files:
            fname = file_.split('-')[2]
            df = DataReader(file_, 'result').data_df
            df = df.rename(index=int, columns={"qid": "qid", "score": 'score_{}'.format(fname)})
            list_.append(df)
        ap_df = DataReader(ap_file, 'ap').data_df
        list_.append(ap_df)
        self.full_set = pd.concat(list_, axis=1)
        return self.full_set

    @staticmethod
    def calc_corr_df(df, test='pearson'):
        dict_ = {}
        for col in df.columns:
            if col == 'ap':
                continue
            dict_[col] = df[col].corr(df['ap'], method=test)
        return dict_


# def calc_cor_files(first_file, second_file, test):
#     first_df = pd.read_table(first_file, delim_whitespace=True, header=None, index_col=0, names=['x'])
#     second_df = pd.read_table(second_file, delim_whitespace=True, header=None, index_col=0, names=['y'])
#     return calc_cor_df(first_df, second_df, test)
#
#
# def calc_cor_df(first_df, second_df, test='pearson'):
#     merged_df = pd.merge(first_df, second_df, left_index=True, right_index=True, how='inner', validate='1:1')
#     return merged_df['score_x'].corr(merged_df['score_y'], method=test)


def main(args):
    # parameters_xml = args.parameters
    # test_parameter = args.testing
    labeled_file = args.labeled
    # queries = args.queries
    # correlation_measure = args.measure
    # repeats = args.repeats
    # splits = args.splits
    load_file = args.load
    predictions_dir = args.predictions
    # raw_data = []
    # data = input('Insert file path')
    # while data != '':
    #     raw_data.append(data)
    #     data = input('Insert file path')
    # for file in raw_data:
    #     docs = file.split('-')[1]
    #     data = DataReader(file, docs)
    y = CrossValidation(file_to_load=load_file)
    y.create_full_set(predictions_dir, labeled_file)
    y.calc_correlations()
    y.calc_test_results()


# tmp-testing/clarity-Fiana/predictions-25

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
