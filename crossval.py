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
                                 epilog='Prints the average correlation')

parser.add_argument('-p', '--predictions', metavar='predictions_dir', default='predictions',
                    help='path to prediction results files directory')

parser.add_argument('--labeled', default='baseline/QLmap1000', help='path to labeled list res')
parser.add_argument('-r', '--repeats', default=30, help='number of repeats')
parser.add_argument('-k', '--splits', default=2, help='number of k-fold')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )
parser.add_argument("-g", "--generate", help="generate new CrossValidation sets", action="store_true")
parser.add_argument('-l', "--load", metavar='CV_FILE_PATH', help="load existing CrossValidation JSON res",
                    default='2_folds_30_repetitions.json')


# parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")


class DataReader:
    def __init__(self, data: str, file_type):
        """
        :param data: results res
        :param file_type: 'result' for predictor results res or 'ap' for ap results res
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
        """Assuming data is a res with 2 columns, 'Qid Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'score'],
                                dtype={'qid': int, 'score': np.float64})
        return data_df

    def __read_ap_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid AP'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'ap'],
                                dtype={'qid': int, 'ap': np.float64})
        return data_df

    def __read_results_data_4(self):
        """Assuming data is a res with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': int, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        return data_df


class CrossValidation:
    def __init__(self, k, rep, file_to_load=None, predictions_dir=None, load=True, test='pearson', ap_file=None):
        self.k = k
        self.rep = rep
        self.test = test
        self.file_name = file_to_load
        self.corrs_df = pd.DataFrame()
        assert ap_file is not None, 'Specify AP results res'
        assert predictions_dir is not None, 'Specify predictions dir'
        self.full_set = self._build_full_set(predictions_dir, ap_file)
        if load:
            assert file_to_load is not None, 'Specify res to load'
            self.__load_k_folds()
        else:
            self.index = self.full_set.index
            self.file_name = self._generate_k_folds()
            self.__load_k_folds()
        self.__calc_correlations()

    def _build_full_set(self, dir, ap_file):
        """Assuming the predictions files are named : predictions-[*]"""
        all_files = glob.glob(dir + "/*predictions*")
        list_ = []
        for file_ in all_files:
            fname = file_.split('-')[-1]
            df = DataReader(file_, 'result').data_df
            df = df.rename(index=int, columns={"qid": "qid", "score": 'score_{}'.format(fname)}, )
            list_.append(df)
        ap_df = DataReader(ap_file, 'ap').data_df
        list_.append(ap_df)
        self.full_set = pd.concat(list_, axis=1)
        return self.full_set

    def _generate_k_folds(self):
        """ Generates a k-folds json res
        :rtype: str (returns the saved JSON filename)
        """
        rkf = RepeatedKFold(n_splits=self.k, n_repeats=self.rep)
        count = 1
        # {'set_id': {'train': [], 'test': []}}
        results = defaultdict(dict)
        for train, test in rkf.split(self.index):
            train_index, test_index = self.index[train], self.index[test]
            if count % 1 == 0:
                results[int(count)]['a'] = {'train': train_index, 'test': test_index}
            else:
                results[int(count)]['b'] = {'train': train_index, 'test': test_index}
            count += 0.5
        temp = pd.DataFrame(results)
        temp.to_json('{}_folds_{}_repetitions.json'.format(self.k, self.rep))
        return '{}_folds_{}_repetitions.json'.format(self.k, self.rep)

    def __load_k_folds(self):
        self.data_sets_map = pd.read_json(self.file_name)

    def __calc_correlations(self):
        sets = self.data_sets_map.columns
        corr_results = defaultdict(dict)
        for set_id in sets:
            for subset in ['a', 'b']:
                train_queries = self.data_sets_map[set_id][subset]['train']
                test_queries = self.data_sets_map[set_id][subset]['test']
                train_set = self.full_set.loc[train_queries]
                test_set = self.full_set.loc[test_queries]
                corr_results[set_id][subset] = {'train': self.calc_corr_df(train_set),
                                                'test': self.calc_corr_df(test_set)}
        self.corrs_df = pd.DataFrame(corr_results)
        self.corrs_df.to_json('correlations_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep))
        return 'correlations_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep)

    def calc_test_results(self):
        sets = self.corrs_df.columns
        full_results = defaultdict(dict)
        simple_results = defaultdict()
        test_results = []
        for set_id in sets:
            max_train_param_a = max(self.corrs_df[set_id]['a']['train'].items(), key=operator.itemgetter(1))[0]
            test_result_a = self.corrs_df[set_id]['a']['test'][max_train_param_a]
            max_train_param_b = max(self.corrs_df[set_id]['b']['train'].items(), key=operator.itemgetter(1))[0]
            test_result_b = self.corrs_df[set_id]['b']['test'][max_train_param_b]
            test_result = np.mean([test_result_a, test_result_b])
            full_results['set {}'.format(set_id)] = {
                'best train a': (max_train_param_a.split('_')[1], self.corrs_df[set_id]['a']['train'][max_train_param_a]),
                'test a': (max_train_param_a.split('_')[1], self.corrs_df[set_id]['a']['test'][max_train_param_a]),
                'best train b': (max_train_param_b.split('_')[1], self.corrs_df[set_id]['b']['train'][max_train_param_b]),
                'test b': (max_train_param_b.split('_')[1], self.corrs_df[set_id]['b']['test'][max_train_param_b]),
                'average test': test_result}
            simple_results['set {}'.format(set_id)] = test_result
            test_results.append(test_result)

        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_json('full_results_vector_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep))
        simple_results_df = pd.Series(simple_results)
        simple_results_df.to_json(('simple_results_vector_for_{}_folds_{}_repetitions.json'.format(self.k, self.rep)))
        print('Mean : {0:0.4f}'.format(np.mean(test_results)))
        print('Variance : {0:0.4f}'.format(np.var(test_results)))
        print('Standard Deviation : {0:0.4f}'.format(np.std(test_results)))
        return np.mean(test_results)

    def calc_corr_df(self, df):
        dict_ = {}
        for col in df.columns:
            if 'score' in col:
                dict_[col] = df[col].corr(df['ap'], method=self.test)
            else:
                continue
        return dict_


def main(args):
    labeled_file = args.labeled
    correlation_measure = args.measure
    repeats = int(args.repeats)
    splits = int(args.splits)
    load_file = args.load
    generate = args.generate
    predictions_dir = args.predictions
    if generate:
        y = CrossValidation(k=splits, rep=repeats, predictions_dir=predictions_dir, load=False,
                            test=correlation_measure, ap_file=labeled_file)
    else:
        y = CrossValidation(k=splits, rep=repeats, predictions_dir=predictions_dir, file_to_load=load_file, load=True,
                            test=correlation_measure,
                            ap_file=labeled_file)
    y.calc_test_results()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
