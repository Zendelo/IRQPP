#!/usr/bin/env python

import argparse
import glob
import operator
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from dataparser import ResultsReader, ensure_dir

# TODO: change the functions to work with pandas methods such as idxmax
# TODO: Consider change to the folds file to be more convenient for pandas DF

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


class CrossValidation:
    def __init__(self, k=2, rep=30, file_to_load=None, predictions_dir=None, load=True, test='pearson', ap_file=None):
        self.k = k
        self.rep = rep
        self.test = test
        self.file_name = file_to_load
        assert predictions_dir is not None, 'Specify predictions dir'
        predictions_dir = os.path.abspath(os.path.normpath(os.path.expanduser(predictions_dir)))
        self.output_dir = predictions_dir.replace('predictions', 'evaluation')
        ensure_dir(self.output_dir)
        if ap_file:
            self.full_set = self._build_full_set(predictions_dir, ap_file)
            if '-' in ap_file:
                self.ap_func = ap_file.split('-')[-1]
            else:
                self.ap_func = None
        else:
            self.full_set = self._build_full_set(predictions_dir)
        if load:
            assert file_to_load is not None, 'Specify k-folds file to load'
            self.__load_k_folds()
        else:
            self.index = self.full_set.index
            self.file_name = self._generate_k_folds()
            self.__load_k_folds()
        if ap_file:
            self.corrs_df = self.__calc_correlations()

    @staticmethod
    def _build_full_set(predictions_dir, ap_file=None):
        """Assuming the predictions files are named : predictions-[*]"""
        all_files = glob.glob(predictions_dir + "/*predictions*")
        list_ = []
        for file_ in all_files:
            fname = file_.split('-')[-1]
            df = ResultsReader(file_, 'predictions').data_df
            df = df.rename(columns={"score": 'score_{}'.format(fname)})
            list_.append(df)
        if ap_file:
            ap_df = ResultsReader(ap_file, 'ap').data_df
            list_.append(ap_df)
        full_set = pd.concat(list_, axis=1)
        return full_set

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
                train_queries = np.array(self.data_sets_map[set_id][subset]['train']).astype(str)
                test_queries = np.array(self.data_sets_map[set_id][subset]['test']).astype(str)
                train_set = self.full_set.loc[train_queries]
                test_set = self.full_set.loc[test_queries]
                corr_results[set_id][subset] = {'train': self.calc_corr_df(train_set),
                                                'test': self.calc_corr_df(test_set)}
        corrs_df = pd.DataFrame(corr_results)
        corrs_df.to_json('{}/correlations_for_{}_folds_{}_repetitions_{}.json'.format(self.output_dir, self.k, self.rep,
                                                                                      self.ap_func))
        return corrs_df

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
                'best train a': (
                    max_train_param_a.split('_')[1], self.corrs_df[set_id]['a']['train'][max_train_param_a]),
                'test a': (max_train_param_a.split('_')[1], self.corrs_df[set_id]['a']['test'][max_train_param_a]),
                'best train b': (
                    max_train_param_b.split('_')[1], self.corrs_df[set_id]['b']['train'][max_train_param_b]),
                'test b': (max_train_param_b.split('_')[1], self.corrs_df[set_id]['b']['test'][max_train_param_b]),
                'average test': test_result}
            simple_results['set {}'.format(set_id)] = test_result
            test_results.append(test_result)
        full_results_df = pd.DataFrame(full_results)
        full_results_df.to_json(
            '{}/full_results_vector_for_{}_folds_{}_repetitions_{}.json'.format(self.output_dir, self.k, self.rep,
                                                                                self.ap_func))
        simple_results_df = pd.Series(simple_results)
        simple_results_df.to_json(
            ('{}/simple_results_vector_for_{}_folds_{}_repetitions_{}.json'.format(self.output_dir, self.k, self.rep,
                                                                                   self.ap_func)))
        mean = np.mean(test_results)
        return '{:.3f}'.format(mean)

    def calc_corr_df(self, df):
        dict_ = {}
        for col in df.columns:
            if 'score' in col:
                dict_[col] = df[col].corr(df['ap'], method=self.test)
            else:
                continue
        return dict_

    @staticmethod
    def read_eval_results(results_file):
        temp_df = pd.read_json(results_file, orient='index')

        # Split column of lists into several columns
        res_df = pd.DataFrame(temp_df['best train a'].values.tolist(), index=temp_df.index.str.split().str[1],
                              columns=['a', 'train_correlation_a'])
        res_df.rename_axis('set', inplace=True)
        res_df[['b', 'train_correlation_b']] = pd.DataFrame(temp_df['best train b'].values.tolist(),
                                                                   index=temp_df.index.str.split().str[1])
        return res_df


def char_range(a, z):
    """Creates a generator that iterates the characters from `c1` to `c2`, inclusive."""
    # ord returns the ASCII value, chr returns the char of ASCII value
    for c in range(ord(a), ord(z) + 1):
        yield chr(c)


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
