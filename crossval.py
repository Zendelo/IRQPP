#!/usr/bin/env python

import argparse
import glob
import operator
import os
from abc import abstractmethod, ABC
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

import dataparser as dp

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
    def __init__(self, k=2, rep=30, folds_map_file=None, predictions_dir=None, load=True, test='pearson', ap_file=None):
        self.k = k
        self.rep = rep
        self.test = test
        self.folds_file = folds_map_file
        assert predictions_dir is not None, 'Specify predictions dir'
        predictions_dir = os.path.abspath(os.path.normpath(os.path.expanduser(predictions_dir)))
        assert os.listdir(predictions_dir), f'{predictions_dir} is empty'
        self.output_dir = predictions_dir.replace('predictions', 'evaluation')
        dp.ensure_dir(self.output_dir)
        if ap_file:
            self.full_set = self._build_full_set(predictions_dir, ap_file)
            if '-' in ap_file:
                self.ap_func = ap_file.split('-')[-1]
            else:
                self.ap_func = 'basic'
        else:
            self.full_set = self._build_full_set(predictions_dir)
        if load:
            assert folds_map_file is not None, 'Specify k-folds file to load'
            self.__load_k_folds()
        else:
            self.index = self.full_set.index
            self.folds_file = self._generate_k_folds()
            self.__load_k_folds()
        self.corr_df = None

    @abstractmethod
    def calc_function(self, df: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def _build_full_set(predictions_dir, ap_file=None):
        """Assuming the predictions files are named : predictions-[*]"""
        all_files = glob.glob(predictions_dir + "/*predictions*")
        if 'uef' in predictions_dir:
            # Excluding all the 5 and 10 docs predictions
            if 'qf' in predictions_dir:
                all_files = [fn for fn in all_files if
                             not os.path.basename(fn).endswith('-5+', 11, 14) and not os.path.basename(fn).endswith(
                                 '-10+', 11, 15)]
            else:
                all_files = [fn for fn in all_files if
                             not os.path.basename(fn).endswith('-5') and not os.path.basename(fn).endswith('-10')]
        list_ = []
        for file_ in all_files:
            fname = file_.split('-')[-1]
            df = dp.ResultsReader(file_, 'predictions').data_df
            df = df.rename(columns={"score": f'score_{fname}'})
            list_.append(df)
        if ap_file:
            ap_df = dp.ResultsReader(ap_file, 'ap').data_df
            list_.append(ap_df)
        full_set = pd.concat(list_, axis=1, sort=True)
        assert not full_set.empty, f'The Full set DF is empty, make sure that {predictions_dir} is not empty'
        return full_set

    def _generate_k_folds(self):
        # FIXME: Need to fix it to generate a DF with folds, without redundancy
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
        temp.to_json(f'{self.k}_folds_{self.rep}_repetitions.json')
        return f'{self.k}_folds_{self.rep}_repetitions.json'

    def __load_k_folds(self):
        # self.data_sets_map = pd.read_json(self.file_name).T['a'].apply(pd.Series).rename(
        #     mapper={'train': 'fold-1', 'test': 'fold-2'}, axis='columns')
        self.data_sets_map = pd.read_json(self.folds_file)

    def _calc_eval_metric_df(self):
        sets = self.data_sets_map.index
        folds = self.data_sets_map.columns
        corr_results = defaultdict(dict)
        for set_id in sets:
            for fold in folds:
                train_queries = set()
                # a hack to create a set out of train queries, from multiple lists
                _ = {train_queries.update(i) for i in self.data_sets_map.loc[set_id, folds != fold].values}
                test_queries = set(self.data_sets_map.loc[set_id, fold])
                train_set = self.full_set.loc[map(str, train_queries)]
                test_set = self.full_set.loc[map(str, test_queries)]
                corr_results[set_id][fold] = pd.DataFrame(
                    {'train': self.calc_function(train_set), 'test': self.calc_function(test_set)})
        corr_df = pd.DataFrame.from_dict(corr_results, orient='index')
        try:
            corr_df.to_pickle(
                f'{self.output_dir}/correlations_for_{self.k}_folds_{self.rep}_repetitions_{self.ap_func}.pkl')
        except AttributeError:
            corr_df.to_pickle(f'{self.output_dir}/correlations_for_{self.k}_folds_{self.rep}_repetitions_pageRank.pkl')
        return corr_df

    def calc_test_results(self):
        if not hasattr(self, 'corr_df'):
            self.corr_df = self._calc_eval_metric_df()
        sets = self.data_sets_map.index
        full_results = defaultdict(dict)
        simple_results = defaultdict()
        for set_id in sets:
            _res_per_set = []
            for fold in self.corr_df.loc[set_id].index:
                max_train_param = self.corr_df.loc[set_id, fold].idxmax()['train']
                train_result, test_result = self.corr_df.loc[set_id, fold].loc[max_train_param]
                _res_per_set.append(test_result)
                full_results[set_id, fold] = {'best_train_param': max_train_param.split('_')[1],
                                              'best_train_val': train_result, 'test_val': test_result}
            simple_results[f'set_{set_id}'] = np.mean(_res_per_set)
        full_results_df = pd.DataFrame.from_dict(full_results, orient='index')
        try:
            full_results_df.to_json(
                f'{self.output_dir}/'
                f'full_results_vector_for_{self.k}_folds_{self.rep}_repetitions_{self.ap_func}_{self.test}.json')
        except AttributeError:
            full_results_df.to_json(
                f'{self.output_dir}/'
                f'full_results_vector_for_{self.k}_folds_{self.rep}_repetitions_pageRank_{self.test}.json')

        simple_results_df = pd.Series(simple_results)
        try:
            simple_results_df.to_json(
                f'{self.output_dir}/'
                f'simple_results_vector_for_{self.k}_folds_{self.rep}_repetitions_{self.ap_func}.json')
        except AttributeError:
            simple_results_df.to_json(
                f'{self.output_dir}/'
                f'simple_results_vector_for_{self.k}_folds_{self.rep}_repetitions_pageRank.json')

        mean = simple_results_df.mean()
        return f'{mean:.3f}'

    @staticmethod
    def read_eval_results(results_file):
        # FIXME: need to fix it after changing the format of the eval files
        temp_df = pd.read_json(results_file, orient='index')

        # Split column of lists into several columns
        res_df = pd.DataFrame(temp_df['best train a'].values.tolist(), index=temp_df.index.str.split().str[1],
                              columns=['a', 'train_correlation_a'])
        res_df.rename_axis('set', inplace=True)
        res_df[['b', 'train_correlation_b']] = pd.DataFrame(temp_df['best train b'].values.tolist(),
                                                            index=temp_df.index.str.split().str[1])
        return res_df


class InterTopicCrossValidation(CrossValidation, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('ap_file'):
            self.calc_function = self.calc_inter_topic_corr
            self.corr_df = self._calc_eval_metric_df()
        else:
            self.calc_function = self.calc_inter_topic_scores

    def calc_inter_topic_corr(self, df):
        dict_ = {}
        for col in df.columns:
            if col != 'ap':
                dict_[col] = df[col].corr(df['ap'], method=self.test)
            else:
                continue
        return pd.Series(dict_)

    def calc_inter_topic_scores(self, df):
        return df.mean().to_dict()


class IntraTopicCrossValidation(CrossValidation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calc_intra_topic_corr(self, df):
        # FIXME: need to fix this bad boy
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

    res_dir, data_dir = dp.set_environment_paths()

    # # Debugging
    # print('\n\n\n------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------\n\n\n')
    # # predictor = input('What predictor should be used for debugging?\n')
    # predictor = 'nqc'
    # corpus = 'ROBUST'
    # # corpus = 'ClueWeb12B'
    # res_dir = os.path.join(res_dir, corpus)
    # labeled_file = f'{res_dir}/test/ref/QLmap1000-title'
    # load_file = f'{res_dir}/test/2_folds_30_repetitions.json'
    # predictions_dir = f'{res_dir}/uqvPredictions/referenceLists/title/all_vars/general/jac/{predictor}/predictions/'

    if generate:
        y = InterTopicCrossValidation(k=splits, rep=repeats, predictions_dir=predictions_dir, load=False,
                                      test=correlation_measure, ap_file=labeled_file)
    else:
        y = InterTopicCrossValidation(k=splits, rep=repeats, predictions_dir=predictions_dir, folds_map_file=load_file,
                                      load=True,
                                      test=correlation_measure,
                                      ap_file=labeled_file)
    y.calc_test_results()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
