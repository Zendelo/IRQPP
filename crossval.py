#!/usr/bin/env python

import argparse
import glob
import os
from abc import abstractmethod, ABC
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from qpputils import dataparser as dp

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
parser.add_argument('-f', "--folds_file", metavar='CV_FILE_PATH', help="load existing CrossValidation JSON res",
                    default='2_folds_30_repetitions.json')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


class CrossValidation:
    def __init__(self, folds_map_file=None, k=2, rep=30, predictions_dir=None, test='pearson', ap_file=None,
                 generate_folds=False, **kwargs):
        logging.debug("testing logger")
        self.k = k
        self.rep = rep
        self.test = test
        assert predictions_dir, 'Specify predictions dir'
        assert folds_map_file, 'Specify path for CV folds file'
        predictions_dir = os.path.abspath(os.path.normpath(os.path.expanduser(predictions_dir)))
        assert os.listdir(predictions_dir), f'{predictions_dir} is empty'
        self.output_dir = dp.ensure_dir(predictions_dir.replace('predictions', 'evaluation'))
        if ap_file:
            self.full_set = self._build_full_set(predictions_dir, ap_file)
            if '-' in ap_file:
                self.ap_func = ap_file.split('-')[-1]
            else:
                self.ap_func = 'basic'
        else:
            self.full_set = self._build_full_set(predictions_dir)
        if generate_folds:
            self.index = self.full_set.index
            self.folds_file = self._generate_k_folds()
            self.__load_k_folds()
        else:
            try:
                self.folds_file = dp.ensure_file(folds_map_file)
            except FileExistsError:
                print("The folds file specified doesn't exist, going to generate the file and save")
            self.__load_k_folds()

        # self.corr_df = NotImplemented

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
        self.calc_function = self.calc_inter_topic_corr if kwargs.get('ap_file') else self.calc_inter_topic_scores
        # self.corr_df = self._calc_eval_metric_df()

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


class IntraTopicCrossValidation(CrossValidation, ABC):
    """
    Class for intra topic evaluation, i.e. evaluation is per topic across its variants

    Parameters
    ----------
    :param bool save_calculations: set to True to save the intermediate results.
        in order to load intermediate results use a specific method to do that explicitly, the results will not
        be loaded during calculation in order to avoid bugs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sir = kwargs.get('save_calculations', False)
        self.full_set = dp.add_topic_to_qdf(self.full_set).set_index('topic')
        if kwargs.get('ap_file'):
            self.calc_function = self.calc_intra_topic_corr
            # self.corr_df = self._calc_eval_metric_df()
        # else:
        #     self.calc_function = self.calc_intra_topic_corr
        self.test_per_topic = pd.DataFrame(index=self.full_set.index.unique())

    def _calc_eval_metric_df(self):
        sets = self.data_sets_map.index
        folds = self.data_sets_map.columns
        corr_results = defaultdict(dict)
        for set_id in sets:
            _test = []
            for fold in folds:
                train_queries = set()
                # a hack to create a set out of train queries, from multiple lists
                _ = {train_queries.update(i) for i in self.data_sets_map.loc[set_id, folds != fold].values}
                test_queries = set(self.data_sets_map.loc[set_id, fold])
                train_set = self.full_set.loc[map(str, train_queries)]
                test_set = self.full_set.loc[map(str, test_queries)]
                _ts_df = self.calc_function(test_set)
                _tr_df = self.calc_function(train_set)
                _test_df = _ts_df.loc[:, _ts_df.columns != 'weight'].apply(np.average, axis='index',
                                                                           weights=_ts_df['weight'])
                _train_df = _tr_df.loc[:, _tr_df.columns != 'weight'].apply(np.average, axis='index',
                                                                            weights=_tr_df['weight'])
                _sr = _ts_df.loc[:, _train_df.idxmax()]
                _sr.name = set_id
                self.test_per_topic = self.test_per_topic.join(_sr, rsuffix=f'-{set_id}')
                corr_results[set_id][fold] = pd.DataFrame({'train': _train_df, 'test': _test_df})
        self.test_per_topic['weight'] = self.full_set.groupby('topic')['qid'].count()
        corr_df = pd.DataFrame.from_dict(corr_results, orient='index')
        try:
            corr_df.to_pickle(
                f'{self.output_dir}/correlations_for_{self.k}_folds_{self.rep}_repetitions_{self.ap_func}.pkl')
        except AttributeError:
            corr_df.to_pickle(f'{self.output_dir}/correlations_for_{self.k}_folds_{self.rep}_repetitions_pageRank.pkl')
        if self.sir:
            self.test_per_topic.to_pickle(
                f'{self.output_dir}/per_topic_correlations_for_{self.k}_folds_{self.rep}_repetitions_pageRank.pkl')
        return corr_df

    def calc_intra_topic_corr(self, df: pd.DataFrame):
        """
        This method calculates Kendall tau's correlation coefficient per topic, and returns
        the weighted average correlation over the topics. Weighted by number of vars.
        :param df:
        :return: pd.Series, the index is all the hyper params and values are weighted average correlations
        """
        dict_ = {}
        df = df.reset_index().set_index(['topic', 'qid'])
        for topic, _df in df.groupby('topic'):
            dict_[topic] = _df.loc[:, _df.columns != 'ap'].corrwith(_df['ap'], method=self.test).append(
                pd.Series({'weight': len(_df)}))
            # dict_[topic] = _df.loc[:, _df.columns != 'ap'].corrwith(_df['ap'], method='pearson')
        _df = pd.DataFrame.from_dict(dict_, orient='index')
        # self.test_per_topic = _df
        return _df

    def load_per_topic_df(self):
        try:
            inter_res_file = dp.ensure_file(
                f'{self.output_dir}/per_topic_correlations_for_{self.k}_folds_{self.rep}_repetitions_pageRank.pkl')
        except AssertionError:
            logging.warning(
                f"File {self.output_dir}/per_topic_correlations_for_{self.k}_folds_{self.rep}_repetitions_pageRank.pkl doesnt exist")
            return None
        df = pd.read_pickle(inter_res_file)
        return df


def main(args):
    labeled_file = args.labeled
    correlation_measure = args.measure
    repeats = int(args.repeats)
    splits = int(args.splits)
    folds_file = args.folds_file
    generate = args.generate
    predictions_dir = args.predictions

    res_dir, data_dir = dp.set_environment_paths()

    # Debugging
    print('\n\n\n------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------\n\n\n')
    # predictor = input('What predictor should be used for debugging?\n')
    predictor = 'nqc'
    corpus = 'ROBUST'
    # corpus = 'ClueWeb12B'
    correlation_measure = 'kendall'
    # correlation_measure = 'pearson'
    res_dir = os.path.join(res_dir, corpus)
    # labeled_file = f'{res_dir}/test/ref/QLmap1000-title'
    labeled_file = f'{res_dir}/test/raw/QLmap1000'
    folds_file = f'{res_dir}/test/2_folds_30_repetitions.json'
    # predictions_dir = f'{res_dir}/uqvPredictions/referenceLists/title/all_vars/general/jac/{predictor}/predictions/'
    predictions_dir = f'{res_dir}/uqvPredictions/referenceLists/pageRank/raw/Jac_coefficient/{predictor}/predictions/'

    y = IntraTopicCrossValidation(folds_map_file=folds_file, k=splits, rep=repeats, predictions_dir=predictions_dir,
                                  test=correlation_measure, ap_file=labeled_file, generate_folds=generate)

    y.calc_test_results()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
