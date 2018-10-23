from reportlab.lib.pagesizes import elevenSeventeen

import dataparser as dp
from itertools import combinations
from collections import defaultdict
from RBO import rbo_dict
import pandas as pd
import numpy as np
import os

import argparse
from Timer.timer import Timer

# TODO: fix the constructor of the class, to work only with the results for the chosen variations

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3.6 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])

parser.add_argument('-g', '--group', help='group of queries to predict', choices=['top', 'low', 'title'])
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')

parser.add_argument('--generate', help="generate new features file", action="store_true")
parser.add_argument('--predict', help="generate new predictions", action="store_true")


class QueryFeatureFactory:
    def __init__(self, corpus, queries_group='title', rbo_top=100):
        self.rbo_top = rbo_top
        self.__set_paths(corpus, queries_group)
        _raw_res_data = dp.ResultsReader(self.results_file, 'trec')
        _title_res_data = dp.ResultsReader(self.title_res_file, 'trec')
        self.queries_data = dp.QueriesTextParser(self.queries_full_file, 'uqv')
        self.topics_data = dp.QueriesTextParser(self.queries_topic_file)
        self.variations_data = dp.QueriesTextParser(self.queries_variations_file, 'uqv')
        # _var_scores_df.loc[_var_scores_df['qid'].isin(_vars_list)]
        self.raw_res_data = _raw_res_data
        self.title_res_data = _title_res_data
        # A simple sanity check to make sure number of results and query variations is identical
        _x = [len(i) for i in self.raw_res_data.query_vars.values()]
        _z = [len(i) for i in self.queries_data.query_vars.values()]
        assert _x == _z, 'Results and Queries files don\'t match'
        self.fused_data = dp.ResultsReader(self.fused_results_file, 'trec')
        self.query_vars = self.queries_data.query_vars

    @classmethod
    def __set_paths(cls, corpus, qgroup):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        # cls.predictor = predictor
        _results_file = f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res'
        cls.results_file = os.path.normpath(os.path.expanduser(_results_file))
        dp.ensure_file(cls.results_file)

        _title_results_file = f'~/QppUqvProj/Results/{corpus}/test/basic/QL.res'
        cls.title_res_file = os.path.normpath(os.path.expanduser(_title_results_file))
        dp.ensure_file(cls.title_res_file)

        _queries_full_file = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.txt'
        cls.queries_full_file = os.path.normpath(os.path.expanduser(_queries_full_file))
        dp.ensure_file(cls.queries_full_file)

        _queries_variations_file = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_{qgroup}.txt'
        cls.queries_variations_file = os.path.normpath(os.path.expanduser(_queries_variations_file))
        dp.ensure_file(cls.queries_variations_file)

        _queries_topic_file = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_{qgroup}.txt'
        cls.queries_topic_file = os.path.normpath(os.path.expanduser(_queries_topic_file))
        dp.ensure_file(cls.queries_topic_file)

        _fused_results_file = f'~/QppUqvProj/Results/{corpus}/test/fusion/QL.res'
        cls.fused_results_file = os.path.normpath(os.path.expanduser(_fused_results_file))
        dp.ensure_file(cls.fused_results_file)

        cls.output_dir = f'~/QppUqvProj/Results/{corpus}/test/raw/'
        dp.ensure_dir(cls.output_dir)

    def _calc_features(self):
        _dict = {'topic': [], 'qid': [], 'Jac_coefficient': [], 'Top_10_Docs_overlap': [], 'RBO_EXT_100': [],
                 'RBO_FUSED_EXT_100': []}

        for topic in self.topics_data.queries_dict.keys():
            q_vars = self.query_vars.get(topic)
            _dict['topic'] += [topic] * len(q_vars)
            dt_100 = self.fused_data.get_res_dict_by_qid(topic, top=self.rbo_top)
            topic_txt = self.topics_data.get_qid_txt(topic)
            topics_top_list = self.title_res_data.get_docs_by_qid(topic, 10)
            # topics_top_list = self.title_res_data.get_docs_by_qid(topic, 25)
            topic_results_list = self.title_res_data.get_res_dict_by_qid(topic, top=self.rbo_top)

            for var in q_vars:
                var_txt = self.queries_data.get_qid_txt(var)
                jc = self.jaccard_coefficient(topic_txt, var_txt)

                var_top_list = self.raw_res_data.get_docs_by_qid(var, 10)
                # var_top_list = self.raw_res_data.get_docs_by_qid(var, 25)
                docs_overlap = self.list_overlap(topics_top_list, var_top_list)

                # All RBO values are rounded to 10 decimal digits, to avoid float overflow
                var_results_list = self.raw_res_data.get_res_dict_by_qid(var, top=self.rbo_top)
                _rbo_scores_100 = rbo_dict(topic_results_list, var_results_list, p=0.95)
                _ext_100 = np.around(_rbo_scores_100['ext'], 10)

                _fused_rbo_scores_100 = rbo_dict(dt_100, var_results_list, p=0.95)
                _var_fext_100 = np.around(_fused_rbo_scores_100['ext'], 10)

                _dict['qid'] += [var]
                _dict['Jac_coefficient'] += [jc]
                _dict['Top_10_Docs_overlap'] += [docs_overlap]
                _dict['RBO_EXT_100'] += [_ext_100]
                _dict['RBO_FUSED_EXT_100'] += [_var_fext_100]

        _df = pd.DataFrame.from_dict(_dict)
        # _df.set_index(['topic', 'qid'], inplace=True)
        return _df

    def _filter_queries(self, df):
        # return df[df['Jac_coefficient'] != 1]
        return df.loc[df['qid'].isin(self.variations_data.queries_df['qid'])]

    def _soft_max_scores(self, df):
        # _df = self._filter_queries(df)
        _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        _exp_df = _df.apply(np.exp)
        # For debugging purposes
        z_e = _exp_df.groupby(['topic']).sum()

        softmax_df = (_exp_df.groupby(['topic', 'qid']).sum() / z_e)
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return softmax_df

    def _average_scores(self, df):
        _df = self._filter_queries(df)
        # _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        # _exp_df = _df.apply(np.exp)
        # For debugging purposes
        avg_df = _df.groupby(['topic']).mean()

        # avg_df = (_df.groupby(['topic', 'qid']).mean())
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return avg_df

    def _max_norm_scores(self, df):
        # _df = self._filter_queries(df)
        _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        # For debugging purposes
        z_m = _df.groupby(['topic']).max()

        max_norm_df = (_df.groupby(['topic', 'qid']).sum() / z_m).fillna(0)
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return max_norm_df

    def _sum_scores(self, df):
        # TODO: add save to rbo and rbof to a file here as predictions
        _df = df
        # filter only variations different from original query
        # _df = self._filter_queries(df)
        z_n = _df.groupby(['topic']).sum()
        # TODO: check this - filling nan with 0
        norm_df = (_df.groupby(['topic', 'qid']).sum() / z_n).fillna(0)
        # _temp = norm_df.dropna()
        return norm_df

    def save_rbo_predictions(self, df: pd.DataFrame):
        _df = self._filter_queries(df)
        _df = df.groupby('topic').mean()
        _df['RBO_EXT_100'].to_csv(f'RBO_predictions-{self.rbo_top}', sep=' ')
        _df['RBO_FUSED_EXT_100'].to_csv(f'Fused_predictions-{self.rbo_top}', sep=' ')

    def generate_features(self):
        _df = self._calc_features()
        # return self._soft_max_scores(_df)
        return self._sum_scores(_df)
        # return self._average_scores(_df)
        # return self._max_norm_scores(_df)

    def generate_rbo_predictions(self):
        _df = self._calc_features()
        self.save_rbo_predictions(_df)

    @staticmethod
    def jaccard_coefficient(st1: str, st2: str):
        st1_set = set(st1.split())
        st2_set = set(st2.split())
        union = st1_set.union(st2_set)
        intersect = st1_set.intersection(st2_set)
        return float(len(intersect) / len(union))

    @staticmethod
    def list_overlap(x, y):
        x_set = set(x)
        intersection = x_set.intersection(y)
        return len(intersection)


def features_loader(file_to_load, corpus):
    if file_to_load is None:
        file = dp.ensure_file('features_{}_uqv.JSON'.format(corpus))
    else:
        file = dp.ensure_file(file_to_load)

    features_df = pd.read_json(file, dtype={'topic': str, 'qid': str})
    features_df.reset_index(drop=True, inplace=True)
    features_df.set_index(['topic', 'qid'], inplace=True)
    features_df.sort_values(['topic', 'qid'], axis=0, inplace=True)
    return features_df


def main(args):
    corpus = args.corpus
    generate = args.generate
    predict = args.predict
    queries_group = args.group
    file_to_load = args.load

    # # Debugging
    # testing_feat = QueryFeatureFactory('ROBUST')
    # norm_features_df = testing_feat.generate_features()
    # norm_features_df.reset_index().to_json('query_features_{}_uqv.JSON'.format(corpus))

    if generate:
        testing_feat = QueryFeatureFactory(corpus)
        norm_features_df = testing_feat.generate_features()
        norm_features_df.reset_index().to_json(f'{queries_group}_query_features_{corpus}_uqv.JSON')
    elif predict:
        for n in [5, 10, 25, 50, 100, 250, 500, 1000]:
            rbo_pred = QueryFeatureFactory(corpus, queries_group, n)
            rbo_pred.generate_rbo_predictions()
    elif file_to_load:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
