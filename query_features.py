import argparse
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd

import dataparser as dp
from RBO import rbo_dict
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])

parser.add_argument('-g', '--group', help='group of queries to predict',
                    choices=['top', 'low', 'medh', 'medl', 'title'])
parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default=None,
                    choices=['all', 'low', 'med', 'top'])
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')
parser.add_argument('--generate', help="generate new features file", action="store_true")
parser.add_argument('--predict', help="generate new predictions", action="store_true")

NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500, 1000)


def jaccard_coefficient(st1: str, st2: str):
    st1_set = set(st1.split())
    st2_set = set(st2.split())
    union = st1_set.union(st2_set)
    intersect = st1_set.intersection(st2_set)
    return float(len(intersect) / len(union))


def list_overlap(x, y):
    x_set = set(x)
    intersection = x_set.intersection(y)
    return len(intersection)


class QueryFeatureFactory:
    def __init__(self, corpus, queries_group, vars_quantile, rbo_top=100, top_docs_overlap=10):
        self.top_docs_overlap = top_docs_overlap
        self.rbo_top = rbo_top
        self.corpus = corpus
        self.queries_group = queries_group
        self.__set_paths(corpus, queries_group, vars_quantile)
        _raw_res_data = dp.ResultsReader(self.results_file, 'trec')
        if queries_group == 'title':
            _title_res_data = dp.ResultsReader(self.title_res_file, 'trec')
            self.prediction_queries_res_data = _title_res_data
        else:
            self.prediction_queries_res_data = _raw_res_data
        self.queries_data = dp.QueriesTextParser(self.queries_full_file, 'uqv')
        self.topics_data = dp.QueriesTextParser(self.queries_topic_file)
        # Uncomment the next lines if you want to write the basic results of the topic queries.
        # write_basic_results(self.prediction_queries_res_data.data_df.loc[self.topics_data.queries_df['qid']], corpus,
        #                     queries_group)
        # exit()
        self.variations_data = dp.QueriesTextParser(self.queries_variations_file, 'uqv')
        # _var_scores_df.loc[_var_scores_df['qid'].isin(_vars_list)]
        self.raw_res_data = _raw_res_data

        self.fused_data = dp.ResultsReader(self.fused_results_file, 'trec')
        self.query_vars = self.queries_data.query_vars

    @classmethod
    def __set_paths(cls, corpus, qgroup, vars_quantile):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        # cls.predictor = predictor
        _corpus_res_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}')
        _corpus_dat_dir = dp.ensure_dir(f'~/QppUqvProj/data/{corpus}')

        _results_file = f'{_corpus_res_dir}/test/raw/QL.res'
        cls.results_file = os.path.normpath(_results_file)
        dp.ensure_file(cls.results_file)

        _title_results_file = f'{_corpus_res_dir}/test/basic/QL.res'
        cls.title_res_file = os.path.normpath(_title_results_file)
        dp.ensure_file(cls.title_res_file)

        if vars_quantile == 'all':
            _queries_full_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_full.txt'
        else:
            _queries_full_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_{vars_quantile}_variants.txt'

        cls.queries_full_file = dp.ensure_file(_queries_full_file)

        # The variations file is used in the filter function - it consists of all the vars w/o the query at hand
        _queries_variations_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_wo_{qgroup}.txt'
        cls.queries_variations_file = dp.ensure_file(_queries_variations_file)

        _queries_topic_file = f'{_corpus_dat_dir}/queries_{corpus}_{qgroup}.txt'
        cls.queries_topic_file = dp.ensure_file(_queries_topic_file)

        _fused_results_file = f'{_corpus_res_dir}/test/fusion/QL.res'
        cls.fused_results_file = dp.ensure_file(_fused_results_file)

        cls.output_dir = dp.ensure_dir(f'{_corpus_res_dir}/test/raw/')

        _predictions_out = f'{_corpus_res_dir}/uqvPredictions/referenceLists/{qgroup}/{vars_quantile}_vars/sim_as_pred/'
        cls.predictions_output_dir = dp.ensure_dir(_predictions_out)

    def _calc_features(self):
        _dict = {'topic': [], 'qid': [], 'Jac_coefficient': [], 'Top_10_Docs_overlap': [], 'RBO_EXT_100': [],
                 'RBO_FUSED_EXT_100': []}

        for topic in self.topics_data.queries_dict.keys():
            _topic = topic.split('-')[0]
            q_vars = self.query_vars.get(_topic)
            _dict['topic'] += [topic] * len(q_vars)
            res_dict = self.fused_data.get_res_dict_by_qid(_topic, top=self.rbo_top)
            topic_txt = self.topics_data.get_qid_txt(topic)
            topics_top_list = self.prediction_queries_res_data.get_docs_by_qid(topic, self.top_docs_overlap)
            # topics_top_list = self.title_res_data.get_docs_by_qid(topic, 25)
            topic_results_list = self.prediction_queries_res_data.get_res_dict_by_qid(topic, top=self.rbo_top)

            for var in q_vars:
                var_txt = self.queries_data.get_qid_txt(var)
                jc = jaccard_coefficient(topic_txt, var_txt)

                var_top_list = self.raw_res_data.get_docs_by_qid(var, self.top_docs_overlap)
                # var_top_list = self.raw_res_data.get_docs_by_qid(var, 25)
                docs_overlap = list_overlap(topics_top_list, var_top_list)

                # All RBO values are rounded to 10 decimal digits, to avoid float overflow
                var_results_list = self.raw_res_data.get_res_dict_by_qid(var, top=self.rbo_top)
                _rbo_scores_dict = rbo_dict(topic_results_list, var_results_list, p=0.95)
                rbo_ext_score = np.around(_rbo_scores_dict['ext'], 10)

                _fused_rbo_scores_dict = rbo_dict(res_dict, var_results_list, p=0.95)
                _rbo_fused_ext_score = np.around(_fused_rbo_scores_dict['ext'], 10)

                _dict['qid'] += [var]
                _dict['Jac_coefficient'] += [jc]
                _dict['Top_10_Docs_overlap'] += [docs_overlap]
                _dict['RBO_EXT_100'] += [rbo_ext_score]
                _dict['RBO_FUSED_EXT_100'] += [_rbo_fused_ext_score]

        _df = pd.DataFrame.from_dict(_dict)
        # _df.set_index(['topic', 'qid'], inplace=True)
        return _df

    def _filter_queries(self, df):
        # return df[df['Jac_coefficient'] != 1]
        return df.loc[df['qid'].isin(self.variations_data.queries_df['qid'])]

    def _soft_max_scores(self, df):
        _df = self._filter_queries(df)
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
        _df = df
        # filter only variations different from original query
        # _df = self._filter_queries(df)
        z_n = _df.groupby(['topic']).sum()
        # All nan values will be filled with 0
        norm_df = (_df.groupby(['topic', 'qid']).sum() / z_n).fillna(0)
        return norm_df

    def save_predictions(self, df: pd.DataFrame):
        _df = self._filter_queries(df)
        _df = df.groupby('topic').mean()
        _df = dp.convert_vid_to_qid(_df)
        _rboP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/rboP/predictions')
        _FrboP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/FrboP/predictions')
        _topDocsP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/topDocsP/predictions')
        _jcP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/jcP/predictions')

        _df['RBO_EXT_100'].to_csv(f'{_rboP_dir}/predictions-{self.rbo_top}', sep=' ')
        _df['RBO_FUSED_EXT_100'].to_csv(f'{_FrboP_dir}/predictions-{self.rbo_top}', sep=' ')
        _df['Top_10_Docs_overlap'].to_csv(f'{_topDocsP_dir}/predictions-{self.top_docs_overlap}', sep=' ')
        _df['Jac_coefficient'].to_csv(f'{_jcP_dir}/predictions-{self.rbo_top}', sep=' ')

    def generate_features(self):
        _df = self._calc_features()
        return _df
        # return self._soft_max_scores(_df)
        # return self._sum_scores(_df)
        # return self._average_scores(_df)
        # return self._max_norm_scores(_df)

    def generate_predictions(self):
        _df = self._calc_features()
        self.save_predictions(_df)


def features_loader(file_to_load, corpus):
    if file_to_load is None:
        file = dp.ensure_file('features_{}_uqv.JSON'.format(corpus))
    else:
        file = dp.ensure_file(file_to_load)

    features_df = pd.read_json(file, dtype={'topic': str, 'qid': str})
    features_df.reset_index(drop=True, inplace=True)
    features_df.set_index(['topic', 'qid'], inplace=True)
    features_df.rename(index=lambda x: x.split('-')[0], level=0, inplace=True)
    features_df.sort_values(['topic', 'qid'], axis=0, inplace=True)
    return features_df


def write_basic_results(df: pd.DataFrame, corpus, qgroup):
    """The function is used to save basic predictions of a given queries set"""
    _df = dp.convert_vid_to_qid(df)
    _df.insert(loc=0, column='trec_Q0', value='Q0')
    _df.insert(loc=4, column='trec_indri', value='indri')
    _file_path = f'~/QppUqvProj/Results/{corpus}/test/ref/QL_{qgroup}.res'
    # dp.ensure_dir(os.path.normpath(os.path.expanduser(_file_path)))
    _df.to_csv(_file_path, sep=" ", header=False, index=True)


def run_predictions_process(n, corpus, queries_group, quantile):
    sim_ref_pred = QueryFeatureFactory(corpus, queries_group, quantile, rbo_top=n, top_docs_overlap=n)
    sim_ref_pred.generate_predictions()
    return sim_ref_pred


def main(args):
    corpus = args.corpus
    generate = args.generate
    predict = args.predict
    queries_group = args.group
    file_to_load = args.load
    quantile = args.quantile

    # # Debugging
    # testing_feat = QueryFeatureFactory('ROBUST')
    # norm_features_df = testing_feat.generate_features()
    # norm_features_df.reset_index().to_json('query_features_{}_uqv.JSON'.format(corpus))

    cores = mp.cpu_count() - 1

    if generate:
        testing_feat = QueryFeatureFactory(corpus, queries_group, quantile)
        norm_features_df = testing_feat.generate_features()

        _path = f'~/QppUqvProj/Results/{corpus}/test/ref'
        _path = dp.ensure_dir(_path)
        norm_features_df.reset_index().to_json(
            f'{_path}/{queries_group}_query_{quantile}_variations_features_{corpus}_uqv.JSON')
    elif predict:
        with mp.Pool(processes=cores) as pool:
            sim_ref_pred = pool.map(
                partial(run_predictions_process, corpus=corpus, queries_group=queries_group, quantile=quantile),
                NUMBER_OF_DOCS)

    elif file_to_load:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
