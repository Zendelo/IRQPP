import argparse
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations_with_replacement
from functools import partial

import numpy as np
import pandas as pd

import dataparser as dp
from RBO import rbo_dict
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Features for PageRank UQV query variations Generator',
                                 usage='python3.7 features.py -q queries.txt -c CORPUS',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])

# parser.add_argument('-g', '--group', help='group of queries to predict',
#                     choices=['top', 'low', 'medh', 'medl', 'title'])
# parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default=None,
#                     choices=['all', 'low', 'med', 'top'])
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')
parser.add_argument('--generate', help="generate new features file", action="store_true")
parser.add_argument('--predict', help="generate new predictions", action="store_true")


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
    def __init__(self, corpus, rbo_top=100, top_docs_overlap=10):
        self.top_docs_overlap = top_docs_overlap
        self.rbo_top = rbo_top
        self.corpus = corpus
        # self.queries_group = queries_group
        self.__set_paths(corpus)
        _raw_res_data = dp.ResultsReader(self.results_file, 'trec')
        # if queries_group == 'title':
        #     _title_res_data = dp.ResultsReader(self.title_res_file, 'trec')
        #     self.prediction_queries_res_data = _title_res_data
        # else:
        #     self.prediction_queries_res_data = _raw_res_data
        self.queries_data = dp.QueriesTextParser(self.queries_full_file, 'uqv')
        # self.topics_data = dp.QueriesTextParser(self.queries_topic_file)
        #
        # self.variations_data = dp.QueriesTextParser(self.queries_variations_file, 'uqv')
        # # _var_scores_df.loc[_var_scores_df['qid'].isin(_vars_list)]
        self.raw_res_data = _raw_res_data
        #
        self.fused_data = dp.ResultsReader(self.fused_results_file, 'trec')
        self.query_vars = self.queries_data.query_vars
        self.features_index = self._create_query_var_pairs()

    @classmethod
    def __set_paths(cls, corpus):
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

        _queries_full_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_full.txt'

        cls.queries_full_file = dp.ensure_file(_queries_full_file)

        _fused_results_file = f'{_corpus_res_dir}/test/fusion/QL.res'
        cls.fused_results_file = dp.ensure_file(_fused_results_file)

        cls.output_dir = dp.ensure_dir(f'{_corpus_res_dir}/test/raw/')

    def _create_query_var_pairs(self):
        """This method returns a dictionary where for each key (topic-qid) the value is a list of all possible
        pair combinations of the variants  {'qid': [(q1, q1), (q1, q2), (q2, q2)]}"""
        return {qid: list(combinations_with_replacement(variations, 2)) for qid, variations in self.query_vars.items()}

    def _calc_features(self):
        _dict = {'topic': [], 'qid': [], 'Jac_coefficient': [], f'Top_{self.top_docs_overlap}_Docs_overlap': [],
                 f'RBO_EXT_{self.rbo_top}': [], f'RBO_FUSED_EXT_{self.rbo_top}': []}
        for topic, pairs in self.features_index.items():
            # number of combination with replacement is n(n+1)/2
            _dict['topic'] += [topic] * (2 * len(pairs) - len(self.query_vars[topic]))
            fused_res_dict = self.fused_data.get_res_dict_by_qid(topic, top=100)
            for q1, q2 in pairs:
                txt1 = self.queries_data.get_qid_txt(q1)
                txt2 = self.queries_data.get_qid_txt(q2)
                jc = jaccard_coefficient(txt1, txt2)

                l1 = self.raw_res_data.get_docs_by_qid(q1, self.top_docs_overlap)
                l2 = self.raw_res_data.get_docs_by_qid(q2, self.top_docs_overlap)
                docs_overlap = list_overlap(l1, l2)

                # All RBO values are rounded to 10 decimal digits, to avoid float overflow
                q1_results_dict = self.raw_res_data.get_res_dict_by_qid(q1, top=self.rbo_top)
                q2_results_dict = self.raw_res_data.get_res_dict_by_qid(q2, top=self.rbo_top)
                _rbo_scores_dict = rbo_dict(q1_results_dict, q2_results_dict, p=0.95)
                rbo_ext_score = np.around(_rbo_scores_dict['ext'], 10)

                _q1_fused_rbo_scores_dict = rbo_dict(fused_res_dict, q1_results_dict, p=0.95)
                _q1_rbo_fused_ext_score = np.around(_q1_fused_rbo_scores_dict['ext'], 10)

                _q2_fused_rbo_scores_dict = rbo_dict(fused_res_dict, q2_results_dict, p=0.95)
                _q2_rbo_fused_ext_score = np.around(_q2_fused_rbo_scores_dict['ext'], 10)

                def _save_to_dict(q_1, q_2):
                    _dict['qid'] += [(q_1, q_2)]
                    _dict['Jac_coefficient'] += [jc]
                    _dict[f'Top_{self.top_docs_overlap}_Docs_overlap'] += [docs_overlap]
                    _dict[f'RBO_EXT_{self.rbo_top}'] += [rbo_ext_score]
                    # The RBO-F feature in that case for edge (q1, q2) will be the RBO similarity of q2 to fused list
                    _dict[f'RBO_FUSED_EXT_{self.rbo_top}'] += [_q2_rbo_fused_ext_score]
                if q1 == q2:
                    _save_to_dict(q1, q2)
                else:
                    _save_to_dict(q1, q2)
                    _save_to_dict(q2, q1)

        _df = pd.DataFrame.from_dict(_dict)
        _df.set_index(['topic', 'qid'], inplace=True)
        _df.to_pickle(f'{self.corpus}_raw_PageRank_Features.pkl')
        print(_df)
        return _df

    def _sum_scores(self, df):
        _exp_df = df.apply(np.exp)
        # For debugging purposes

        z_n = df.groupby(['topic']).sum()
        z_e = _exp_df.groupby(['topic']).sum()

        norm_df = (df.groupby(['topic', 'qid']).sum() / z_n)
        softmax_df = (_exp_df.groupby(['topic', 'qid']).sum() / z_e)
        # For debugging purposes
        return norm_df

    def generate_features(self):
        _df = self._calc_features()
        return self._sum_scores(_df)


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


def main(args):
    corpus = args.corpus
    generate = args.generate
    # predict = args.predict
    # queries_group = args.group
    file_to_load = args.load
    # quantile = args.quantile

    # # Debugging
    # print('------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------')
    # testing_feat = QueryFeatureFactory('ROBUST')
    # norm_features_df = testing_feat.generate_features()
    # # norm_features_df.reset_index().to_json('query_features_{}_uqv.JSON'.format(corpus))

    cores = mp.cpu_count() - 1

    if generate:
        testing_feat = QueryFeatureFactory(corpus)
        norm_features_df = testing_feat.generate_features()

        _path = f'~/QppUqvProj/Results/{corpus}/test/pageRank'
        _path = dp.ensure_dir(_path)
        norm_features_df.reset_index().to_json(f'{_path}/PageRank_Features.JSON')

    elif file_to_load:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
