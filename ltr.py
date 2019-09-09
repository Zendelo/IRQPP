"""This code will construct the features vectors for the learning process"""

import argparse
import glob
import multiprocessing as mp
import os
from functools import partial, reduce
import itertools as it
from collections import defaultdict

import numpy as np
import pandas as pd

import dataparser as dp
from RBO import rbo_dict
from Timer.timer import Timer


def jaccard_coefficient(st1: str, st2: str):
    st1_set = set(st1.split())
    st2_set = set(st2.split())
    union = st1_set.union(st2_set)
    intersect = st1_set.intersection(st2_set)
    return float(len(intersect) / len(union))


def list_overlap(x: list, y: list):
    list_len = max(len(x), len(y))
    x_set = set(x)
    intersection = x_set.intersection(y)
    return float(len(intersection) / list_len)


class FeaturesFactory:
    def __init__(self, corpus, predictor, **kwargs):
        self.corpus = corpus
        self.predictor = predictor
        # self.ql_results_file = None
        # self.queries_txt_file = None
        # self.predictions_output_dir = None
        # self.pkl_dir = None
        self.__set_paths()
        self.queries_obj = dp.QueriesTextParser(self.queries_txt_file, kind='uqv')
        self.queries_obj.queries_df = dp.add_topic_to_qdf(self.queries_obj.queries_df).set_index('qid')
        self.features_df = self.initialize_features_df()
        self.ql_results_obj = dp.ResultsReader(self.ql_results_file, 'trec')

    def __set_paths(self):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        _corpus_res_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{self.corpus}')
        _corpus_dat_dir = dp.ensure_dir(f'~/QppUqvProj/data/{self.corpus}')
        self.ql_results_file = dp.ensure_file(f'{_corpus_res_dir}/test/raw/QL.res')
        self.queries_txt_file = dp.ensure_file(f'{_corpus_dat_dir}/queries_{self.corpus}_UQV_full.stemmed.txt')
        self.predictions_dir = dp.ensure_dir(f'{_corpus_res_dir}/uqvPredictions/raw/{self.predictor}')
        self.pkl_dir = dp.ensure_dir(f'{_corpus_res_dir}/test/raw/pkl_files/')

    def initialize_features_df(self):
        """This method will create a df with 3 columns: topic, q1, q2. with all the possible combinations of queries"""
        _list = []
        for topic, query_vars in self.queries_obj.query_vars.items():
            for q1, q2 in it.combinations(query_vars, 2):
                _list.append({'topic': topic, 'q1': q1, 'q2': q2})
        return pd.DataFrame(_list)

    def calc_features(self, overlap_size=100, rbo_size=100):
        features_df = self.features_df.set_index(['topic', 'q1', 'q2']).assign(jac=None, overlap=None, rbo=None)
        for topic, (q1, q2) in self.features_df.set_index('topic').loc[:, ['q1', 'q2']].iterrows():
            q1_txt = self.queries_obj.queries_df.loc[q1].text
            q2_txt = self.queries_obj.queries_df.loc[q2].text
            jac_sim = jaccard_coefficient(q1_txt, q2_txt)

            over_sim = list_overlap(self.ql_results_obj.get_docs_by_qid(q1, overlap_size),
                                    self.ql_results_obj.get_docs_by_qid(q2, overlap_size))

            rbo_sim = rbo_dict(self.ql_results_obj.get_res_dict_by_qid(q1, rbo_size),
                               self.ql_results_obj.get_res_dict_by_qid(q2, rbo_size))['min']

            features_df.loc[topic, q1, q2] = [jac_sim, over_sim, rbo_sim]
        features_df.rename({'overlap': f'overlap_{overlap_size}', 'rbo': f'rbo_{rbo_size}'})
        return features_df

    def load_similarity_features_df(self):
        sim_features_file = f'{self.pkl_dir}/similarity_features_df.pkl'
        try:
            df_file = dp.ensure_file(sim_features_file)
            df = pd.read_pickle(df_file)
        except AssertionError:
            print(f'-- Failed loading {sim_features_file}, will generate and save --')
            df = self.calc_features_parallel()
            df.to_pickle(sim_features_file)
        return df

    def calc_features_parallel(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            result = pool.starmap(self.calc_features, ((i, i) for i in {5, 10, 25, 50, 100, 250, 500}))
        pool.close()
        return pd.concat(result, axis=0)


if __name__ == '__main__':
    # Debugging
    # corpus = 'ClueWeb12B'
    corpus = 'ROBUST'
    predictor = 'wig'
    print('\n------+++^+++------ Debugging !! ------+++^+++------\n')

    tes = FeaturesFactory(corpus, predictor)
    # tes.calc_features()
    print(tes.load_similarity_features_df())
