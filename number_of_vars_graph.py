import pandas as pd
import numpy as np
import dataparser as dp
from Timer.timer import Timer
from qpp_ref import run_calc_process
import argparse
import glob
import itertools
import multiprocessing as mp
import os
from functools import partial
from subprocess import run
from query_features import QueryFeatureFactory
from queries_pre_process import filter_n_top_queries, filter_n_low_queries, add_topic_to_qdf

parser = argparse.ArgumentParser(description='Results generator for QPP with Reference lists graphs',
                                 usage='',
                                 epilog='')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])

PREDICTORS_WO_QF = ['clarity', 'wig', 'nqc', 'smv', 'preret', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv']
PREDICTORS_QF = ['qf', 'uef/qf']
PREDICTORS = PREDICTORS_WO_QF + PREDICTORS_QF
NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500, 1000)


class GraphsFactory:
    def __init__(self, corpus):
        self.__set_paths(corpus)
        self.corpus = corpus
        self.queries_obj = dp.QueriesTextParser(self.queries_file)
        self.queries_obj.queries_df = add_topic_to_qdf(self.queries_obj.queries_df)
        self.ap_obj = dp.ResultsReader(self.ap_file, 'ap')
        self.max_variations = self.queries_obj.queries_df.groupby('topic').count().max()['qid']

    @classmethod
    def __set_paths(cls, corpus):
        cls.ap_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QLmap1000')
        cls.queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_title.txt')
        cls.results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res')
        cls.corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/logqlc.res')
        cls.rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/rsd/data')
        cls.data_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/data')
        cls.predictions_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/referenceLists/title')

    def create_query_files(self, n):
        for direction, func in {('asce', filter_n_low_queries), ('desc', filter_n_top_queries)}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direction}/queries')
            _file = f'{_dir}/queries_wo_title_{n}_vars.txt'
            _df = func(self.queries_obj.queries_df, self.ap_obj, n)
            _df[['qid', 'text']].to_csv(_file, sep=":", header=False, index=False)

    def generate_features(self, n):
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}/features')
            _feat_obj = QueryFeatureFactory(corpus=self.corpus, queries_group='title', vars_quantile='all',
                                            graphs=direct, n=n)
            _df = _feat_obj.generate_features()
            _df.reset_index().to_json(f'{_dir}/title_query_{n}_variations_features_{self.corpus}_uqv.JSON')

    def generate_sim_predictions(self, n):
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}')
            for k in NUMBER_OF_DOCS:
                sim_ref_pred = QueryFeatureFactory(self.corpus, queries_group='title', vars_quantile='all', rbo_top=k,
                                                   top_docs_overlap=k, graphs=direct, n=n)
                sim_ref_pred.generate_predictions()


def main(args):
    corpus = args.corpus

    #TODO:
    # Continue to qpp_ref next, make sure it knows how to use the new features format and queries
    # then need to write CV and save all the results in order to plot graphs

    # ap_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QLmap1000')
    #
    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_title.stemmed.txt')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/logqlc.res')
    # rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/rsd/data')

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries.stemmed.txt')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/logqlc.res')
    # rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/rsd/data')

    max_n = 20
    # corpus = 'ROBUST'

    testing = GraphsFactory(corpus)
    # testing.create_query_files(13)
    # testing.generate_features(1)
    # testing.generate_sim_predictions(1)

    for n in range(1, min(testing.max_variations, max_n) + 1):
        testing.create_query_files(n)

    cores = mp.cpu_count() - 1

    """The first run will generate the pkl files, all succeeding runs will load and use it"""
    testing.generate_features(1)
    with mp.Pool(processes=cores) as pool:
        pool.map(testing.generate_features, range(2, min(testing.max_variations, max_n) + 1))

    testing.generate_sim_predictions(1)
    with mp.Pool(processes=cores) as pool:
        pool.map(testing.generate_sim_predictions, range(2, min(testing.max_variations, max_n) + 1))


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
