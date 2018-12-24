import argparse
import glob
import itertools
import multiprocessing as mp
import os
import random
from functools import partial
from subprocess import run
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from Crypto.Util import number

import dataparser as dp
from Timer.timer import Timer
from crossval import CrossValidation
from query_features import features_loader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RSD(wig) predictor',
                                 usage='Input QL(q|d) scores and queries files',
                                 epilog='Generates the RSD predictor scores')

# parser.add_argument('results', metavar='QL(q|d)_results_file', help='The QL results file for the documents scores')
# parser.add_argument('corpus_scores', metavar='QLC', help='logqlc QL Corpus scores of the queries')
# parser.add_argument('queries', metavar='queries_txt_file', default='data/ROBUST/queries.txt',
#                     help='The queries txt file')
# parser.add_argument('-d', '--docs', metavar='Docs', type=int, default=None, help='Number of documents')
parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])

NUMBER_OF_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_LENGTH = [30, 50, 100, 150, 200]


def random_sampling(list_length, df):
    """This function implements rank biased sampling of l documents as described in section 2.2"""
    docs_list = []
    df = df.assign(available=True)
    while len(docs_list) < list_length:
        for rank in itertools.compress(df.index, df['available']):
            u = random.random()
            if df.loc[rank, 'prob'] >= u:
                docs_list.append(df.loc[rank, 'docID'])
                df.loc[rank, 'available'] = False
            if len(docs_list) >= list_length:
                break
    return docs_list


def generate_probabilities_sr(k):
    """This function implements the probability distribution of documents as mentioned in section 2.2"""
    _dict = {}
    for i in range(1, k + 1):
        _dict[i] = (2 * (k + 1 - i)) / (k * (k + 1))
    p_r = pd.Series(_dict)
    return p_r[::-1].cumsum()[::-1]


class RSD:
    """This class implements the QPP method as described in:
    'Robust Standard Deviation Estimation for query Performance Prediction'
    The predictor is implemented to work with log(QL) scores (not -CE)"""

    def __init__(self, number_of_docs, list_length, queries_obj: dp.QueriesXMLParser, results_obj: dp.ResultsReader,
                 corpus_scores_obj: dp.ResultsReader, corpus, uqv=False, load_cache=True):
        self.qdb = queries_obj
        self.res_df = results_obj.data_df
        self.corpus_df = corpus_scores_obj.data_df
        self.predictions = defaultdict(float)
        # pd.Series the index is a rank of a doc, value is its probability
        self.probabilities_sr = generate_probabilities_sr(number_of_docs)
        self.docs_num = number_of_docs
        self.list_length = list_length
        if uqv:
            self._pkl_dir = f'~/QppUqvProj/Results/{corpus}/test/rsd/pkl_files/uqv/'
        else:
            self._pkl_dir = f'~/QppUqvProj/Results/{corpus}/test/rsd/pkl_files/basic/'
        if load_cache:
            try:
                # Will try loading a dictionary, if fails will generate and save a new one
                file_to_load = dp.ensure_file(
                    f'{self._pkl_dir}/{self.docs_num}_docs_lists_length_{self.list_length}_dict.pkl')
                with open(file_to_load, 'rb') as handle:
                    self.docs_lists_dict = pickle.load(handle)
            except AssertionError:
                print(f'\nFailed to load {self.docs_num}_docs_lists_length_{self.list_length}_dict.pkl')
                print(f'Will generate the lists with {self.docs_num} docs and {self.list_length} list len and save')
                self.docs_lists_dict = self.generate_sampled_lists(list_length)
                self.__save_new_dictionary()
        else:
            self.docs_lists_dict = self.generate_sampled_lists(list_length)
            self.__save_new_dictionary()

    def __save_new_dictionary(self):
        """This method saves the sampled lists dictionary into a pickle file"""
        _dir = dp.ensure_dir(self._pkl_dir)
        with open(f'{_dir}/{self.docs_num}_docs_lists_length_{self.list_length}_dict.pkl', 'wb') as handle:
            pickle.dump(self.docs_lists_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _calc_denominator(self, qid, num_docs):
        _score = self.corpus_df.loc[qid]['score']
        return _score

    def _calc_numerator(self, qid, num_docs):
        _scores = self.res_df.loc[qid]['docScore'].head(num_docs)
        _mean = _scores.mean()
        _scores_dev = _scores / _mean
        _ln_scores = abs(_scores_dev.apply(np.log))
        return (_scores * _ln_scores).sum()

    def __full_sample(self):
        _dict = defaultdict(list)
        for qid, _df in self.res_df.groupby('qid'):
            df = _df.head(self.docs_num).set_index('docRank')
            _dict[qid].append(df['docID'].tolist())
        return _dict

    def generate_sampled_lists(self, list_length):
        docs_lists_dict = defaultdict(list)
        if list_length >= self.docs_num:
            return self.__full_sample()
        for qid, _df in self.res_df.groupby('qid'):
            df = _df.head(self.docs_num).set_index('docRank')
            # Check if a specific query has less results than the hyper parameter docs_num
            if len(df) < self.docs_num:
                _probabilities_sr = generate_probabilities_sr(len(df))
            else:
                _probabilities_sr = self.probabilities_sr
            list_length = min(list_length, self.docs_num, len(df))
            # df = _df.head(self.docs_num)[['docID', 'docRank']]
            df.insert(loc=0, column='available', value=True)
            # df.set_index('docRank', drop=True, inplace=True)
            df.loc[_probabilities_sr.index, 'prob'] = _probabilities_sr
            for _ in range(100):
                _docs_list = random_sampling(list_length, df)
                docs_lists_dict[qid].append(_docs_list)
        return docs_lists_dict

    def calc_results(self, number_of_docs):
        for qid in self.qdb.query_length:
            _denominator = self._calc_denominator(qid, number_of_docs) * number_of_docs
            _numerator = self._calc_numerator(qid, number_of_docs)
            _score = _numerator / _denominator
            self.predictions[qid] = _score
            print('{} {:f}'.format(qid, _score))
        # predictions_df = pd.Series(self.predictions)
        # predictions_df.to_json('wig-predictions-{}.res'.format(number_of_docs))


def main(args):
    # results_file = args.results
    # corpus_scores_file = args.corpus_scores
    # queries_file = args.queries
    # number_of_docs = args.docs
    corpus = args.corpus

    # corpus = 'ROBUST'

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.xml')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/logqlc.res')

    queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries.txt')
    results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/QL.res')
    corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/logqlc.res')

    queries_obj = dp.QueriesXMLParser(queries_file)
    # queries_obj = dp.QueriesTextParser(queries_file)
    results_obj = dp.ResultsReader(results_file, 'trec')
    corpus_scores_obj = dp.ResultsReader(corpus_scores_file, 'predictions')

    cores = mp.cpu_count() - 1
    uqv = True if 'uqv' in queries_file.lower() else False

    with mp.Pool(processes=cores) as pool:
        predictor = pool.starmap(
            partial(RSD, queries_obj=queries_obj, results_obj=results_obj, corpus_scores_obj=corpus_scores_obj,
                    corpus=corpus, uqv=uqv, load_cache=True), itertools.product(NUMBER_OF_DOCS, LIST_LENGTH))


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
