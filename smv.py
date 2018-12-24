import argparse
import glob
import itertools
import multiprocessing as mp
import os
from functools import partial
from subprocess import run
from collections import defaultdict

import numpy as np
import pandas as pd

import dataparser as dp
from Timer.timer import Timer
from crossval import CrossValidation
from query_features import features_loader

parser = argparse.ArgumentParser(description='SMV predictor',
                                 usage='Input QL(q|d) scores and queries files',
                                 epilog='Prints the SMV predictor scores')

parser.add_argument('results', metavar='QL(q|d)_results_file', help='The QL results file for the documents scores')
parser.add_argument('corpus_scores', metavar='QLC', help='logqlc QL Corpus scores of the queries')
parser.add_argument('queries', metavar='queries_txt_file', default='data/ROBUST/queries.txt',
                    help='The queries txt file')
parser.add_argument('-d', '--docs', metavar='Docs', type=int, default=None, help='Number of documents')
# parser.add_argument('-c', '--corpus', metavar='fbDocs', default=None, help='Number of documents')

NUMBER_OF_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]


class SMV:
    """This class implements the QPP method as described in:
    'Query Performance Prediction By Considering Score Magnitude and Variance Together'
    The predictor is implemented to work with log(QL) scores (not -CE)"""

    def __init__(self, queries_obj: dp.QueriesXMLParser, results_obj: dp.ResultsReader,
                 corpus_scores_obj: dp.ResultsReader):
        self.qdb = queries_obj
        self.res_df = results_obj.data_df
        self.corpus_df = corpus_scores_obj.data_df
        self.predictions = defaultdict(float)

    def _calc_denominator(self, qid, num_docs):
        _score = self.corpus_df.loc[qid]['score']
        return _score

    def _calc_numerator(self, qid, num_docs):
        _scores = self.res_df.loc[qid]['docScore'].head(num_docs)
        _mean = _scores.mean()
        _scores_dev = _scores / _mean
        _ln_scores = abs(_scores_dev.apply(np.log))
        return (_scores * _ln_scores).sum()

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
    results_file = args.results
    corpus_scores_file = args.corpus_scores
    queries_file = args.queries
    number_of_docs = args.docs

    # corpus = 'ROBUST'

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.txt')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/logqlc.res')

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries.txt')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/logqlc.res')

    queries_obj = dp.QueriesXMLParser(queries_file)
    # queries_obj = dp.QueriesTextParser(queries_file)
    results_obj = dp.ResultsReader(results_file, 'trec')
    corpus_scores_obj = dp.ResultsReader(corpus_scores_file, 'predictions')

    predictor = SMV(queries_obj, results_obj, corpus_scores_obj)
    if number_of_docs:
        predictor.calc_results(number_of_docs)
    else:
        for n in NUMBER_OF_DOCS:
            predictor.calc_results(n)


if __name__ == '__main__':
    args = parser.parse_args()
    # overall_timer = Timer('Total runtime')
    main(args)
    # overall_timer.stop()
