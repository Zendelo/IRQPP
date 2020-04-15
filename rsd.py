import argparse
import glob
import itertools
import multiprocessing as mp
import pickle
import random
from collections import defaultdict
from functools import partial
from math import sqrt

import numpy as np
import pandas as pd
import scipy.stats as st

from qpputils import dataparser as dp
from Timer import Timer

parser = argparse.ArgumentParser(description='RSD(wig) predictor',
                                 usage='Change the paths in the code in order to predict UQV/Base queries',
                                 epilog='Generates the RSD predictor scores')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])

NUMBER_OF_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_LENGTH = [5, 10, 25, 50, 100, 250, 500]


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
                 corpus_scores_obj: dp.ResultsReader, rm_probabilities_df, corpus, uqv=False, load_cache=True):
        self.qdb = queries_obj
        self.res_df = results_obj.data_df
        self.corpus_df = corpus_scores_obj.data_df
        self.rm_prob_df = rm_probabilities_df
        # self.predictions = defaultdict(float)
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
        # self.raw_scores_dict = self.calc_raw_scores()

    def __save_new_dictionary(self):
        """This method saves the sampled lists dictionary into a pickle file"""
        _dir = dp.ensure_dir(self._pkl_dir)
        with open(f'{_dir}/{self.docs_num}_docs_lists_length_{self.list_length}_dict.pkl', 'wb') as handle:
            pickle.dump(self.docs_lists_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    def __calc_raw_sigma_sq(self, qid, docs_lists):
        """This method implements the calculation of the estimator sigma_{s|q} as it's defined in section 2.1"""
        df = self.res_df.loc[qid]
        _scores_list = []
        corpus_score = self.corpus_df.loc[qid].score
        for _list in docs_lists:
            # This notation is a hint type. i.e. _df is of type pd.Series
            _df: pd.Series = df.loc[df['docID'].isin(_list)]['docScore']
            list_length = len(_df)
            scores_sum = _df.sum()
            # Pandas unbiased variance function
            scores_var = _df.var()
            wig_weight = max(0, (scores_sum / list_length) - corpus_score)
            _scores_list.append(wig_weight * scores_var)
        return sqrt(sum(_scores_list))

    def calc_nperp(self):
        """This method implements the calculation of nperp(q|R) as it's defined in section 2.4"""
        entropy_df = self.rm_prob_df.groupby('qid').aggregate(st.entropy, base=2).fillna(0)
        n_q_df = self.rm_prob_df.groupby('qid').count()
        nperp_df = entropy_df.apply(lambda x: 2 ** x) / n_q_df.apply(lambda x: 2 ** np.log2(x))
        return nperp_df

    def calc_raw_scores(self):
        _scores_dict = {}
        for qid, docs_lists in self.docs_lists_dict.items():
            _scores_dict[qid] = self.__calc_raw_sigma_sq(qid, docs_lists)
        return pd.DataFrame.from_dict(_scores_dict, orient='index',
                                      columns=[f'score-{self.docs_num}+{self.list_length}'])

    def calc_normalized_scores(self):
        """This method implements the calculation of the normalized scores as it's defined in section 2.4 eq (2)"""
        nperp_df = self.calc_nperp()
        raw_scores_df = self.calc_raw_scores()
        raw_div_corp_df = raw_scores_df.div(self.corpus_df['score'].abs(), axis=0, level='qid')
        # raw_div_corp_df.index.rename('qid', inplace=True)
        final_scores_df = nperp_df.multiply(raw_div_corp_df.iloc[:, 0], axis=0, level='qid')
        return final_scores_df


def read_rm_prob_files(data_dir, number_of_docs):
    """The function creates a DF from files, the probabilities are p(w|RM1) for all query words
    If a query term doesn't appear in the file, it's implies p(w|R)=0"""
    data_files = glob.glob(f'{data_dir}/probabilities-{number_of_docs}+*')
    _list = []
    for _file in data_files:
        _col = f'{_file.rsplit("/")[-1].rsplit("-")[-1]}'
        _df = pd.read_table(_file, names=['qid', 'term', _col], sep=' ')
        _df = _df.astype({'qid': str}).set_index(['qid', 'term'])
        _list.append(_df)
    return pd.concat(_list, axis=1).fillna(0)


def run_predictions(number_of_docs, list_length, queries_obj, results_obj, corpus_scores_obj, rm_probabilities_dir,
                    corpus, uqv, load_cache=True):
    rm_prob_df = read_rm_prob_files(rm_probabilities_dir, number_of_docs)
    predictor = RSD(number_of_docs=number_of_docs, list_length=list_length, queries_obj=queries_obj,
                    results_obj=results_obj, corpus_scores_obj=corpus_scores_obj, rm_probabilities_df=rm_prob_df,
                    corpus=corpus, uqv=uqv, load_cache=load_cache)
    df = predictor.calc_normalized_scores()
    if uqv:
        _dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/rsd/predictions')
    else:
        _dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/rsd/predictions')
    for col in df:
        file_name = f'{_dir}/predictions-{col}+{list_length}'
        df[col].to_csv(file_name, sep=" ", header=False, index=True, float_format='%f')


def main(args):
    corpus = args.corpus

    queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.xml')
    results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QL.res')
    corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/logqlc.res')
    rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/rsd/data')

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries.xml')
    # results_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/QL.res')
    # corpus_scores_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/logqlc.res')
    # rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/rsd/data')

    queries_obj = dp.QueriesXMLParser(queries_file)
    results_obj = dp.ResultsReader(results_file, 'trec')
    corpus_scores_obj = dp.ResultsReader(corpus_scores_file, 'predictions')

    cores = mp.cpu_count() - 1
    uqv = True if 'uqv' in queries_file.split('/')[-1].lower() else False

    with mp.Pool(processes=cores) as pool:
        predictor = pool.starmap(
            partial(run_predictions, queries_obj=queries_obj, results_obj=results_obj,
                    corpus_scores_obj=corpus_scores_obj, rm_probabilities_dir=rm_probabilities_dir,
                    corpus=corpus, uqv=uqv, load_cache=True), itertools.product(NUMBER_OF_DOCS, LIST_LENGTH))


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
