#! /usr/bin/env python

import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd

NUMBER_OF_DOCS = [5, 10, 25, 50, 100]

parser = argparse.ArgumentParser(description='UEF predictor',
                                 usage='Input 2 lists of scores',
                                 epilog='Prints the UEF predictor scores')

parser.add_argument('list1', metavar='QL_results_file', help='The original QL results file for the documents scores')
parser.add_argument('list2', metavar='RM1_results_file', help='The re-ranked list with RM1')
parser.add_argument('list3', metavar='predictor_results_file', help='The predictor scores list')
parser.add_argument('-d', '--docs', metavar='K', default=5, type=int, help='Number of k top documents')


# parser.add_argument('-d', '--docs', metavar='KDocs', default=20, help='Number of K top documents')

class DataReader:
    def __init__(self, data: str, file_type):
        """
        :param data: results res
        :param file_type: 'result' for predictor results res or 'ap' for ap results res
        """
        self.file_type = file_type
        self.data = data
        self.__number_of_col = self.__check_number_of_col()
        if self.file_type == 'result':
            assert self.__number_of_col == 2 or self.__number_of_col == 4, 'Wrong File format'
            self.data_df = self.__read_results_data_2() if self.__number_of_col == 2 else self.__read_results_data_4()
        elif self.file_type == 'ap':
            self.data_df = self.__read_ap_data_2()

    def __check_number_of_col(self):
        with open(self.data) as f:
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
            first_row = next(reader)
            num_cols = len(first_row)
        return int(num_cols)

    def __read_results_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'score'],
                                dtype={'qid': str, 'score': np.float64})
        return data_df

    def __read_ap_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid AP'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'ap'],
                                dtype={'qid': str, 'ap': np.float64})
        return data_df

    def __read_results_data_4(self):
        """Assuming data is a res with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': str, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        return data_df


class ResultsReader:
    def __init__(self, results_file):
        self.results = results_file
        self.results_df = self.__read_file()

    def __read_file(self):
        return pd.read_table(self.results, delim_whitespace=True, header=None, index_col=[0],
                             names=['qid', 'Q0', 'docID', 'docRank', 'docScore', 'ind'],
                             dtype={'qid': str, 'Q0': str, 'docID': str, 'docRank': int, 'docScore': float,
                                    'ind': str})


class UEF:
    def __init__(self, original_list, modified_list, number_of_docs, method='pearson'):
        self.number_of_docs = number_of_docs
        self.method = method
        self.orig_list_df = ResultsReader(original_list).results_df
        self.mod_list_df = ResultsReader(modified_list).results_df
        self.predictions = defaultdict(float)
        self.queries = self.__generate_queries()

    def __generate_queries(self):
        _qids = set(self.mod_list_df.index)
        return tuple(sorted(_qids))

    def __create_docs_sets(self, df, k):
        _docs_dict = defaultdict(pd.DataFrame)
        for qid in self.queries:
            _temp_df = df.loc[qid].head(k)[['docID', 'docScore']]
            _docs_dict[qid] = _temp_df.set_index('docID')
        return _docs_dict

    def __calc_similarity(self):
        _sim_dict = defaultdict(float)
        orig_docs_dict = self.__create_docs_sets(self.orig_list_df, self.number_of_docs)
        mod_docs_dict = self.__create_docs_sets(self.mod_list_df, self.number_of_docs)
        for qid in self.queries:
            _sim = orig_docs_dict[qid]['docScore'].corr(mod_docs_dict[qid]['docScore'], method=self.method)
            _sim_dict[qid] = _sim
        return _sim_dict

    @staticmethod
    def _generate_predictor_res(res):
        _pred_df = DataReader(res, 'result').data_df
        return _pred_df[['score']]

    def calc_results(self, predictor_res):
        _sim_dict = self.__calc_similarity()
        _pred_results = self._generate_predictor_res(predictor_res)
        for qid in self.queries:
            # Similarity between lists - currently pearson correlation
            _sim_score = _sim_dict[qid]
            _pred_score = _pred_results.loc[qid]['score']
            _score = _sim_score * _pred_score
            print('{} {:0.4f}'.format(qid, _score))


def main(args):
    list_file = args.list1
    mod_list_file = args.list2
    predictor_scores_file = args.list3
    k = args.docs
    uef = UEF(list_file, mod_list_file, k)
    uef.calc_results(predictor_scores_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
