#! /usr/bin/env python

import argparse
import xml.etree.ElementTree as eT
from collections import defaultdict
from math import sqrt
import numpy as np
import glob

import pandas as pd

NUMBER_OF_DOCS = [5, 10, 25, 50, 100]

parser = argparse.ArgumentParser(description='NQC predictor',
                                 usage='Input CE(q|d) scores and queries files',
                                 epilog='Prints the NQC predictor scores')

parser.add_argument('list1', metavar='QL_results_file', help='The original QL results file for the documents scores')
parser.add_argument('list2', metavar='RM1_results_dir', help='The directory with the RM1 lists')
parser.add_argument('-d', '--docs', metavar='K', default=5, help='Number of k top documents')


# parser.add_argument('-d', '--docs', metavar='KDocs', default=20, help='Number of K top documents')

class ResultsReader:
    def __init__(self, results_file):
        self.results = results_file
        self.results_df = self.__read_file()

    def __read_file(self):
        return pd.read_table(self.results, delim_whitespace=True, header=None, index_col=[0],
                             names=['qid', 'Q0', 'docID', 'docRank', 'docScore', 'ind'],
                             dtype={'qid': int, 'Q0': str, 'docID': str, 'docRank': int, 'docScore': float,
                                    'ind': str})


class QueriesParser:
    def __init__(self, query_file):
        self.file = query_file
        self.tree = eT.parse(self.file)
        self.root = self.tree.getroot()
        # query number: "Full command"
        self.full_queries = defaultdict(str)
        self.text_queries = defaultdict(str)
        self.query_length = defaultdict(int)
        self.fb_docs = defaultdict(list)
        self.__parse_queries()

    def __parse_queries(self):
        for query in self.root.iter('query'):
            qid_ = int(query.find('number').text)
            qstr_ = query.find('text').text
            qtxt_ = qstr_[qstr_.find("(") + 1:qstr_.rfind(")")].split()
            self.full_queries[qid_] = qstr_
            self.text_queries[qid_] = qtxt_
            self.query_length[qid_] = len(qtxt_)

    def add_feedback_docs(self, num_docs, res):
        """
        Adds the fbDocs from results file to the original queries
        :parameter: num_files: number of fbDocs to add to each query
        """
        for qid in self.full_queries.keys():
            qid = int(qid)
            docs = res.loc[qid]['docID'].head(num_docs)
            self.fb_docs[qid] = list(docs)

    def write_to_file(self):
        for query in self.root.iter('query'):
            qid = int(query.find('number').text)
            fbDocs = self.fb_docs[qid]
            for doc in fbDocs:
                temp = eT.SubElement(query, 'feedbackDocno')
                temp.text = doc
        eT.dump(self.tree)


class QF:
    def __init__(self, original_list, modified_list, results_dir=None):
        self.orig_list_df = ResultsReader(original_list).results_df
        self.mod_list_df = ResultsReader(modified_list).results_df
        # self.res_dir = results_dir
        # self.lists_dict = defaultdict()
        # self.__read_lists()
        self.predictions = defaultdict(float)
        self.queries = self.__generate_queries()

    def __read_lists(self):
        """Assuming the results lists files are named : list-[0-9]"""
        all_files = glob.glob(self.res_dir + "/*list*")
        for file_ in all_files:
            file_name = file_.split('-')[-1]
            df = ResultsReader(file_).results_df
            self.lists_dict[file_name] = df

    def __generate_queries(self):
        _qids = set(self.orig_list_df.index)
        return tuple(sorted(_qids))

    def __create_docs_sets(self, df, k):
        _docs_dict = defaultdict(set)
        for qid in self.queries:
            _docs_dict[qid] = set(df.loc[qid].head(k)['docID'])
        return _docs_dict

    def calc_results(self, number_of_docs):
        orig_docs_dict = self.__create_docs_sets(self.orig_list_df, number_of_docs)
        mod_docs_dict = self.__create_docs_sets(self.mod_list_df, number_of_docs)
        for qid in self.queries:
            print('{} {}'.format(qid, len(orig_docs_dict[qid].intersection(mod_docs_dict[qid]))))


def main(args):
    list_file = args.list1
    mod_list_file = args.list2
    k = int(args.docs)
    qf = QF(list_file, mod_list_file)
    qf.calc_results(k)

    # for k in NUMBER_OF_DOCS:
    #     qf.calc_results(k)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
