#!/usr/bin/env python

import argparse
from lxml import etree as eT
# import xml.etree.ElementTree as eT
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser(description='Adding WorkingSetDocs to queries xml file',
                                 usage='Receives a query xml file and trec results file',
                                 epilog='Adds a list of documents to each query as WorkingSet')

parser.add_argument('list', metavar='results_file', help='The results file for the WorkingSet')
parser.add_argument('queries', metavar='queries_xml_file', help='The queries xml file')
parser.add_argument('-d', '--docs', metavar='fbDocs', default=5, help='Number of Feedback documents to add')


class QueriesParser:
    def __init__(self, query_file, results_df):
        self.file = query_file
        _parser = eT.XMLParser(remove_blank_text=True)
        self.tree = eT.parse(self.file, _parser)
        self.root = self.tree.getroot()
        # query number: "Full command"
        self.full_queries = defaultdict(str)
        self.fb_docs = defaultdict(list)
        self.__parse_queries()
        self.res = results_df

    def __parse_queries(self):
        for query in self.root.iter('query'):
            self.full_queries[query.find('number').text] = query.find('text').text

    def add_working_set_docs(self, number_of_docs):
        """
        Adds the workingSetDocs from results file to the original queries
        :parameter: number_of_docs: number of docs to add to each query
        """
        for qid in self.full_queries.keys():
            qid = int(qid)
            docs = self.res.loc[qid]['docID'].head(number_of_docs)
            self.fb_docs[qid] = list(docs)

    def print_output(self):
        for query in self.root.iter('query'):
            qid = int(query.find('number').text)
            fbDocs = self.fb_docs[qid]
            for doc in fbDocs:
                temp = eT.SubElement(query, 'workingSetDocno')
                temp.text = doc
        # eT.dump(self.tree)
        # eT.dump(self.tree, pretty_print=True)
        print(eT.tostring(self.tree, pretty_print=True, encoding='unicode'))


def main(args):
    results_file = args.list
    query_file = args.queries
    number_of_docs = int(args.docs)
    results_df = pd.read_table(results_file, delim_whitespace=True, header=None, index_col=[0, 3],
                               names=['qid', 'Q0', 'docID', 'docRank', 'docScore', 'ind'],
                               dtype={'qid': int, 'Q0': str, 'docID': str, 'docRank': int, 'docScore': float,
                                      'ind': str})

    qdb = QueriesParser(query_file, results_df)
    qdb.add_working_set_docs(number_of_docs)
    qdb.print_output()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
