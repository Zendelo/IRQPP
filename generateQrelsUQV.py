#!/usr/bin/env python

import csv
from collections import defaultdict
import argparse
from lxml import etree as etree
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Queries UQV pre-processing',
                                 usage='Create new UQV qrels file',
                                 epilog='ROBUST version')

parser.add_argument('-q', '--queries', default='data/ROBUST/queries.txt', help='path to queries.txt file')
parser.add_argument('-u', '--UQV', default='data/ROBUST/fullqueriesUQV.txt', help='path to queriesUQV.txt file')
parser.add_argument('-r', '--qrels', default='data/ROBUST/qrels', help='path to qrels file')


class QueriesTextParser:
    def __init__(self, queries_file, kind: str = 'original'):
        self.queries_df = pd.read_table(queries_file, delim_whitespace=False, delimiter=':', header=None,
                                        names=['qid', 'text'])
        self.queries_dict = defaultdict(str)
        self.__generate_queries_dict(queries_file)
        self.kind = kind
        if self.kind.lower() == 'uqv':
            # {qid: [qid-x-y]} list of all variations
            self.query_var = self.__generate_query_var()

    def __generate_query_var(self):
        vars_dict = defaultdict(list)
        for row in self.queries_df.values:
            rawqid = row[0]
            qid = int(rawqid.split('-')[0])
            vars_dict[qid].append(rawqid)
        return vars_dict

    def __generate_queries_dict(self, file):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = row[0].split(':')
                qid = row[0]
                text = row[1]
                self.queries_dict[qid] = text
            f.close()


class QrelsParser:
    def __init__(self, file, queries: QueriesTextParser, uqv: QueriesTextParser):
        self.queries = queries
        self.uqv = uqv
        self._add_original_queries()
        self.results_df = pd.read_table(file, delim_whitespace=True, header=None, names=['qid', 'iter', 'docNo', 'rel'])
        self.results_dict = defaultdict(list)
        self._generate_qrels_dict()
        self.new_results_dict = defaultdict(list)
        self._expand_results()

    def _generate_qrels_dict(self):
        for qid in self.queries.queries_dict.keys():
            temp_df = self.results_df[self.results_df['qid'] == int(qid)]
            docs = temp_df[temp_df['rel'] == 1]['docNo'].values
            self.results_dict[qid] = docs

    def _expand_results(self):
        for rawqid in self.uqv.queries_dict.keys():
            qid = rawqid.split('-')[0]
            self.new_results_dict[rawqid] = self.results_dict[qid]

    def _find_missing_queries(self):
        missing_qids = list()
        for qid, text in self.queries.queries_df.values:
            if text not in self.uqv.queries_df.values:
                missing_qids.append(qid)
        return missing_qids

    def _make_query_pairs(self):
        missing = self._find_missing_queries()
        pairs = list()
        for qid in missing:
            qid, x, y = self.uqv.query_var[qid][-1].split('-')
            x = int(x) + 1
            y = 1
            new_qid = '{}-{}-{}'.format(qid, x, y)
            pairs.append((qid, new_qid))
        return pairs

    def _add_original_queries(self):
        pairs = self._make_query_pairs()
        for qid, new_qid in pairs:
            text = self.queries.queries_dict[qid]
            self.uqv.queries_dict[new_qid] = text

    def print_uqv(self):
        for qid, text in self.uqv.queries_dict.items():
            print('{}:{}'.format(qid, text))

    def print_results(self):
        for qid, docs in self.new_results_dict.items():
            for doc in docs:
                print('{} 1 {} 1'.format(qid, doc))


def main(args: parser):
    queries_file = args.queries
    uqv_file = args.UQV
    qrels_file = args.qrels

    original_queries = QueriesTextParser(queries_file, 'original')
    uqv_queries = QueriesTextParser(uqv_file, 'uqv')
    qrels_obj = QrelsParser(qrels_file, original_queries, uqv_queries)

    qrels_obj.print_results()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
