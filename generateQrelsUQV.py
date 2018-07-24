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
parser.add_argument('-u', '--UQV', default='data/ROBUST/queriesUQV.txt', help='path to queriesUQV.txt file')
parser.add_argument('-r', '--qrels', default='data/ROBUST/qrels', help='path to qrels file')
parser.add_argument('-x', '--queriestxt', default='data/ROBUST/fullqueriesUQV.txt',
                    help='path to txt file to convert to xml')


class QueriesTextParser:
    def __init__(self, queries_file, kind: str = 'original'):
        self.queries_df = pd.read_table(queries_file, delim_whitespace=False, delimiter=':', header=None,
                                        names=['qid', 'text'])
        self.queries_dict = defaultdict(str)
        self.__generate_queries_dict(queries_file)
        self.kind = kind
        if self.kind.lower() == 'uqv':
            # {qid: [qid-x-y]} list of all variations
            self.query_var, self.var_qid = self.__generate_query_var()

    def __generate_query_var(self):
        qid_vars_dict = defaultdict(list)
        vars_qid_dict = defaultdict(str)
        for row in self.queries_df.values:
            rawqid = row[0]
            qid = rawqid.split('-')[0]
            qid_vars_dict[qid].append(rawqid)
            vars_qid_dict[rawqid] = qid
        return qid_vars_dict, vars_qid_dict

    def __generate_queries_dict(self, file):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row = row[0].split(':')
                qid = row[0]
                text = row[1]
                self.queries_dict[qid] = text
            f.close()

    def get_orig_qid(self, var_qid):
        return self.var_qid[var_qid]


class QrelsParser:
    def __init__(self, file, queries: QueriesTextParser, uqv: QueriesTextParser):
        self.queries = queries
        self.uqv = uqv
        self._add_original_queries()
        self.results_df = pd.read_table(file, delim_whitespace=True, header=None, names=['qid', 'iter', 'docNo', 'rel'],
                                        dtype={'qid': str, 'iter': int, 'docNo': str, 'rel': int})
        self.results_dict = defaultdict(list)
        self._generate_qrels_dict()
        self.new_results_dict = defaultdict(list)
        self._expand_results()

    def _generate_qrels_dict(self):
        for qid in self.queries.queries_dict.keys():
            temp_df = self.results_df[self.results_df['qid'] == qid]
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
        for qid in self.uqv.queries_dict.keys():
            orig_qid = self.uqv.get_orig_qid(qid)
            it = self.results_df[self.results_df['qid'] == orig_qid]['iter']
            rel = self.results_df[self.results_df['qid'] == orig_qid]['rel']
            docs = self.results_df[self.results_df['qid'] == orig_qid]['docNo']
            _df = pd.concat([it, docs, rel], axis=1)
            _df.insert(0, 'qid', qid)
            print(_df.to_string(index=False, header=False, justify='left'))


class QueriesXMLParser:
    def __init__(self, queries):
        self.queries = queries
        self.root = etree.Element('parameters')
        self._add_queries()

    def _add_queries(self):
        for qid, text in self.queries.queries_dict.items():
            query = etree.SubElement(self.root, 'query')
            number = etree.SubElement(query, 'number')
            number.text = qid
            txt = etree.SubElement(query, 'text')
            txt.text = '#combine( {} )'.format(text)

    def print_queries_xml(self):
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'))


def main(args: parser):
    queries_file = args.queries
    uqv_file = args.UQV
    qrels_file = args.qrels
    txt_file = args.queriestxt

    original_queries = QueriesTextParser(queries_file, 'original')
    uqv_queries = QueriesTextParser(uqv_file, 'uqv')
    qrels_obj = QrelsParser(qrels_file, original_queries, uqv_queries)
    queries_txt = QueriesTextParser(txt_file)
    query_xml = QueriesXMLParser(queries_txt)

    # query_xml.print_queries_xml()

    # qrels_obj.print_results()

    qrels_obj.print_results()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
