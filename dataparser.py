#!/usr/bin/env python

import csv
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from lxml import etree
import os


# TODO: implement all necessary objects and functions, in order to switch all calculations to work with those classes

class ResultsReader:
    def __init__(self, data_file: str, file_type):
        """
        :param data_file: results file path
        :param file_type: 'predictions' for predictor results or 'ap' for ap results or trec for trec format results
        """
        self.file_type = file_type.lower()
        self.data = data_file
        self.__number_of_col = self.__check_number_of_col()
        if self.file_type == 'predictions':
            assert self.__number_of_col == 2 or self.__number_of_col == 4, 'Wrong File format'
            self.data_df = self.__read_results_data_2() if self.__number_of_col == 2 else self.__read_results_data_4()
        elif self.file_type == 'ap':
            self.data_df = self.__read_ap_data_2()
        elif self.file_type == 'trec':
            assert self.__number_of_col == 6, 'Wrong File format, trec format should have 6 columns'
            self.data_df = self.__read_trec_data()
        else:
            sys.exit('Unknown file type, use ap, trec or predictions file type')
        self.query_vars, self.var_qid = self.__generate_qids_from_res()

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
        data_df.index = data_df.index.astype(str)
        data_df.sort_values(by=['qid', 'score'], ascending=[True, False], inplace=True)
        return data_df

    def __read_ap_data_2(self):
        """Assuming data is a res with 2 columns, 'Qid AP'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'ap'],
                                dtype={'qid': str, 'ap': np.float64})
        data_df.sort_values(by=['qid', 'ap'], ascending=[True, False], inplace=True)
        data_df.index = data_df.index.astype(str)
        return data_df

    def __read_results_data_4(self):
        """Assuming data is a res with 4 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'entropy', 'cross_entropy', 'score'],
                                dtype={'qid': str, 'score': np.float64, 'entropy': np.float64,
                                       'cross_entropy': np.float64})
        data_df = data_df.filter(['qid', 'score'], axis=1)
        data_df.index = data_df.index.astype(str)
        data_df.sort_values(by=['qid', 'score'], ascending=[True, False], inplace=True)
        return data_df

    def __read_trec_data(self):
        """Assuming data is a trec format results file with 6 columns, 'Qid entropy cross_entropy Score'"""
        data_df = pd.read_table(self.data, delim_whitespace=True, header=None, index_col=0,
                                names=['qid', 'Q0', 'docID', 'docRank', 'docScore', 'ind'],
                                dtype={'qid': str, 'Q0': str, 'docID': str, 'docRank': int, 'docScore': float,
                                       'ind': str})
        data_df = data_df.filter(['qid', 'docID', 'docRank', 'docScore'], axis=1)
        data_df.index = data_df.index.astype(str)
        data_df.sort_values(by=['qid', 'docRank'], ascending=True, inplace=True)
        return data_df

    def __generate_qids_from_res(self):
        qid_vars = defaultdict(list)
        var_qid = defaultdict(str)
        raw_qids = self.data_df.index.unique()
        for _qid in raw_qids:
            qid = _qid.split('-')[0]
            qid_vars[qid].append(_qid)
            var_qid[_qid] = qid
        return qid_vars, var_qid

    def get_res_dict_by_qid(self, qid, top=1000):
        assert self.file_type == 'trec', '{} wrong file type'.format(self.file_type)
        _df = self.data_df.loc[qid, ['docID', 'docScore']].head(top)
        _df.reset_index(drop=True, inplace=True)
        _df.set_index('docID', inplace=True)
        _dict = _df.to_dict()['docScore']
        return _dict

    def get_qid_by_var(self, var):
        return self.var_qid.get(var)

    def get_vars_by_qid(self, qid):
        return self.query_vars.get(qid)

    def get_docs_by_qid(self, qid, top=1000):
        assert self.file_type == 'trec', '{} wrong file type'.format(self.file_type)
        return self.data_df.loc[qid, 'docID'].head(top).values


class QueriesTextParser:

    def __init__(self, queries_file, kind: str = 'original'):
        """
        :param queries_file: path to txt queries file
        :param kind: 'original' or 'uqv'
        """
        self.queries_df = pd.read_table(queries_file, delim_whitespace=False, delimiter=':', header=None,
                                        names=['qid', 'text'])
        self.queries_dict = self.__generate_queries_dict()
        self.kind = kind.lower()
        if self.kind == 'uqv':
            # {qid: [qid-x-y]} list of all variations
            self.query_vars, self.var_qid = self.__generate_query_var()

    def __generate_query_var(self):
        qid_vars_dict = defaultdict(list)
        vars_qid_dict = defaultdict(str)
        for rawqid in self.queries_dict.keys():
            qid = rawqid.split('-')[0]
            qid_vars_dict[qid].append(rawqid)
            vars_qid_dict[rawqid] = qid
        return qid_vars_dict, vars_qid_dict

    def __generate_queries_dict(self):
        queries_dict = defaultdict(str)
        for qid, text in self.queries_df.values:
            queries_dict[qid] = text
        return queries_dict

    def get_orig_qid(self, var_qid):
        return self.var_qid.get(var_qid)

    def get_vars(self, orig_qid):
        return self.query_vars.get(orig_qid)

    def get_qid_txt(self, qid):
        return self.queries_dict.get(qid)


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
            qid, x, y = self.uqv.query_vars[qid][-1].split('-')
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
    def __init__(self, queries_df: pd.DataFrame):
        self.queries_df = queries_df
        self.root = etree.Element('parameters')
        self._add_queries()

    def _add_queries(self):
        for qid, text in self.queries_df.values:
            query = etree.SubElement(self.root, 'query')
            number = etree.SubElement(query, 'number')
            number.text = qid
            txt = etree.SubElement(query, 'text')
            txt.text = '#combine( {} )'.format(text)

    def print_queries_xml(self):
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'))


def ensure_file(file):
    """Ensure a single file exists, returns the full path of the file if True"""
    # tilde expansion
    file_path = os.path.expanduser(file)
    assert os.path.isfile(file_path), "The file {} doesn't exist. Please create the file first".format(file)
    return file_path


def ensure_dir(file_path):
    # tilde expansion
    file_path = os.path.expanduser(file_path)
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)