#!/usr/bin/env python

import csv
import glob
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as eT
from collections import defaultdict
from subprocess import run

import numpy as np
import pandas as pd
from lxml import etree


# TODO: implement all necessary objects and functions, in order to switch all calculations to work with those classes

class ResultsReader:
    def __init__(self, data_file: str, file_type):
        """
        :param data_file: results file path
        :param file_type: 'predictions' for predictor results or 'ap' for ap results or trec for trec format results
        """
        ensure_file(data_file)
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
            try:
                first_row = next(reader)
            except StopIteration:
                sys.exit(f'The file {self.data} is empty')
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
        """The function receives a list of qids, and returns a dict of results in format: {'docID': 'docScore'} """
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

    def filter_results_by_queries(self):
        pass


class QueriesTextParser:

    def __init__(self, queries_file, kind: str = 'original'):
        """
        :param queries_file: path to txt queries file
        :param kind: 'original' or 'uqv'
        """
        self.queries_df = pd.read_table(queries_file, delim_whitespace=False, delimiter=':', header=None,
                                        names=['qid', 'text'], dtype={'qid': str, 'text': str})
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


class QueriesXMLParser:
    # TODO: add queries_df
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
            qid_ = query.find('number').text
            qstr_ = query.find('text').text
            qtxt_ = qstr_[qstr_.find("(") + 1:qstr_.rfind(")")].split()
            self.full_queries[qid_] = qstr_
            self.text_queries[qid_] = qtxt_
            self.query_length[qid_] = len(qtxt_)


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


class QueriesXMLWriter:
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
        """Prints to STD.OUT (usually the screen)"""
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'))

    def print_queries_xml_file(self, file_name):
        """Prints to a File"""
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'), file=open(file_name, 'w'))


def ensure_file(file):
    """Ensure a single file exists, returns the full path of the file if True or throws an Assertion error if not"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file))
    assert os.path.isfile(file_path), "The file {} doesn't exist. Please create the file first".format(file)
    return file_path


def ensure_dir(file_path):
    """The function ensures the dir exists, if it doesn't it creates it and returns the path"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
    return directory


def empty_dir(dir_path, force=False):
    if force:
        run(f'rm -v {dir_path}/*', shell=True)
    else:
        files = os.listdir(dir_path)
        if len(files) and not mp.current_process().daemon:
            answer = input(
                f'The directory {dir_path} contains {len(files)} files, do you want to remove them?\n [yes\\No] ')
            if answer.lower() == 'yes':
                run(f'rm -v {dir_path}/*', shell=True)


def convert_vid_to_qid(df: pd.DataFrame):
    if df.index.name != 'qid' and df.index.name != 'topic':
        if 'qid' in df.columns:
            _df = df.set_index('qid')
        elif 'topic' in df.columns:
            _df = df.set_index('topic')
        else:
            assert False, "The DF doesn't has qid or topic"
    else:
        _df = df
    _df.rename(index=lambda x: f'{x.split("-")[0]}', inplace=True)
    return _df


def read_rm_prob_files(data_dir, number_of_docs, clipping='*'):
    """The function creates a DF from files, the probabilities are p(w|RM1) for all query words
    If a query term doesn't appear in the file, it's implies p(w|R)=0"""
    data_files = glob.glob(f'{data_dir}/probabilities-{number_of_docs}+{clipping}')
    if len(data_files) < 1:
        data_files = glob.glob(f'{data_dir}/probabilities-{number_of_docs}')
    _list = []
    for _file in data_files:
        _col = f'{_file.rsplit("/")[-1].rsplit("-")[-1]}'
        _df = pd.read_table(_file, names=['qid', 'term', _col], sep=' ')
        _df = _df.astype({'qid': str}).set_index(['qid', 'term'])
        _list.append(_df)
    return pd.concat(_list, axis=1).fillna(0)
