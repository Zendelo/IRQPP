#!/usr/bin/env python

import csv
from collections import defaultdict
import argparse
from lxml import etree as etree
from matplotlib import pyplot as plt
from statistics import median_high, median_low
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Queries UQV post-processing',
                                 usage='Create new UQV queries xml file',
                                 epilog='ROBUST version')

parser.add_argument('-q', '--queries', default='data/ROBUST/fullqueriesUQV.txt', help='path to queries.txt file')
parser.add_argument('-m', '--map', default='baseline/UQVmap1000', help='path to UQV AP@1000 results')
parser.add_argument('-f', '--function', default='max', choices=['max', 'min', 'med_low', 'med_high'], help='Select queries function')


class DataReader:
    def __init__(self, data_file: str, file_type):
        """
        :param data_file: results res
        :param file_type: 'result' for predictor results res or 'ap' for ap results res
        """
        self.file_type = file_type
        self.data = data_file
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
        data_df = data_df.filter(['qid', 'score'], axis=1)
        return data_df


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


class Aggregate:
    def __init__(self, queries_data, ap, agg_func):
        self.queries_data = queries_data
        self.agg_func = agg_func
        self.ap = ap
        self.queries_dict = self._reform_new_qid()

    def _find_qid(self):
        qids_dict = defaultdict()
        self.queries_data.queries_df = self.queries_data.queries_df.set_index('qid')
        for _qid, vars in self.queries_data.query_var.items():
            _dict = self.ap.data_df.loc[vars].to_dict()['ap']
            if self.agg_func == 'max':
                qid = max(_dict, key=lambda key: _dict[key])
            elif self.agg_func == 'min':
                qid = min(_dict, key=lambda key: _dict[key])
            elif self.agg_func == 'med_low':
                x = median_low(_dict.values())
                qid = self.ap.data_df.loc[vars][self.ap.data_df['ap'] == x].head(1).index[0]
            elif self.agg_func == 'med_high':
                y = median_high(_dict.values())
                qid = self.ap.data_df.loc[vars][self.ap.data_df['ap'] == y].head(1).index[0]
            else:
                assert False, 'Unknown aggregate function {}'.format(self.agg_func)

            qids_dict[_qid] = qid
        return qids_dict

    def _reform_new_qid(self):
        qids = self._find_qid()
        new_dict = defaultdict(str)
        for qid in self.queries_data.query_var.keys():
            _qid = qids[qid]
            new_dict[qid] = self.queries_data.queries_dict[_qid]
        return new_dict


class QueriesXMLParser:
    def __init__(self, queries):
        self.queries = queries
        self.root = etree.Element('parameters')
        self._add_queries()

    def _add_queries(self):
        for qid, text in self.queries.queries_dict.items():
            query = etree.SubElement(self.root, 'query')
            number = etree.SubElement(query, 'number')
            number.text = str(qid)
            txt = etree.SubElement(query, 'text')
            txt.text = '#combine( {} )'.format(text)

    def print_queries_xml(self):
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'))


def main(args: parser):
    queries_file = args.queries
    map_file = args.map
    func = args.function

    ap = DataReader(map_file, 'ap')
    original_queries = QueriesTextParser(queries_file, 'uqv')
    agg = Aggregate(original_queries, ap, func)
    query_xml = QueriesXMLParser(agg)
    query_xml.print_queries_xml()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
