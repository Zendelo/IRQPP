from Timer import timer
from collections import namedtuple
import pandas as pd
import qpputils as qu
import os

from qpp_poc.qpp_poc import Config, read_line, get_file_len

"""
The expected files in the index dump dir:

FILE NAME:  FILE FORMAT
----------  -----------

dict.txt:   term    term_id df_t    cf_t
text.inv:   term    cf_t    df_t    doc_id:tf_t
doc_lens.txt:   doc_len
doc_names.txt:  doc_name
global.txt: total_docs  total_terms
"""


def parse_posting_lists(posting_lists):
    # TODO: test what's better, generator vs list vs tuple
    result = []
    if posting_lists:
        if not isinstance(posting_lists, list):
            if isinstance(posting_lists, str):
                posting_lists = [posting_lists]
            else:
                raise TypeError(f'parse_posting_lists is expecting list of strings, or str\n'
                                f'{type(posting_lists)} was passed')
    else:
        return []
    posting = namedtuple('posting', ['doc_id', 'tf'])
    term_record = namedtuple('TermRecord', ['term', 'cf', 'df', 'posting_list'])
    for post in posting_lists:
        row = post.split()
        term, cf_t, df_t, posting_list = row[0], row[1], row[2], row[3:]
        result.append(term_record(term, cf_t, df_t, (posting(*map(int, p.split(':'))) for p in posting_list)))
    return result


def create_dict_from_index(file_path):
    """This function used to create a dictionary from an inverted index file"""
    result = []
    term_record = namedtuple('TermRecord', ['term', 'term_id', 'cf', 'df'])
    with open(file_path, 'r') as fp:
        for i, post in enumerate(fp):
            row = post.split()
            term, cf_t, df_t, _ = row[0], row[1], row[2], row[3:]
            result.append(term_record(term, i + 1, cf_t, df_t))
    return pd.DataFrame.from_records(result, columns=term_record._fields)


def read_dict_file(file_path):
    return pd.read_table(file_path, names=['term', 'term_id', 'df_t', 'cf_t'], index_col='term', delim_whitespace=True,
                         keep_default_na=False)


class Index:
    # TODO: Possibly can be optimised by a new implementation of linecache
    def __init__(self, text_inverted, terms_dict, index_global, **kwargs):
        """
        Tess
        :param text_inverted: inverted index text file
        :param terms_dict: terms dictionary, can be generated from the inverted index using create_dict_from_index function
        :param index_global: global index stats for validation
        :param kwargs: currently unused
        """
        with open(index_global) as fp:
            self.number_of_docs, self.total_terms = map(int, fp.read().split())
        self.terms_df = read_dict_file(terms_dict)
        self.inv_index_size = get_file_len(text_inverted)
        self.inverted_file = text_inverted
        self.vocab_size = len(self.terms_df)
        assert self.vocab_size == self.inv_index_size, f"The vocabulary and index sizes differ" \
                                                       f"\nVocab: {self.vocab_size} vs Index: {self.inv_index_size}"

    def _read_index_line(self, n):
        assert 0 < n <= self.inv_index_size, f"Row {n} is out of the index range 1 - {self.inv_index_size}"
        return read_line(self.inverted_file, n)

    def _get_raw_posting_list(self, term):
        term_id, df_t, cf_t = self.terms_df.loc[term]
        return self._read_index_line(term_id)

    def get_posting_list(self, term):
        posting_lists = self._get_raw_posting_list(term)
        return parse_posting_lists(posting_lists)

    def get_term_cf(self, term):
        return self.terms_df.loc[term, 'cf_t']


@timer
def main():
    config = Config()
    text_inv, dict_txt, index_globals = config.TEXT_INV, config.DICT_TXT, config.INDEX_GLOBALS
    idx_obj = Index(text_inv, dict_txt, index_globals)
    postings = []
    for q_term in ['international', 'organized', 'crime']:
        postings.append(idx_obj._get_raw_posting_list(q_term))
    x = parse_posting_lists(postings)
    df = pd.DataFrame(x).set_index('term')
    i = 0
    for doc in df.loc['organized'].posting_list:
        print(doc)
        i += 1
        if i > 5:
            break
    print(df.to_latex())
    # df = create_dict_from_index(text_inv)
    # df.to_csv(dict_txt, sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()
