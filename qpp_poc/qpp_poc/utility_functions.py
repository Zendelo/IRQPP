import linecache
from collections import namedtuple

import pandas as pd


def get_file_len(file_path):
    """Function that opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Function used to return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)


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
