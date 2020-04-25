import pandas as pd
from Timer import timer

# from qpp_poc import Config, read_line, get_file_len, TermPosting, Posting, TermRecord
from qpp_poc.qpp_poc import Config, read_line, get_file_len, TermPosting, Posting, TermRecord

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


def parse_posting_list(posting_list: str) -> TermPosting:
    if posting_list:
        if not isinstance(posting_list, str):
            raise TypeError(f'parse_posting_lists is expecting a str {type(posting_list)} was passed')
    row = posting_list.split()
    term, cf_t, df_t, posting_list = row[0], int(row[1]), int(row[2]), row[3:]
    return TermPosting(term, cf_t, df_t, tuple(Posting(*map(int, p.split(':'))) for p in posting_list))


def create_dict_from_index(file_path):
    """This function used to create a dictionary from an inverted index file"""
    result = []
    with open(file_path, 'r') as fp:
        for i, post in enumerate(fp):
            row = post.split()
            term, cf_t, df_t, _ = row[0], row[1], row[2], row[3:]
            result.append(TermRecord(term, i + 1, cf_t, df_t))
    return pd.DataFrame.from_records(result, columns=TermRecord._fields)


def read_dict_file(file_path):
    return pd.read_table(file_path, names=['term', 'term_id', 'df_t', 'cf_t'], index_col='term', delim_whitespace=True,
                         keep_default_na=False)


def read_doc_lens_file(file_path):
    with open(file_path, 'r') as fp:
        result = {(k + 1): int(v) for k, v in enumerate(fp.readlines())}
    return result


def read_doc_names_file(file_path):
    with open(file_path, 'r') as fp:
        result = {(k + 1): str(v.strip('\n')) for k, v in enumerate(fp.readlines())}
    return result


class Index:
    @classmethod
    def oov(cls, term):
        return f"{term} 0 0"  # Out of vocabulary terms

    # TODO: Possibly can be optimised by a new implementation of linecache
    def __init__(self, text_inverted, terms_dict, index_global, document_lengths, document_names, **kwargs):
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
        self.document_length_dict = read_doc_lens_file(document_lengths)
        self.document_names_dict = read_doc_names_file(document_names)

    def _read_index_line(self, n):
        assert 0 < n <= self.inv_index_size, f"Row {n} is out of the index range 1 - {self.inv_index_size}"
        return read_line(self.inverted_file, n)

    def _get_raw_posting_list(self, term):
        if not hasattr(self, 'terms_dict'):
            self.terms_dict = self._generate_terms_dict()
        term_id = self.terms_dict.get(term)
        if term_id:
            return self._read_index_line(term_id)
        else:
            return Index.oov(term)

    def _generate_terms_dict(self):
        return self.terms_df['term_id'].to_dict()

    def get_posting_list(self, term: str) -> TermPosting:
        posting_lists = self._get_raw_posting_list(term)
        return parse_posting_list(posting_lists)

    def get_term_cf(self, term: str) -> int:
        if self.terms_dict.get(term):
            return self.terms_df.loc[term, 'cf_t']
        else:
            return 0

    def get_doc_len(self, doc_id):
        return self.document_length_dict.get(doc_id)

    def get_doc_name(self, doc_id):
        return self.document_names_dict.get(doc_id)


@timer
def main():
    index = Index(text_inverted=text_inv, terms_dict=dict_txt, index_global=index_globals, document_lengths=doc_lens,
                  document_names=doc_names)
    x = index.get_posting_list('ingathering')
    print(len(x.posting_list))


if __name__ == '__main__':
    config = Config()
    text_inv, dict_txt, index_globals, doc_lens, doc_names = config.TEXT_INV, config.DICT_TXT, config.INDEX_GLOBALS, config.DOC_LENS, config.DOC_NAMES
    main()
