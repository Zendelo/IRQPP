from Timer import timer

from qpp_poc.qpp_poc.utility_functions import *

"""
The files in the index dump dir:

FILE NAME:  FILE FORMAT
----------  -----------

dict.txt_new:   term    term_id df_t    cf_t
text.inv:   term    cf_t    df_t    doc_id:tf_t
doc_lens.txt:   doc_len
doc_names.txt:  doc_name
global.txt: total_docs  total_terms
"""

TEXT_INV = "/research/local/olz/ROBUST04/dump/text.inv"
DICT_TXT = "/research/local/olz/ROBUST04/dump/dict_new.txt"
DOC_LENS = "/research/local/olz/ROBUST04/dump/doc_lens.txt"
DOC_NAMES = "/research/local/olz/ROBUST04/dump/doc_names.txt"
INDEX_GLOBALS = "/research/local/olz/ROBUST04/dump/global.txt"


@timer
def testing():
    lines = []
    x = get_file_len(TEXT_INV)
    for n in range(1, x, 10000):
        lines.append(read_line(TEXT_INV, n))
    print(len(lines))
    print(parse_posting_lists(lines))
    return lines, x


class Index:
    # TODO: Possibly can be optimised by a new implementation of linecache
    def __init__(self, text_inverted, terms_dict, index_global, **kwargs):
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

    def get_posting_list(self, term):
        term_id, df_t, cf_t = self.terms_df.loc[term]
        return self._read_index_line(term_id)


@timer
def main():
    idx_obj = Index(TEXT_INV, DICT_TXT, INDEX_GLOBALS)
    postings = []
    for q_term in ['international', 'organized', 'crime']:
        postings.append(idx_obj.get_posting_list(q_term))
    x = parse_posting_lists(postings)
    df = pd.DataFrame(x).set_index('term')
    i = 0
    for doc in df.loc['organized'].posting_list:
        print(doc)
        i += 1
        if i > 5:
            break
    print(df.to_latex())
    # df = create_dict_from_index(TEXT_INV)
    # df.to_csv(DICT_TXT, sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()
