import os

import qpputils as qu
from Timer import timer

from qpp_poc.qpp_poc import Index, QueryParser, LocalManager

INDEX_DIR = qu.ensure_dir("/research/local/olz/ROBUST04/dump/", create_if_not=False)
TEXT_INV = qu.ensure_file(os.path.join(INDEX_DIR, 'text.inv'))
DICT_TXT = qu.ensure_file(os.path.join(INDEX_DIR, 'dict_new.txt'))
DOC_LENS = qu.ensure_file(os.path.join(INDEX_DIR, 'doc_lens.txt'))
DOC_NAMES = qu.ensure_file(os.path.join(INDEX_DIR, 'doc_names.txt'))
INDEX_GLOBALS = qu.ensure_file(os.path.join(INDEX_DIR, 'global.txt'))
QUERIES_FILE = "/research/local/olz/data/robust04.qry"


@timer
def main():
    index = Index(text_inverted=TEXT_INV, terms_dict=DICT_TXT, index_global=INDEX_GLOBALS)
    queries = QueryParser(QUERIES_FILE)
    qids = queries.get_query_ids()
    process = LocalManager(index_obj=index, query_obj=queries, qid=qids[0])
    process.run_matching()


if __name__ == '__main__':
    main()
