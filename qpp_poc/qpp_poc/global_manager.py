import multiprocessing as mp
from functools import partial
import pandas as pd

from Timer import timer
# from qpp_poc import Index, QueryParser, LocalManager, Config
from qpp_poc.qpp_poc import Index, QueryParser, LocalManager, Config
import qpputils as qu

TEXT_INV = Config.TEXT_INV
DICT_TXT = Config.DICT_TXT
DOC_LENS = Config.DOC_LENS
DOC_NAMES = Config.DOC_NAMES
INDEX_GLOBALS = Config.INDEX_GLOBALS
QUERIES_FILE = Config.QUERIES_FILE


@timer
def run_retrieval_process(qid, index, queries):
    columns = ['qid', 'iteration', 'doc_no', 'rank', 'score', 'method']
    process = LocalManager(index_obj=index, query_obj=queries, qid=qid)
    result = process.run_retrieval()
    print(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['doc_no', 'score'])
    return df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='QppPipe')[columns]


@timer
def main():
    index = Index(text_inverted=TEXT_INV, terms_dict=DICT_TXT, index_global=INDEX_GLOBALS, document_lengths=DOC_LENS,
                  document_names=DOC_NAMES)
    queries = QueryParser(QUERIES_FILE)
    qids = queries.get_query_ids()
    # result = [run_retrieval_process(qids[1], index, queries)]
    with mp.Pool(processes=20) as pool:
        result = pool.map(partial(run_retrieval_process, index=index, queries=queries), qids)
    df = pd.concat(result)
    print(df)
    df.astype({'qid': str, 'rank': int, 'score': float}).to_csv('QL.res', sep=' ', index=False, header=False,
                                                                float_format="%.4f")


def test():
    queries = QueryParser(QUERIES_FILE)
    qu.QueriesXMLWriter(pd.DataFrame(queries.raw_queries_sr.apply(' '.join)).reset_index()).print_queries_xml_file('test_queries.xml')
    qu.QueriesXMLWriter(pd.DataFrame(queries.raw_queries_sr.apply(' '.join)).reset_index()).print_queries_xml()


if __name__ == '__main__':
    main()
    # test()
