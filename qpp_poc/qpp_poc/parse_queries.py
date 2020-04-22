import pandas as pd
from Timer import timer

# from qpp_poc import Config
from qpp_poc.qpp_poc import Config


def transform_list_to_counts_dict(_list):
    counts = [_list.count(i) for i in _list]
    return {i: j for i, j in zip(_list, counts)}


class QueryParser:
    def __init__(self, queries_txt_file, **kwargs):
        self.queries_file = queries_txt_file
        self.raw_queries_sr = self._read_queries()
        self.queries_sr = self._weight_queries()

    def _read_queries(self):
        with open(self.queries_file, 'r') as fp:
            queries = [line.strip().split(' ', maxsplit=1) for line in fp]
        _queries_df = pd.DataFrame(queries, columns=['qid', 'terms']).set_index('qid')
        return _queries_df.terms.str.split()

    def _weight_queries(self):
        return self.raw_queries_sr.apply(transform_list_to_counts_dict)

    def get_query(self, qid: str) -> dict:
        return self.queries_sr.loc[qid]

    def get_query_ids(self) -> list:
        return self.queries_sr.index.tolist()


@timer
def test():
    for _ in range(100):
        _ = QueryParser(Config.QUERIES_FILE)


if __name__ == '__main__':
    qp = QueryParser(Config.QUERIES_FILE)
    print(qp.get_query_ids())
    print(qp.get_query(qp.get_query_ids()[0]))
    # test()
