#!/usr/bin/env python

import numpy as np
import pandas as pd

from qpputils.dataparser import ResultsReader
import argparse

parser = argparse.ArgumentParser(description='Fusion script',
                                 usage='python3.6 fusion.py -r QL.res ',
                                 epilog='prints the fused QL result')
parser.add_argument('QLresults', default=None, help="path to UQV QL results file")
parser.add_argument('QLCresults', default=None, help="path to  LogQLC results file")


# TODO: add a version with only top 5 variations for topic

class CombSUM:
    def __init__(self, raw_ql_res: str, raw_qlc_res: str):
        self.raw_ql_res = ResultsReader(raw_ql_res, 'trec')
        self.raw_qlc_res = ResultsReader(raw_qlc_res, 'predictions')
        self.ql_data_df = self.raw_ql_res.data_df
        self.qlc_data_df = self.raw_qlc_res.data_df

    @staticmethod
    def _normalize_scores(df: pd.DataFrame):
        scores = np.exp(df.docScore.values)
        min = np.min(scores)
        max = np.max(scores)
        interval = max - min
        df.loc[:, 'docScore'] = (scores - min) / interval
        return df

    def calc_scores(self):
        """Calculates and prints the fused list of top 1000 documents in trec format"""
        final_list = []
        data_df = self.ql_data_df.filter(['qid', 'docID', 'docScore'], axis=1)
        for topic, vars in self.raw_ql_res.query_vars.items():
            number_of_vars = len(vars)
            _list = []
            for var in vars:
                var_df = data_df.loc[var]
                var_df = self._normalize_scores(var_df)
                _list.append(var_df)
            norm_topic_df = pd.concat(_list)
            _grouped_df = norm_topic_df.groupby('docID').sum()
            _grouped_df = _grouped_df['docScore'] / number_of_vars
            fused_df = _grouped_df.sort_values(ascending=False).head(1000).to_frame()
            fused_df.insert(0, 'docRank', np.arange(1, 1001))
            fused_df.insert(0, 'qid', topic)
            fused_df.reset_index(inplace=True)
            fused_df = fused_df.reindex(['qid', 'docID', 'docRank', 'docScore'], axis=1)
            final_list.append(fused_df)
        result_df = pd.concat(final_list, ignore_index=True)
        result_df.insert(1, 'Q0', 'Q0')
        result_df.insert(5, 'ind', 'indri')
        print(result_df.to_string(header=False, index=False, index_names=False))
        return result_df

    def average_qlc(self):
        """Saves a new file with the average QLC scores for each topic"""
        topic_qlc_dict = {}
        for topic, vars in self.raw_ql_res.query_vars.items():
            _score = self.qlc_data_df.loc[vars]['score'].mean()
            topic_qlc_dict[topic] = _score
        topic_df = pd.DataFrame.from_dict(topic_qlc_dict, orient='index')
        # Also consider using %g as float format
        topic_df.to_csv('fused-logqlc.res', header=False, index=True, sep=' ', float_format='%.4f')
        return topic_df


def main(args):
    ql_res_file = args.QLresults
    qlc_res_file = args.QLCresults
    x = CombSUM(ql_res_file, qlc_res_file)
    x.calc_scores()
    x.average_qlc()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
