#!/usr/bin/env python

import numpy as np
import pandas as pd

from Timer.timer import Timer
from dataparser import DataReader
import argparse

parser = argparse.ArgumentParser(description='Fusion script',
                                 usage='python3.6 fusion.py -r QL.res ',
                                 epilog='prints the fused QL result')
parser.add_argument('QLresults', help="path to UQV QL results file")


class CombSUM:
    def __init__(self, raw_ql_res: str):
        self.raw_ql_res = DataReader(raw_ql_res, 'trec')
        self.data_df = self.raw_ql_res.data_df

    @staticmethod
    def _normalize_scores(df: pd.DataFrame):
        scores = np.exp(df.docScore.values)
        min = np.min(scores)
        max = np.max(scores)
        interval = max - min
        df.loc[:, 'docScore'] = (scores - min) / interval
        return df

    def calc_scores(self):
        final_list = []
        data_df = self.data_df.filter(['qid', 'docID', 'docScore'], axis=1)
        for topic, vars in self.raw_ql_res.qid_vars.items():
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


def main(args):
    res_file = args.QLresults
    x = CombSUM(res_file)
    x.calc_scores()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
