from dataparser import QueriesTextParser, DataReader, ensure_file
from itertools import combinations
from collections import defaultdict
from RBO import rbo_dict
import pandas as pd

import argparse
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3.6 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-q', '--queries', metavar='queries.txt', help='path to UQV queries txt file')
parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-r', '--results', default=None, type=str, help='QL.res file of the queries')
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')

parser.add_argument('--generate', help="generate new predictions", action="store_true")


class FeatureFactory:
    def __init__(self, results_file, queries_file):
        self.res_data = DataReader(results_file, 'trec')
        self.queries_data = QueriesTextParser(queries_file, 'uqv')
        # A simple sanity check to make sure number of results and query variations is identical
        _x = [len(i) for i in self.res_data.query_vars.values()]
        _y = [len(i) for i in self.queries_data.query_vars.values()]
        assert _x == _y, 'Results and Queries files don\'t match'
        self.query_vars = self.queries_data.query_vars
        self.features_index = self._create_feature_pairs()

    def _create_feature_pairs(self):
        feature_keys = defaultdict(list)
        for qid, vars in self.query_vars.items():
            feature_keys[qid] = list(combinations(vars, 2))
        return feature_keys

    def generate_features(self):
        _list = []
        features_dict = defaultdict(dict)
        for topic, pairs in self.features_index.items():
            for q1, q2 in pairs:
                txt1 = self.queries_data.get_qid_txt(q1)
                txt2 = self.queries_data.get_qid_txt(q2)
                jc = self.jaccard_coefficient(txt1, txt2)

                l1 = self.res_data.get_docs_by_qid(q1, 10)
                l2 = self.res_data.get_docs_by_qid(q2, 10)
                docs_overlap = self.list_overlap(l1, l2)

                d1 = self.res_data.get_res_dict_by_qid(q1, top=1000)
                d2 = self.res_data.get_res_dict_by_qid(q2, top=1000)
                _rbo_scores_1000 = rbo_dict(d1, d2, p=0.95)
                _min_1000 = _rbo_scores_1000['min']
                _res_1000 = _rbo_scores_1000['res']
                _ext_1000 = _rbo_scores_1000['ext']

                d1 = self.res_data.get_res_dict_by_qid(q1, top=100)
                d2 = self.res_data.get_res_dict_by_qid(q2, top=100)
                _rbo_scores_100 = rbo_dict(d1, d2, p=0.95)
                _min_100 = _rbo_scores_100['min']
                _res_100 = _rbo_scores_100['res']
                _ext_100 = _rbo_scores_100['ext']

                features_dict[topic, q1, q2] = {'Jac_coefficient': jc, 'Top_10_Docs_overlap': docs_overlap,
                                                'RBO_MIN_1000': _min_1000, 'RBO_RES_1000': _res_1000,
                                                'RBO_EXT_1000': _ext_1000, 'RBO_MIN_100': _min_100,
                                                'RBO_RES_100': _res_100, 'RBO_EXT_100': _ext_100}
        features_df = pd.DataFrame.from_dict(features_dict, orient='index')
        features_df.index.rename(['topic', 'var-1', 'var-2'], inplace=True)
        return features_df

    @staticmethod
    def jaccard_coefficient(st1: str, st2: str):
        st1_set = set(st1.split())
        st2_set = set(st2.split())
        union = st1_set.union(st2_set)
        intersect = st1_set.intersection(st2_set)
        return float(len(intersect) / len(union))

    @staticmethod
    def list_overlap(x, y):
        x_set = set(x)
        intersection = x_set.intersection(y)
        return len(intersection)


def main(args):
    queries_file = args.queries
    corpus = args.corpus
    results_file = args.results
    file_to_load = args.load
    generate = args.generate

    if generate:
        queries_file = ensure_file(queries_file)
        results_file = ensure_file(results_file)
        testing_feat = FeatureFactory(results_file, queries_file)
        features_df = testing_feat.generate_features()
        features_df.reset_index().to_json('features_{}_uqv.JSON'.format(corpus))
    else:
        if file_to_load is None:
            file = ensure_file('features_{}_uqv.JSON'.format(corpus))
        else:
            file = ensure_file(file_to_load)

        features_df = pd.read_json(file, dtype={'topic': str, 'var-1': str, 'var-2': str})
        features_df.reset_index(drop=True, inplace=True)
        features_df.set_index(['topic', 'var-1', 'var-2'], inplace=True)

    print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
