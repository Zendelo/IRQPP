from qpputils.dataparser import QueriesTextParser, ResultsReader, ensure_file
from itertools import combinations
from collections import defaultdict
from RBO import rbo_dict
import pandas as pd
import numpy as np

import argparse
from Timer import Timer

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3.6 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-q', '--queries', metavar='queries.txt', help='path to UQV queries txt file')
parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-r', '--results', default=None, type=str, help='QL.res file of the queries')
parser.add_argument('-f', '--fused', default=None, type=str, help='fusedQL.res file of the queries')
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')

parser.add_argument('--generate', help="generate new predictions", action="store_true")


class FeatureFactory:
    def __init__(self, queries_file, results_file, fused):
        self.res_data = ResultsReader(results_file, 'trec')
        self.queries_data = QueriesTextParser(queries_file, 'uqv')
        # A simple sanity check to make sure number of results and query variations is identical
        _x = [len(i) for i in self.res_data.query_vars.values()]
        _z = [len(i) for i in self.queries_data.query_vars.values()]
        assert _x == _z, 'Results and Queries files don\'t match'
        self.fused_data = ResultsReader(fused, 'trec')
        self.query_vars = self.queries_data.query_vars
        self.features_index = self._create_query_var_pairs()
        self.writer = pd.ExcelWriter('test_ROBUST_output.xlsx')

    def _create_query_var_pairs(self):
        feature_keys = defaultdict(list)
        for qid, variations in self.query_vars.items():
            feature_keys[qid] = list(combinations(variations, 2))
        return feature_keys

    def _calc_features(self):
        _dict = {'topic': [], 'qid': [], 'Jac_coefficient': [], 'Top_10_Docs_overlap': [], 'RBO_MIN_1000': [],
                 'RBO_EXT_1000': [], 'RBO_MIN_100': [], 'RBO_EXT_100': [],
                 'RBO_FUSED_MIN_100': [], 'RBO_FUSED_EXT_100': [], 'RBO_FUSED_MIN_1000': [],
                 'RBO_FUSED_EXT_1000': []}
        for topic, pairs in self.features_index.items():
            _dict['topic'] += [topic] * 2 * len(pairs)
            dt_100 = self.fused_data.get_res_dict_by_qid(topic, top=100)
            dt_1000 = self.fused_data.get_res_dict_by_qid(topic, top=1000)
            for q1, q2 in pairs:
                txt1 = self.queries_data.get_qid_txt(q1)
                txt2 = self.queries_data.get_qid_txt(q2)
                jc = self.jaccard_coefficient(txt1, txt2)

                l1 = self.res_data.get_docs_by_qid(q1, 10)
                l2 = self.res_data.get_docs_by_qid(q2, 10)
                docs_overlap = self.list_overlap(l1, l2)

                # All RBO values are rounded to 10 decimal digits, to avoid float overflow
                d1 = self.res_data.get_res_dict_by_qid(q1, top=1000)
                d2 = self.res_data.get_res_dict_by_qid(q2, top=1000)
                _rbo_scores_1000 = rbo_dict(d1, d2, p=0.95)
                _min_1000 = np.around(_rbo_scores_1000['min'], 10)
                # _res_1000 = np.around(_rbo_scores_1000['res'], 10)
                _ext_1000 = np.around(_rbo_scores_1000['ext'], 10)

                d1 = self.res_data.get_res_dict_by_qid(q1, top=100)
                d2 = self.res_data.get_res_dict_by_qid(q2, top=100)
                _rbo_scores_100 = rbo_dict(d1, d2, p=0.95)
                _min_100 = np.around(_rbo_scores_100['min'], 10)
                # _res_100 = np.around(_rbo_scores_100['res'], 10)
                _ext_100 = np.around(_rbo_scores_100['ext'], 10)

                _fused_rbo_scores_100 = rbo_dict(dt_100, d1, p=0.95)
                _q1_fmin_100 = np.around(_fused_rbo_scores_100['min'], 10)
                # _q1_fres_100 = np.around(_fused_rbo_scores_100['res'], 10)
                _q1_fext_100 = np.around(_fused_rbo_scores_100['ext'], 10)

                _fused_rbo_scores_1000 = rbo_dict(dt_1000, d1, p=0.95)
                _q1_fmin_1000 = np.around(_fused_rbo_scores_1000['min'], 10)
                # _q1_fres_1000 = np.around(_fused_rbo_scores_1000['res'], 10)
                _q1_fext_1000 = np.around(_fused_rbo_scores_1000['ext'], 10)

                _fused_rbo_scores_100 = rbo_dict(dt_100, d2, p=0.95)
                _q2_fmin_100 = np.around(_fused_rbo_scores_100['min'], 10)
                # _q2_fres_100 = np.around(_fused_rbo_scores_100['res'], 10)
                _q2_fext_100 = np.around(_fused_rbo_scores_100['ext'], 10)

                _fused_rbo_scores_1000 = rbo_dict(dt_1000, d2, p=0.95)
                _q2_fmin_1000 = np.around(_fused_rbo_scores_1000['min'], 10)
                # _q2_fres_1000 = np.around(_fused_rbo_scores_1000['res'], 10)
                _q2_fext_1000 = np.around(_fused_rbo_scores_1000['ext'], 10)

                _dict['qid'] += [q1, q2]
                _dict['Jac_coefficient'] += [jc] * 2
                _dict['Top_10_Docs_overlap'] += [docs_overlap] * 2
                _dict['RBO_MIN_1000'] += [_min_1000] * 2
                # _dict['RBO_RES_1000'] += [_res_1000] * 2
                _dict['RBO_EXT_1000'] += [_ext_1000] * 2
                _dict['RBO_MIN_100'] += [_min_100] * 2
                # _dict['RBO_RES_100'] += [_res_100] * 2
                _dict['RBO_EXT_100'] += [_ext_100] * 2
                _dict['RBO_FUSED_MIN_1000'] += [_q1_fmin_1000, _q2_fmin_1000]
                # _dict['RBO_FUSED_RES_1000'] += [_q1_fres_1000, _q2_fres_1000]
                _dict['RBO_FUSED_EXT_1000'] += [_q1_fext_1000, _q2_fext_1000]
                _dict['RBO_FUSED_MIN_100'] += [_q1_fmin_100, _q2_fmin_100]
                # _dict['RBO_FUSED_RES_100'] += [_q1_fres_100, _q2_fres_100]
                _dict['RBO_FUSED_EXT_100'] += [_q1_fext_100, _q2_fext_100]

        _df = pd.DataFrame.from_dict(_dict)
        _df.set_index(['topic', 'qid'], inplace=True)
        return _df

    def _sum_scores(self, df):
        _exp_df = df.apply(np.exp)
        # For debugging purposes
        df.to_excel(self.writer, 'normal_scores')
        _exp_df.to_excel(self.writer, 'exp_scores')

        z_n = df.groupby(['topic']).sum()
        z_e = _exp_df.groupby(['topic']).sum()

        norm_df = (df.groupby(['topic', 'qid']).sum() / z_n)
        softmax_df = (_exp_df.groupby(['topic', 'qid']).sum() / z_e)
        # For debugging purposes
        norm_df.to_excel(self.writer, 'zn_scores')
        softmax_df.to_excel(self.writer, 'ze_scores')
        self.writer.save()
        return norm_df, softmax_df

    def generate_features(self):
        _df = self._calc_features()
        return self._sum_scores(_df)

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


def features_loader(file_to_load, corpus):
    if file_to_load is None:
        file = ensure_file('features_{}_uqv.JSON'.format(corpus))
    else:
        file = ensure_file(file_to_load)

    features_df = pd.read_json(file, dtype={'topic': str, 'qid': str})
    features_df.reset_index(drop=True, inplace=True)
    features_df.set_index(['topic', 'qid'], inplace=True)
    features_df.rename(index=lambda x: x.split('-')[0], level=0, inplace=True)
    features_df.sort_values(['topic', 'qid'], axis=0, inplace=True)
    return features_df


def main(args):
    queries_file = args.queries
    corpus = args.corpus
    results_file = args.results
    fused_res_file = args.fused
    file_to_load = args.load
    generate = args.generate

    assert not queries_file.endswith('.xml'), 'Submit text queries file, not XML'

    if generate:
        queries_file = ensure_file(queries_file)
        results_file = ensure_file(results_file)
        testing_feat = FeatureFactory(queries_file, results_file, fused_res_file)
        norm_features_df, exp_features_df = testing_feat.generate_features()
        norm_features_df.reset_index().to_json('norm_features_{}_uqv.JSON'.format(corpus))
        exp_features_df.reset_index().to_json('exp_features_{}_uqv.JSON'.format(corpus))
    else:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
