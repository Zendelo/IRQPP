import argparse
import glob
import multiprocessing as mp
import os
from functools import partial, reduce

import numpy as np
import pandas as pd

import dataparser as dp
from RBO import rbo_dict
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])

parser.add_argument('-g', '--group', help='group of queries to predict',
                    choices=['top', 'low', 'medh', 'medl', 'title'])
parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default=None,
                    choices=['all', 'low', 'low-0', 'high'])
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')
parser.add_argument('--generate', help="generate new features file", action="store_true")
parser.add_argument('--predict', help="generate new predictions", action="store_true")
parser.add_argument('--graphs', default=None, help="generate new features for graphs", choices=['asce', 'desc'])
parser.add_argument('-v', '--vars', default=None, type=int, help="number of variations, valid with graphs")

NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500, 1000)


def jaccard_coefficient(st1: str, st2: str):
    st1_set = set(st1.split())
    st2_set = set(st2.split())
    union = st1_set.union(st2_set)
    intersect = st1_set.intersection(st2_set)
    return float(len(intersect) / len(union))


def list_overlap(x, y):
    x_set = set(x)
    intersection = x_set.intersection(y)
    return len(intersection)


class QueryFeatureFactory:
    """TODO: At the moment this will save for each combination a separate pickle file, should change it to a pickle file
    that consists of all the calculations and then filter the relevant query variations from it"""

    def __init__(self, corpus, queries_group, vars_quantile, **kwargs):
        self.top_docs_overlap = kwargs.get('top_docs_overlap', 10)
        self.rbo_top = kwargs.get('rbo_top', 100)
        self.corpus = corpus
        self.queries_group = queries_group
        graphs = kwargs.get('graphs', None)
        if graphs:
            n = kwargs.get('n', None)
            assert n, 'Missing number of vars'
            self.__set_graph_paths(corpus, queries_group, graphs, n)
        else:
            self.__set_paths(corpus, queries_group, vars_quantile)
        _raw_res_data = dp.ResultsReader(self.results_file, 'trec')
        if queries_group == 'title':
            _title_res_data = dp.ResultsReader(self.title_res_file, 'trec')
            self.prediction_queries_res_data = _title_res_data
        else:
            self.prediction_queries_res_data = _raw_res_data
        self.queries_data = dp.QueriesTextParser(self.queries_full_file, 'uqv')
        self.topics_data = dp.QueriesTextParser(self.queries_topic_file)
        # Uncomment the next lines if you want to write the basic results of the topic queries.
        # write_basic_results(self.prediction_queries_res_data.data_df.loc[self.topics_data.queries_df['qid']], corpus,
        #                     queries_group)
        # exit()
        # These 2 DF used for the filtering method
        self.variations_data = dp.QueriesTextParser(self.queries_variations_file, 'uqv')
        self.quantile_variations_data = dp.QueriesTextParser(self.queries_quantile_vars, 'uqv')
        # _var_scores_df.loc[_var_scores_df['qid'].isin(_vars_list)]
        self.raw_res_data = _raw_res_data

        self.fused_data = dp.ResultsReader(self.fused_results_file, 'trec')
        self.query_vars = self.queries_data.query_vars

    @classmethod
    def __set_paths(cls, corpus, qgroup, vars_quantile):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        # cls.predictor = predictor
        _corpus_res_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}')
        _corpus_dat_dir = dp.ensure_dir(f'~/QppUqvProj/data/{corpus}')

        _results_file = f'{_corpus_res_dir}/test/raw/QL.res'
        cls.results_file = os.path.normpath(_results_file)
        dp.ensure_file(cls.results_file)

        _title_results_file = f'{_corpus_res_dir}/test/basic/QL.res'
        cls.title_res_file = os.path.normpath(_title_results_file)
        dp.ensure_file(cls.title_res_file)

        cls.queries_full_file = dp.ensure_file(f'{_corpus_dat_dir}/queries_{corpus}_UQV_full.stemmed.txt')

        # The variations file is used in the filter function - it consists of all the vars w/o the query at hand
        _queries_variations_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_wo_{qgroup}.txt'
        cls.queries_variations_file = dp.ensure_file(_queries_variations_file)

        # The vars quantile file is used in the filter function - it consists of the relevant vars quantile
        if vars_quantile == 'all':
            _queries_quantile_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_full.txt'
        else:
            _queries_quantile_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_{vars_quantile}_variants.txt'
        cls.queries_quantile_vars = dp.ensure_file(_queries_quantile_file)

        _queries_topic_file = f'{_corpus_dat_dir}/queries_{corpus}_{qgroup}.stemmed.txt'
        cls.queries_topic_file = dp.ensure_file(_queries_topic_file)

        _fused_results_file = f'{_corpus_res_dir}/test/fusion/QL.res'
        cls.fused_results_file = dp.ensure_file(_fused_results_file)

        # cls.output_dir = dp.ensure_dir(f'{_corpus_res_dir}/test/raw/')

        _predictions_out = f'{_corpus_res_dir}/uqvPredictions/referenceLists/{qgroup}/{vars_quantile}_vars/sim_as_pred/'
        cls.predictions_output_dir = dp.ensure_dir(_predictions_out)

        cls.pkl_dir = dp.ensure_dir(f'{_corpus_res_dir}/test/ref/pkl_files/')

    @classmethod
    def __set_graph_paths(cls, corpus, qgroup, direct, n):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        # cls.predictor = predictor
        _corpus_res_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}')
        _corpus_dat_dir = dp.ensure_dir(f'~/QppUqvProj/data/{corpus}')

        _graphs_base_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}')
        _graphs_res_dir = dp.ensure_dir(f'{_graphs_base_dir}/referenceLists/{qgroup}/{direct}/{n}_vars')
        _graphs_dat_dir = dp.ensure_dir(f'{_graphs_base_dir}/data')

        cls.number_of_vars = n

        _results_file = f'{_corpus_res_dir}/test/raw/QL.res'
        cls.results_file = os.path.normpath(_results_file)
        dp.ensure_file(cls.results_file)

        _title_results_file = f'{_corpus_res_dir}/test/basic/QL.res'
        cls.title_res_file = os.path.normpath(_title_results_file)
        dp.ensure_file(cls.title_res_file)

        _queries_full_file = f'{_corpus_dat_dir}/queries_{corpus}_UQV_full.stemmed.txt'
        cls.queries_full_file = dp.ensure_file(_queries_full_file)

        # The variations file is used in the filter function - it consists of all the vars w/o the query at hand
        _queries_variations_file = f'{_graphs_dat_dir}/{direct}/queries/queries_wo_{qgroup}_{n}_vars.txt'
        cls.queries_variations_file = dp.ensure_file(_queries_variations_file)
        cls.queries_quantile_vars = cls.queries_variations_file

        _queries_topic_file = f'{_corpus_dat_dir}/queries_{corpus}_{qgroup}.stemmed.txt'
        cls.queries_topic_file = dp.ensure_file(_queries_topic_file)

        _fused_results_file = f'{_corpus_res_dir}/test/fusion/QL.res'
        _fused_results_file = f'{_corpus_res_dir}/test/fusion/all_wo_title_fused_QL.res'
        cls.fused_results_file = dp.ensure_file(_fused_results_file)

        # cls.output_dir = dp.ensure_dir(f'{_graphs_res_dir}/test/raw/')

        cls.predictions_output_dir = dp.ensure_dir(f'{_graphs_res_dir}/sim_as_pred/')

        cls.pkl_dir = dp.ensure_dir(f'{_graphs_dat_dir}/pkl_files/features')

    def _calc_features(self):
        """This method calculates the similarity features for all the variations with the 'query at hand' i.e. the query
        that being predicted, including the query itself (if it's among the variations)"""

        _dict = {'topic': [], 'qid': [], 'Jac_coefficient': [], f'Top_{self.top_docs_overlap}_Docs_overlap': [],
                 f'RBO_EXT_{self.rbo_top}': [], f'RBO_FUSED_EXT_{self.rbo_top}': []}

        for topic in self.topics_data.queries_dict.keys():
            _topic = topic.split('-')[0]
            q_vars = self.query_vars.get(_topic)
            _dict['topic'] += [topic] * len(q_vars)
            res_dict = self.fused_data.get_res_dict_by_qid(_topic, top=self.rbo_top)
            topic_txt = self.topics_data.get_qid_txt(topic)
            topics_top_list = self.prediction_queries_res_data.get_docs_by_qid(topic, self.top_docs_overlap)
            # topics_top_list = self.title_res_data.get_docs_by_qid(topic, 25)
            topic_results_list = self.prediction_queries_res_data.get_res_dict_by_qid(topic, top=self.rbo_top)

            for var in q_vars:
                var_txt = self.queries_data.get_qid_txt(var)
                jc = jaccard_coefficient(topic_txt, var_txt)

                var_top_list = self.raw_res_data.get_docs_by_qid(var, self.top_docs_overlap)
                # var_top_list = self.raw_res_data.get_docs_by_qid(var, 25)
                docs_overlap = list_overlap(topics_top_list, var_top_list)

                # All RBO values are rounded to 10 decimal digits, to avoid float overflow
                var_results_list = self.raw_res_data.get_res_dict_by_qid(var, top=self.rbo_top)
                _rbo_scores_dict = rbo_dict(topic_results_list, var_results_list, p=0.95)
                rbo_ext_score = np.around(_rbo_scores_dict['ext'], 10)

                _fused_rbo_scores_dict = rbo_dict(res_dict, var_results_list, p=0.95)
                _rbo_fused_ext_score = np.around(_fused_rbo_scores_dict['ext'], 10)

                _dict['qid'] += [var]
                _dict['Jac_coefficient'] += [jc]
                _dict[f'Top_{self.top_docs_overlap}_Docs_overlap'] += [docs_overlap]
                _dict[f'RBO_EXT_{self.rbo_top}'] += [rbo_ext_score]
                _dict[f'RBO_FUSED_EXT_{self.rbo_top}'] += [_rbo_fused_ext_score]

        _df = pd.DataFrame.from_dict(_dict)
        # _df.set_index(['topic', 'qid'], inplace=True)
        return _df

    def _filter_queries(self, df):
        df.reset_index(inplace=True)
        # Remove the topic queries
        _df = df.loc[df['qid'].isin(self.variations_data.queries_df['qid'])]
        # Filter only the relevant quantile variations
        _df = _df.loc[_df['qid'].isin(self.quantile_variations_data.queries_df['qid'])]
        return _df

    def _soft_max_scores(self, df):
        _df = self._filter_queries(df)
        _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        _exp_df = _df.apply(np.exp)
        # For debugging purposes
        z_e = _exp_df.groupby(['topic']).sum()

        softmax_df = (_exp_df.groupby(['topic', 'qid']).sum() / z_e)
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return softmax_df

    def _average_scores(self, df):
        _df = self._filter_queries(df)
        # _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        # _exp_df = _df.apply(np.exp)
        # For debugging purposes
        avg_df = _df.groupby(['topic']).mean()

        # avg_df = (_df.groupby(['topic', 'qid']).mean())
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return avg_df

    def _max_norm_scores(self, df):
        # _df = self._filter_queries(df)
        _df = df
        _df.set_index(['topic', 'qid'], inplace=True)
        # For debugging purposes
        z_m = _df.groupby(['topic']).max()
        z_m.drop('qid', axis='columns', inplace=True)
        max_norm_df = (_df.groupby(['topic', 'qid']).sum() / z_m).fillna(0)
        # _temp = softmax_df.dropna()
        # For debugging purposes
        return max_norm_df

    def _sum_scores(self, df):
        _df = df
        # filter only variations different from original query
        # _df = self._filter_queries(df)
        z_n = _df.groupby(['topic']).sum()
        z_n.drop('qid', axis='columns', inplace=True)
        # All nan values will be filled with 0
        norm_df = (_df.groupby(['topic', 'qid']).sum() / z_n).fillna(0)
        return norm_df

    def divide_by_size(self, df):
        # _df = df
        # filter only variations different from original query
        _df = self._filter_queries(df)
        z_n = _df.groupby(['topic']).count()
        z_n.drop('qid', axis='columns', inplace=True)
        # All nan values will be filled with 0
        # norm_df = (_df.groupby(['topic', 'qid']) / z_n).fillna('!@#!@#!@#!')
        _df.set_index(['topic', 'qid'], inplace=True)
        norm_df = _df / z_n
        return norm_df

    def __load_features_df(self, _file_name):
        """The method will try to load the features DF from a pkl file, if it fails it will generate a new df
        and save it"""
        try:
            # Will try loading a DF, if fails will generate and save a new one
            file_to_load = dp.ensure_file(_file_name)
            _df = pd.read_pickle(file_to_load)
        except AssertionError:
            print(f'\nFailed to load {_file_name}')
            print(f'Will generate {self.pkl_dir.rsplit("/")[-1]} vars {self.queries_group}_query_features '
                  f'features and save')
            _df = self._calc_features()
            _df.to_pickle(_file_name)
        n = self.top_docs_overlap
        _df[f'Top_{n}_Docs_overlap'] = _df[f'Top_{n}_Docs_overlap'] / n
        return _df

    def __get_pkl_file_name(self):
        _file = '{}/{}_queries_{}_RBO_{}_TopDocs_{}.pkl'.format(self.pkl_dir, self.queries_group, self.corpus,
                                                                self.rbo_top, self.top_docs_overlap)
        return _file

    def generate_features(self, load_from_pkl=True):
        """If `load_from_pkl` is True the method will try to load the features DF from a pkl file, otherwise
        it will generate a new df and save it"""
        _file = self.__get_pkl_file_name()
        if load_from_pkl:
            _df = self.__load_features_df(_file)
        else:
            _df = self._calc_features()
            _df.to_pickle(_file)
        return self.divide_by_size(_df)
        # return _df
        # return self._soft_max_scores(_df)
        # return self._sum_scores(_df)
        # return self._average_scores(_df)
        # return self._max_norm_scores(_df)

    def save_predictions(self, df: pd.DataFrame):
        _df = self._filter_queries(df)
        _df = _df.groupby('topic').mean()
        _df = dp.convert_vid_to_qid(_df)
        _rboP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/rboP/predictions')
        _FrboP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/FrboP/predictions')
        _topDocsP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/topDocsP/predictions')
        _jcP_dir = dp.ensure_dir(f'{self.predictions_output_dir}/jcP/predictions')

        _df[f'RBO_EXT_{self.rbo_top}'].to_csv(f'{_rboP_dir}/predictions-{self.rbo_top}', sep=' ')
        _df[f'RBO_FUSED_EXT_{self.rbo_top}'].to_csv(f'{_FrboP_dir}/predictions-{self.rbo_top}', sep=' ')
        _df[f'Top_{self.top_docs_overlap}_Docs_overlap'].to_csv(f'{_topDocsP_dir}/predictions-{self.top_docs_overlap}',
                                                                sep=' ')
        _df['Jac_coefficient'].to_csv(f'{_jcP_dir}/predictions-{self.rbo_top}', sep=' ')

    def generate_predictions(self, load_from_pkl=True):
        _file = self.__get_pkl_file_name()
        if load_from_pkl:
            _df = self.__load_features_df(_file)
        else:
            _df = self._calc_features()
            _df.to_pickle(_file)
        self.save_predictions(_df)


def features_loader(file_to_load, corpus):
    if file_to_load is None:
        file = dp.ensure_file('features_{}_uqv.JSON'.format(corpus))
    else:
        file = dp.ensure_file(file_to_load)

    features_df = pd.read_json(file, dtype={'topic': str, 'qid': str})
    features_df.reset_index(drop=True, inplace=True)
    features_df.set_index(['topic', 'qid'], inplace=True)
    features_df.rename(index=lambda x: x.split('-')[0], level=0, inplace=True)
    features_df.sort_values(['topic', 'qid'], axis=0, inplace=True)
    return features_df


def write_basic_results(df: pd.DataFrame, corpus, qgroup):
    """The function is used to save basic predictions of a given queries set"""
    _df = dp.convert_vid_to_qid(df)
    _df.insert(loc=0, column='trec_Q0', value='Q0')
    _df.insert(loc=4, column='trec_indri', value='indri')
    _file_path = f'~/QppUqvProj/Results/{corpus}/test/ref/QL_{qgroup}.res'
    # dp.ensure_dir(os.path.normpath(os.path.expanduser(_file_path)))
    _df.to_csv(_file_path, sep=" ", header=False, index=True)


def run_predictions_process(n, corpus, queries_group, quantile):
    sim_ref_pred = QueryFeatureFactory(corpus, queries_group, quantile, rbo_top=n, top_docs_overlap=n)
    sim_ref_pred.generate_predictions()
    return sim_ref_pred


def load_full_features_df(corpus, queries_group, quantile):
    features_obj = QueryFeatureFactory(corpus, queries_group, quantile)
    pkl_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/ref/pkl_files/')
    _list = []
    last_df = pd.DataFrame()
    for n in {5, 10, 25, 50, 100, 250, 500}:
        _file = f'{pkl_dir}/{queries_group}_queries_{corpus}_RBO_{n}_TopDocs_{n}.pkl'
        try:
            dp.ensure_file(_file)
            _df = pd.read_pickle(_file).set_index(['topic', 'qid'])
            _df[f'Top_{n}_Docs_overlap'] = _df[f'Top_{n}_Docs_overlap'] / n
            _list.append(_df.drop('Jac_coefficient', axis=1))
            last_df = _df['Jac_coefficient']
        except AssertionError:
            print(f'!! Warning !! The file {_file} is missing')
    df = pd.concat(_list + [last_df], axis=1)
    _path = f'~/QppUqvProj/Results/{corpus}/test/ref'
    _path = dp.ensure_dir(_path)
    return features_obj.divide_by_size(df).reset_index().to_json(
        f'{_path}/{queries_group}_query_{quantile}_variations_features_{corpus}_uqv.JSON')


def main(args):
    corpus = args.corpus
    generate = args.generate
    predict = args.predict
    queries_group = args.group
    file_to_load = args.load
    quantile = args.quantile
    graphs = args.graphs
    number_of_vars = args.vars

    # Debugging
    # corpus = 'ClueWeb12B'
    # corpus = 'ROBUST'
    # print('\n------+++^+++------ Debugging !! ------+++^+++------\n')
    # queries_group = 'title'
    # quantile = 'all'
    # testing_feat = QueryFeatureFactory('ROBUST', 'title', 'all')
    # norm_features_df = testing_feat.generate_features()
    # norm_features_df.reset_index().to_json('query_features_{}_uqv.JSON'.format(corpus))
    # return

    cores = mp.cpu_count() - 1

    if generate:
        testing_feat = QueryFeatureFactory(corpus, queries_group, quantile)
        norm_features_df = testing_feat.generate_features()

        _path = f'~/QppUqvProj/Results/{corpus}/test/ref'
        _path = dp.ensure_dir(_path)
        norm_features_df.reset_index().to_json(
            f'{_path}/{queries_group}_query_{quantile}_variations_features_{corpus}_uqv.JSON')
    elif predict:
        with mp.Pool(processes=cores) as pool:
            sim_ref_pred = pool.map(
                partial(run_predictions_process, corpus=corpus, queries_group=queries_group, quantile=quantile),
                NUMBER_OF_DOCS)
    elif graphs:
        assert number_of_vars, 'Missing number of variations'
        testing_feat = QueryFeatureFactory(corpus, queries_group, quantile, graphs=graphs)
        norm_features_df = testing_feat.generate_features()

        _path = f'~/QppUqvProj/Graphs/{corpus}/data/ref/'
        _path = dp.ensure_dir(_path)
        norm_features_df.reset_index().to_json(
            f'{_path}/{queries_group}_query_{quantile}_variations_features_{corpus}_uqv.JSON')

    elif file_to_load:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)

    else:
        load_full_features_df(corpus, queries_group, quantile)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
