import argparse
import itertools
import multiprocessing as mp
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

import dataparser as dp
from Timer.timer import Timer
from crossval import CrossValidation
from qpp_ref import QueryPredictionRef
from queries_pre_process import filter_n_top_queries, filter_n_low_queries, add_topic_to_qdf
from query_features import QueryFeatureFactory

parser = argparse.ArgumentParser(description='Results generator for QPP with Reference lists graphs',
                                 usage='',
                                 epilog='')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--preret', action='store_true')
parser.add_argument('--nocache', action='store_false', help='Add this option in order to generate all pkl files')

PREDICTORS_WO_QF = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv']
PREDICTORS_QF = ['qf', 'uef/qf']

PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']

PREDICTORS = PREDICTORS_WO_QF + PREDICTORS_QF + PRE_RET_PREDICTORS
PREDICTORS.remove('rsd')

NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500, 1000)
SIMILARITY_FUNCTIONS = {'Jac_coefficient': 'jac', 'Top_10_Docs_overlap': 'sim', 'RBO_EXT_100': 'rbo',
                        'RBO_FUSED_EXT_100': 'rbof'}
# Filter out filled markers and marker settings that do nothing.
MARKERS = ['x', '+', 'v', '3', 'X']
LINE_STYLES = ['-', ':', '--']
MARKERS_STYLE = [''.join(i) for i in itertools.product(MARKERS, LINE_STYLES)]


def plot_graphs(df: pd.DataFrame):
    df['result'] = pd.to_numeric(df['result'])
    for index, _df in df.groupby(['direction', 'sim_func']):
        _df = _df.set_index('n_vars')
        _dict = {}
        for predictor, sub_df in _df.groupby('predictor')['result']:
            _dict[predictor] = sub_df
        _df = pd.DataFrame(_dict)
        _df.plot(style=MARKERS_STYLE[:len(_df.columns)], grid=True,
                 title=f'Direction: {index[0].capitalize()} Similarity function: {index[1].upper()}')
        plt.xlabel('Maximum Number of Variants')
        plt.ylabel('Correlation')
        plt.show()


class GraphsFactory:
    def __init__(self, corpus, max_n=20, corr_measure='pearson', load_from_pkl=True):
        self.corr_measure = corr_measure
        self.load_from_pkl = load_from_pkl
        self.__set_paths(corpus)
        self.corpus = corpus
        self.queries_obj = dp.QueriesTextParser(self.queries_file)
        self.queries_obj.queries_df = add_topic_to_qdf(self.queries_obj.queries_df)
        self.raw_ap_obj = dp.ResultsReader(self.raw_ap_file, 'ap')
        self.max_n = min(self.queries_obj.queries_df.groupby('topic').count().max()['qid'], max_n)
        self.basic_results_dict = defaultdict(float)
        self.__initialize_basic_results_dict()

    @classmethod
    def __set_paths(cls, corpus):
        _corpus_test_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/')

        # Basic predictions dir
        cls.basic_predictions_dir = dp.ensure_dir(f'~/QppUqvProj/Results/ROBUST/basicPredictions/title/')
        # AP file to pick variations according to AP
        cls.raw_ap_file = dp.ensure_file(f'{_corpus_test_dir}/raw/QLmap1000')
        # AP file for the cross validation process
        cls.query_ap_file = dp.ensure_file(f'{_corpus_test_dir}/ref/QLmap1000-title')
        # CV folds mapping file
        cls.cv_map_file = dp.ensure_file(f'{_corpus_test_dir}/2_folds_30_repetitions.json')
        # Queries file with all the variations except the ones to be predicted
        cls.queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_title.txt')
        # The data dir for the Graphs
        cls.data_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/data')
        # The results base dir for the Graphs
        cls.results_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/referenceLists/title')

    def create_query_files(self, n):
        for direction, func in {('asce', filter_n_low_queries), ('desc', filter_n_top_queries)}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direction}/queries')
            _file = f'{_dir}/queries_wo_title_{n}_vars.txt'
            _df = func(self.queries_obj.queries_df, self.raw_ap_obj, n)
            _df[['qid', 'text']].to_csv(_file, sep=":", header=False, index=False)

    def generate_features(self, n):
        print(f'\n---Generating Features for {n} vars---\n')
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}/features')
            _feat_obj = QueryFeatureFactory(corpus=self.corpus, queries_group='title', vars_quantile='all',
                                            graphs=direct, n=n)
            _df = _feat_obj.generate_features(self.load_from_pkl)
            _df.reset_index().to_json(f'{_dir}/title_query_{n}_variations_features_{self.corpus}_uqv.JSON')

    def generate_sim_predictions(self, k):
        print(f'\n---Generating sim predictions {k} docs---\n')
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}')
            for n in range(1, self.max_n + 1):
                sim_ref_pred = QueryFeatureFactory(self.corpus, queries_group='title', vars_quantile='all', rbo_top=k,
                                                   top_docs_overlap=k, graphs=direct, n=n)
                sim_ref_pred.generate_predictions(self.load_from_pkl)

    def generate_qpp_reference_predictions(self, predictor):
        print(f'\n---Generating qpp ref predictions with {predictor}---\n')
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}')
            for n in range(1, self.max_n + 1):
                qpp_ref = QueryPredictionRef(predictor, self.corpus, qgroup='title', vars_quantile='all', graphs=direct,
                                             n=n)
                qpp_ref.calc_queries()

    def __initialize_basic_results_dict(self):
        _pkl_file = f'{self.data_dir}/pkl_files/basic_results_dict_{self.corpus}_{self.corr_measure}.pkl'
        if self.load_from_pkl:
            try:
                file_to_load = dp.ensure_file(_pkl_file)
                with open(file_to_load, 'rb') as handle:
                    self.basic_results_dict = pickle.load(handle)
            except AssertionError:
                print(f'\nFailed to load {_pkl_file}')
                print(f'Will generate {_pkl_file} and save')
                for predictor in PREDICTORS:
                    self.calc_single_query_result(predictor)
                with open(_pkl_file, 'wb') as handle:
                    pickle.dump(self.basic_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            for predictor in PREDICTORS:
                self.calc_single_query_result(predictor)
            with open(_pkl_file, 'wb') as handle:
                pickle.dump(self.basic_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def calc_single_query_result(self, predictor):
        print(f'\n---Generating {predictor} 0 vars results---\n')
        _predictions_dir = dp.ensure_dir(f'{self.basic_predictions_dir}/{predictor}/predictions')
        cv_obj = CrossValidation(k=2, rep=30, file_to_load=self.cv_map_file, predictions_dir=_predictions_dir,
                                 load=True, ap_file=self.query_ap_file, test=self.corr_measure)
        mean = cv_obj.calc_test_results()
        self.basic_results_dict[predictor] = mean

    def _calc_general_model_result(self, direct, predictor, sim_func):
        print(f'\n---Generating {predictor}-{sim_func} {direct} results---\n')
        _dict = defaultdict(list)

        def append_to_full_results_dict(_mean, _n):
            _dict['direction'].append(direct)
            _dict['predictor'].append(predictor)
            _dict['sim_func'].append(sim_func)
            _dict['n_vars'].append(_n)
            _dict['result'].append(_mean)

        mean = self.basic_results_dict.get(predictor, self.calc_single_query_result(predictor))
        append_to_full_results_dict(mean, 0)
        _dir = f'{self.results_dir}/{direct}'
        for n in range(1, self.max_n + 1):
            _predictions_dir = dp.ensure_dir(f'{_dir}/{n}_vars/general/{sim_func}/{predictor}/predictions')
            cv_obj = CrossValidation(k=2, rep=30, file_to_load=self.cv_map_file, predictions_dir=_predictions_dir,
                                     load=True, ap_file=self.query_ap_file, test=self.corr_measure)
            mean = cv_obj.calc_test_results()
            append_to_full_results_dict(mean, n)
        _df = pd.DataFrame.from_dict(_dict)
        return _df

    def generate_results_df(self, cores):
        _pkl_file = f'{self.data_dir}/pkl_files/full_results_df_{self.max_n}_{self.corpus}_{self.corr_measure}.pkl'
        if self.load_from_pkl:
            try:
                file_to_load = dp.ensure_file(_pkl_file)
                full_results_df = pd.read_pickle(file_to_load)
            except AssertionError:
                print(f'\nFailed to load {_pkl_file}')
                print(f'Will generate {_pkl_file} and save')
                with mp.Pool(processes=cores) as pool:
                    result = pool.starmap(self._calc_general_model_result,
                                          itertools.product({'asce', 'desc'}, PREDICTORS,
                                                            SIMILARITY_FUNCTIONS.values()))
                pool.close()
                full_results_df = pd.concat(result, axis=0)
                full_results_df.to_pickle(_pkl_file)
        else:
            with mp.Pool(processes=cores) as pool:
                result = pool.starmap(self._calc_general_model_result,
                                      itertools.product({'asce', 'desc'}, PREDICTORS,
                                                        SIMILARITY_FUNCTIONS.values()))
            pool.close()
            full_results_df = pd.concat(result, axis=0)
            full_results_df.to_pickle(_pkl_file)
        return full_results_df


def main(args):
    corpus = args.corpus
    pre_ret = args.preret
    load_cache = args.nocache

    if not corpus:
        return

    # corpus = 'ROBUST'

    testing = GraphsFactory(corpus, max_n=40, load_from_pkl=load_cache)
    # testing.generate_results_df(4)
    # exit()

    for n in range(1, testing.max_n + 1):
        testing.create_query_files(n)

    cores = mp.cpu_count() - 1
    """The first run will generate the pkl files, all succeeding runs will load and use it"""
    testing.generate_features(1)
    with mp.Pool(processes=cores) as pool:
        pool.map(testing.generate_features, range(2, testing.max_n + 1))
        print('Finished features generating')
        pool.map(testing.generate_sim_predictions, NUMBER_OF_DOCS)
        print('Finished sim predictions')
        pool.map(testing.generate_qpp_reference_predictions, PREDICTORS)
        print('Finished QppRef generation')
    pool.close()

    full_results_df = testing.generate_results_df(cores)

    print(full_results_df)

    # plot_graphs(full_results_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
