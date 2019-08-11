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
from query_features import QueryFeatureFactory, load_full_features_df

# Define the Font for the plots
plt.rcParams.update({'font.size': 45, 'font.family': 'serif', 'font.weight': 'normal'})
# plt.rcParams.update({'font.size': 55, 'font.family': 'serif', 'font.weight': 'normal'})

parser = argparse.ArgumentParser(description='Results generator for QPP with Reference lists graphs',
                                 usage='',
                                 epilog='')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-g', '--queries_group', default='title', help='The queries group to be used',
                    choices=['title', 'top', 'low', 'med'])
parser.add_argument('--generate', action='store_true')
parser.add_argument('--plot', action='store_true', help='Plot graphs')
parser.add_argument('--nocache', action='store_false', help='Add this option in order to generate all pkl files')

PREDICTORS_WO_QF = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv']
PREDICTORS_QF = ['qf', 'uef/qf']

PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']

# PREDICTORS = PREDICTORS_WO_QF + PRE_RET_PREDICTORS
# PREDICTORS.remove('rsd')
PREDICTORS = ['preret/AvgSCQTFIDF', 'preret/MaxIDF', 'uef/clarity', 'wig']

NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500)
# SIMILARITY_FUNCTIONS = {'Jac_coefficient': 'jac', 'Top_10_Docs_overlap': 'sim', 'RBO_EXT_100': 'rbo',
#                         'RBO_FUSED_EXT_100': 'rbof'}

SIMILARITY_FUNCTIONS = {'Top_Docs_overlap': 'sim', 'RBO': 'rbo'}

MARKERS = ['+', 'x', '.', '*', 'X', 'v']
LINE_STYLES = ['-', '--', ':', ':']
# MARKERS_STYLE = [''.join(i) for i in itertools.product(LINE_STYLES, MARKERS)]
# MARKERS = ['-^', '-v', '-D', '-x', '-h', '-H', 'p-', 's-', '--v', '--1', '--2', '--D', '--x', '--h', '--H', '^-.',
#            '-.v', '1-.', '2-.', '-.D', '-.x', '-.h', '-.H', '3-.', '4-.', 's-.', 'p-.', '+-.', '*-.']

COLORS = ['#2A88AA', '#93BEDB', '#203D78', '#60615C', '#E57270']
# COLORS = ['#1D2735', '#135960', '#2F8F6D', '#8DC05F']
NAMES_DICT = {'rbo': 'Ref-RBO', 'sim': 'Ref-Overlap', 'wig': 'WIG', 'rsd': 'RSD', 'preret/AvgSCQTFIDF': 'AvgSCQ',
              'preret/AvgVarTFIDF': 'AvgVar', 'uef/clarity': 'UEF(Clarity)', 'preret/MaxIDF': 'MaxIDF',
              'asce': 'Ascending', 'desc': 'Descending', 'ClueWeb12B': 'CW12', 'ROBUST': 'ROBUST'}


def plot_graphs(_df: pd.DataFrame, simi, corpus):
    _df['result'] = pd.to_numeric(_df['result'])
    df = _df.loc[(_df['predictor'].isin(PREDICTORS)) & (_df['sim_func'] == simi)]
    for direction, sub_df in df.groupby('direction'):
        mar = 0
        for predictor, pdf in sub_df.drop('direction', axis=1).groupby('predictor'):
            pdf.set_index('n_vars')['result'].plot(legend=True,
                                                   title=f'{NAMES_DICT[corpus]} {NAMES_DICT[direction]}',
                                                   marker=MARKERS[mar], linestyle=LINE_STYLES[mar],
                                                   label=NAMES_DICT[predictor], linewidth=5, markersize=15, mew=1,
                                                   # markerfacecolor='None',
                                                   color=COLORS[mar])
            plt.xlabel('# of reference queries')
            plt.ylabel("Pearson")
            plt.legend()
            mar += 1
        plt.show()


class GraphsFactory:
    def __init__(self, corpus, max_n=20, corr_measure='pearson', load_from_pkl=True, queries_group='title'):
        self.group = queries_group
        self.corr_measure = corr_measure
        self.load_from_pkl = load_from_pkl
        self.__set_paths(corpus, queries_group)
        self.corpus = corpus
        self.queries_obj = dp.QueriesTextParser(self.queries_file)
        self.queries_obj.queries_df = add_topic_to_qdf(self.queries_obj.queries_df)
        self.raw_ap_obj = dp.ResultsReader(self.raw_ap_file, 'ap')
        self.max_n = min(self.queries_obj.queries_df.groupby('topic').count().max()['qid'], max_n)
        self.basic_results_dict = defaultdict(float)
        self.__initialize_basic_results_dict()

    @classmethod
    def __set_paths(cls, corpus, group):
        _corpus_test_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/')

        # Basic predictions dir
        cls.basic_predictions_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/{group}/')
        # AP file to pick variations according to AP
        cls.raw_ap_file = dp.ensure_file(f'{_corpus_test_dir}/raw/QLmap1000')
        # AP file for the cross validation process
        cls.query_ap_file = dp.ensure_file(f'{_corpus_test_dir}/ref/QLmap1000-{group}')
        # CV folds mapping file
        cls.cv_map_file = dp.ensure_file(f'{_corpus_test_dir}/2_folds_30_repetitions.json')
        # Queries file with all the variations except the ones to be predicted
        cls.queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_{group}.txt')
        # The data dir for the Graphs
        cls.data_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/data')
        # The results base dir for the Graphs
        cls.results_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/referenceLists/{group}')

    def create_query_files(self, n):
        for direction, func in {('asce', filter_n_low_queries), ('desc', filter_n_top_queries)}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direction}/queries')
            _file = f'{_dir}/queries_wo_{self.group}_{n}_vars.txt'
            _df = func(self.queries_obj.queries_df, self.raw_ap_obj, n)
            _df[['qid', 'text']].to_csv(_file, sep=":", header=False, index=False)

    def generate_features(self, n):
        print(f'\n---Generating Features for {n} vars---\n')
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}/features')
            _feat_obj = QueryFeatureFactory(corpus=self.corpus, queries_group=self.group, vars_quantile='all',
                                            graphs=direct, n=n)
            _df = load_full_features_df(features_factory_obj=_feat_obj)
            _df.reset_index().to_json(f'{_dir}/{self.group}_query_{n}_variations_features_{self.corpus}_uqv.JSON')

    def generate_sim_predictions(self, k):
        print(f'\n---Generating sim predictions {k} docs---\n')
        load_pickle = self.load_from_pkl
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}')
            for n in range(1, self.max_n + 1):
                sim_ref_pred = QueryFeatureFactory(self.corpus, queries_group=self.group, vars_quantile='all',
                                                   rbo_top=k, top_docs_overlap=k, graphs=direct, n=n)
                sim_ref_pred.generate_predictions(load_pickle)
                load_pickle = True

    def generate_qpp_reference_predictions(self, predictor):
        print(f'\n---Generating qpp ref predictions with {predictor}---\n')
        for direct in {'asce', 'desc'}:
            _dir = dp.ensure_dir(f'{self.data_dir}/{direct}')
            for n in range(1, self.max_n + 1):
                qpp_ref = QueryPredictionRef(predictor, self.corpus, qgroup=self.group, vars_quantile='all',
                                             graphs=direct, n=n)
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

        mean = self.basic_results_dict.get(predictor, None)
        assert mean, f'self.basic_results_dict couldn\'t get {predictor}'
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

    def generate_results_df(self, cores=None, load_from_pkl=None):
        _pkl_file = f'{self.data_dir}/pkl_files/full_results_df_{self.max_n}_{self.corpus}_{self.corr_measure}_{self.group}.pkl'
        if load_from_pkl:
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
    generate = args.generate
    load_cache = args.nocache
    plot = args.plot
    queries_group = args.queries_group

    # Debugging
    # print('\n------+++^+++------ Debugging !! ------+++^+++------\n')
    # corpus = 'ROBUST'
    # corpus = 'ClueWeb12B'
    # generate = True
    # plot = True

    if not corpus:
        return

    testing = GraphsFactory(corpus, max_n=40, load_from_pkl=load_cache, queries_group=queries_group)
    # testing.generate_results_df(4)
    # exit()

    cores = mp.cpu_count() - 1

    if generate:
        for n in range(1, testing.max_n + 1):
            testing.create_query_files(n)

        """The first run will generate the pkl files, all succeeding runs will load and use it"""
        testing.generate_features(1)
        with mp.Pool(processes=cores) as pool:
            pool.map(testing.generate_features, range(2, testing.max_n + 1))
            print('Finished features generating')
            # pool.map(testing.generate_sim_predictions, NUMBER_OF_DOCS)
            # print('Finished sim predictions')
            pool.map(testing.generate_qpp_reference_predictions, PREDICTORS)
            print('Finished QppRef generation')
        pool.close()
    load_from_pkl = not generate
    full_results_df = testing.generate_results_df(load_from_pkl=load_from_pkl, cores=cores)
    print(full_results_df)

    if plot:
        plot_graphs(full_results_df, 'rbo', corpus)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
