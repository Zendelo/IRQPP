import argparse
import itertools
from collections import defaultdict
from glob import glob
from shutil import copy2
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dataparser as dp
from Timer.timer import Timer
from crossval import CrossValidation

# Define the Font for the plots
plt.rcParams.update({'font.size': 45, 'font.family': 'serif'})
plt.rcParams.update({'font.size': 45, 'font.family': 'serif', 'font.weight': 'normal'})

"""The next three lines are used to force matplotlib to use font-Type-1 """
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

parser = argparse.ArgumentParser(description='Query Prediction Using Reference lists',
                                 usage='python3.6 qpp_ref.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the files')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--generate', action='store_true')
parser.add_argument('--nocache', action='store_false', help='Add this option in order to generate all pkl files')

PREDICTORS_WO_QF = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'uef/clarity', 'uef/wig', 'uef/nqc']

PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']

PREDICTORS = PRE_RET_PREDICTORS + PREDICTORS_WO_QF

# NUMBER_OF_DOCS = (5, 10, 25, 50, 100, 250, 500, 1000)
SIMILARITY_FUNCTIONS = {'Jac_coefficient': 'jac', 'Top_Docs_overlap': 'sim', 'RBO_EXT_100': 'rbo',
                        'RBO_FUSED_EXT_100': 'rbof'}
# Filter out filled markers and marker settings that do nothing.
MARKERS = ['+', 'x', '.', '*', 'X', 'v']
LINE_STYLES = ['--', '-', ':', ':']
# MARKERS_STYLE = [''.join(i) for i in itertools.product(LINE_STYLES, MARKERS)]
LAMBDA = np.linspace(start=0, stop=1, num=11)
# MARKERS = ['-^', '-v', '-D', '-x', '-h', '-H', 'p-', 's-', '--v', '--1', '--2', '--D', '--x', '--h', '--H', '^-.',
#            '-.v', '1-.', '2-.', '-.D', '-.x', '-.h', '-.H', '3-.', '4-.', 's-.', 'p-.', '+-.', '*-.']

COLORS = ['#2A88AA', '#93BEDB', '#203D78', '#60615C', '#E57270']
# COLORS = ['#1D2735', '#135960', '#2F8F6D', '#8DC05F']
NAMES_DICT = {'rbo': 'Ref-RBO', 'sim': 'Ref-Overlap', 'wig': 'WIG', 'rsd': 'RSD', 'preret/AvgSCQTFIDF': 'AvgSCQ',
              'preret/AvgVarTFIDF': 'AvgVar', 'uef/clarity': 'UEF(Clarity)', 'preret/MaxIDF': 'MaxIDF'}


class GenerateResults:
    def __init__(self, corpus, corr_measure='pearson', load_from_pkl=True):
        self.corpus = corpus
        self.__set_paths(corpus)
        self.corr_measure = corr_measure
        self.results_dirs_dict = self._cp_result_file_to_dirs()
        self.load_from_pkl = load_from_pkl

    @classmethod
    def __set_paths(cls, corpus):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        _corpus_test_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/')

        # AP file for the cross validation process
        cls.query_ap_file = dp.ensure_file(f'{_corpus_test_dir}/ref/QLmap1000-title')
        # CV folds mapping file
        cls.cv_map_file = dp.ensure_file(f'{_corpus_test_dir}/2_folds_30_repetitions.json')
        # The data dir for the Graphs
        cls.data_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/data')
        # The results base dir for the Graphs
        cls.results_dir = dp.ensure_dir(f'~/QppUqvProj/Graphs/{corpus}/referenceLists/title/all_vars/general')
        cls.raw_res_base_dir = dp.ensure_dir(
            f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/title/all_vars/general')

        _ap_file = f'~/QppUqvProj/Results/{corpus}/test/basic/QLmap1000'
        cls.true_ap_file = dp.ensure_file(_ap_file)

    def _cp_result_file_to_dirs(self):
        destination_dirs = defaultdict(str)
        for lam in LAMBDA:
            for sim, pred in itertools.product(SIMILARITY_FUNCTIONS.values(), PREDICTORS):
                dest_dir = dp.ensure_dir(f'{self.results_dir}/{sim}/{pred}/lambda-{lam}/predictions')
                destination_dirs[sim, pred, f'{lam:.2f}'] = dest_dir
                src_dir = dp.ensure_dir(f'{self.raw_res_base_dir}/{sim}/{pred}/predictions')
                prediction_files = glob(f'{src_dir}/predictions-*+lambda+{lam}')
                for _file in prediction_files:
                    copy2(_file, dest_dir)
        return destination_dirs

    def generate_graph_df(self, similarity, predictor):
        _dict = defaultdict(list)

        def append_to_full_results_dict(result, lambda_param):
            _dict['predictor'].append(predictor)
            _dict['sim_func'].append(similarity)
            _dict['result'].append(result)
            _dict['lambda'].append(lambda_param)

        for lam in LAMBDA:
            lambda_param = f'{lam:.2f}'
            result = self._calc_cv_result(similarity, predictor, lambda_param)
            append_to_full_results_dict(result, lambda_param)
        return pd.DataFrame.from_dict(_dict)

    def _calc_cv_result(self, similarity, predictor, lambda_param):
        predictions_dir = self.results_dirs_dict.get((similarity, predictor, lambda_param))
        cv_obj = CrossValidation(k=2, rep=30, file_to_load=self.cv_map_file, predictions_dir=predictions_dir, load=True,
                                 ap_file=self.query_ap_file, test=self.corr_measure)
        mean = cv_obj.calc_test_results()
        return mean

    def generate_results_df(self, cores=4):
        _pkl_file = f'{self.data_dir}/pkl_files/lambda_full_results_df_{self.corpus}_{self.corr_measure}.pkl'
        if self.load_from_pkl:
            try:
                file_to_load = dp.ensure_file(_pkl_file)
                full_results_df = pd.read_pickle(file_to_load)
            except AssertionError:
                print(f'\nFailed to load {_pkl_file}')
                print(f'Will generate {_pkl_file} and save')
                with mp.Pool(processes=cores) as pool:
                    result = pool.starmap(self.generate_graph_df,
                                          itertools.product(SIMILARITY_FUNCTIONS.values(), PREDICTORS))
                pool.close()
                full_results_df = pd.concat(result, axis=0)
                full_results_df.to_pickle(_pkl_file)
        else:
            with mp.Pool(processes=cores) as pool:
                result = pool.starmap(self.generate_graph_df,
                                      itertools.product(SIMILARITY_FUNCTIONS.values(), PREDICTORS))
            pool.close()
            full_results_df = pd.concat(result, axis=0)
            full_results_df.to_pickle(_pkl_file)
        return full_results_df


def plot_graphs(df: pd.DataFrame, corpus):
    # print(df)
    df['result'] = pd.to_numeric(df['result'])
    df['lambda'] = pd.to_numeric(df['lambda'])

    for simi, _df in df.groupby('sim_func'):
        fig = plt.figure(figsize=(16.0, 10.0))  # in inches!
        print(simi)
        print(_df.drop('sim_func', axis=1).set_index('lambda').groupby('predictor')['result'])
        mar = 0
        for predictor, pdf in _df.drop('sim_func', axis=1).set_index('lambda').groupby('predictor'):
            # if predictor in SKIP:
            #     continue
            pdf['result'].plot(legend=True, marker=MARKERS[mar], label=predictor, linewidth=2, markersize=15, mew=5)
            plt.legend()
            mar += 1
        plt.title(f'\\textbf{{{corpus} - {simi}}}')
        plt.xlabel('$\\mathbf{\\lambda}$')
        plt.ylabel("\\textbf{Pearson}")
        # plt.ylabel('Correlation')
        # plt.savefig(f'../../plot_now/{corpus}-{simi}.png')
        plt.show()


def plot_sim_graph(orig_df: pd.DataFrame, simi, corpus):
    corpus_names = {'ClueWeb12B': 'CW12', 'ROBUST': 'ROBUST'}
    df = orig_df.set_index('sim_func')
    df['result'] = pd.to_numeric(df['result'])
    df['lambda'] = pd.to_numeric(df['lambda'])
    df['lambda'] = df['lambda'].values[::-1]
    # fig = plt.figure(figsize=(16.0, 10.0))  # in inches!
    _df = df.loc[simi].set_index('lambda', drop=True)
    _df = _df.loc[_df['predictor'].isin(['uef/clarity', 'wig', 'preret/AvgSCQTFIDF', 'preret/MaxIDF'])]
    mar = 0
    print(_df)
    for predictor, pdf in _df.groupby('predictor'):
        pdf = pdf.rename(NAMES_DICT).rename(NAMES_DICT, axis=1)
        pdf['result'].plot(legend=True, marker=MARKERS[mar],
                           linestyle=LINE_STYLES[mar], label=NAMES_DICT[predictor], linewidth=5, markersize=30, mew=3,
                           color=COLORS[mar])
        plt.legend()
        mar += 1
    plt.title(f'\\textbf{{{corpus_names[corpus]} {NAMES_DICT[simi]}}}')
    plt.xlabel('$\\mathbf{\\lambda}$')
    plt.ylabel("\\textbf{Pearson}")
    plt.show()


def add_sub_plot(df, ax, marker, markersize=None, markerfacecolor=None, color='None', linestyle='None', linewidth=None,
                 mew=None):
    df.set_index('lambda').plot(legend=True, marker=marker, markersize=markersize, linestyle=linestyle, color=color,
                                markerfacecolor=markerfacecolor, grid=False, linewidth=linewidth, mew=mew, ax=ax)
    plt.legend()


def main(args):
    corpus = args.corpus
    generate = args.generate
    load_cache = args.nocache

    # Debugging
    # corpus = 'ROBUST'
    # corpus = 'ClueWeb12B'

    cores = mp.cpu_count() - 1

    res_gen = GenerateResults(corpus, load_from_pkl=load_cache)
    df = res_gen.generate_results_df(cores=cores)
    plot_sim_graph(df, 'rbo', corpus)
    # plot_graphs(df, corpus)
    # print(os.getcwd())


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
