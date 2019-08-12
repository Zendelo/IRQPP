import argparse
import multiprocessing as mp
from functools import partial
import pickle

import numpy as np
import pandas as pd

import dataparser as dp
from pageRank.topic_graph_features import features_loader
from Timer.timer import Timer
from crossval import CrossValidation

parser = argparse.ArgumentParser(description='PageRank UQV Generator',
                                 usage='python3.7 pagerank.py -c CORPUS',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-p', '--predictor', default=None, type=str, help='Choose the predictor to use',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'all'])
parser.add_argument('--uef', help="use the uef version of the predictor", action="store_true")
parser.add_argument('--nocache', help="add this option to avoid loading from cache", action="store_false")

# parser.add_argument('-g', '--group', help='group of queries to predict',
#                     choices=['top', 'low', 'medh', 'medl', 'title'])
# parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default=None,
#                     choices=['all', 'low', 'med', 'top'])
# parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')
# parser.add_argument('--generate', help="generate new features file", action="store_true")
# parser.add_argument('--predict', help="generate new predictions", action="store_true")

LAMBDA = np.linspace(start=0, stop=1, num=11)


class PageRank:
    def __init__(self, corpus, predictor, load=True):
        self.corpus = corpus
        self.__set_paths(corpus, predictor)
        self.similarity_features_df = self.__initialize_features_df()
        self.norm_similarity_features_df = self.__normalize_similarity()
        self.similarity_measures = self.similarity_features_df.columns.tolist()
        self.var_cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.vars_results_dir)
        self.norm_prediction_scores = self.__normalize_predictions()
        self.prediction_scores = self.var_cv.full_set.columns.tolist()
        self.full_raw_weights_df = self.__initialize_full_weights_df()
        self.dict_all_options = self._set_weights()
        if load:
            try:
                # Will try loading a dictionary, if fails will generate and save a new one
                file_to_load = dp.ensure_file(
                    f'~/QppUqvProj/Results/{corpus}/test/pageRank/pkl_files/{predictor}/dict_all_options_stochastic.pkl')
                with open(file_to_load, 'rb') as handle:
                    self.dict_all_options_stochastic = pickle.load(handle)
            except AssertionError:
                self.dict_all_options_stochastic = self._normalize_rows()
                self.__save_new_dictionary(corpus, predictor)
        else:
            self.dict_all_options_stochastic = self._normalize_rows()
            self.__save_new_dictionary(corpus, predictor)
        # print(self.dict_all_options_stochastic['simfunc-RBO_EXT_100+lambda-1.0'].loc['340'])
        # exit()

    @classmethod
    def __set_paths(cls, corpus, predictor):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        cls.predictor = predictor

        _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/'
        cls.vars_results_dir = dp.ensure_dir(f'{_base_dir}/raw/{cls.predictor}/predictions/')

        cls.output_dir = dp.ensure_dir(f'{_base_dir}/referenceLists/pageRank/')

        _test_dir = f'~/QppUqvProj/Results/{corpus}/test'
        cls.folds = dp.ensure_file(f'{_test_dir}/2_folds_30_repetitions.json')

        # cls.ap_file = dp.ensure_file(f'{_test_dir}/pageRank/QLmap1000')

        cls.features = dp.ensure_file(f'{_test_dir}/pageRank/{corpus}_raw_PageRank_Features.pkl')

    def __save_new_dictionary(self, corpus, predictor):
        _dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/pageRank/pkl_files/{predictor}')
        with open(f'{_dir}/dict_all_options_stochastic.pkl', 'wb') as handle:
            pickle.dump(self.dict_all_options_stochastic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __initialize_features_df(self):
        """This method loads the features df from a pickle file"""
        _features_df = features_loader(self.corpus, self.features)
        return _features_df

    def __initialize_full_weights_df(self):
        """This method merges the df of the normalized prediction scores with the normalized similarity features df,
        adding the prediction score to the relevant destination query-node"""
        _var_scores_df = pd.merge(left=self.norm_similarity_features_df,
                                  right=self.norm_prediction_scores.reset_index('topic', drop=True), left_on='dest',
                                  how='left', right_index=True)
        return _var_scores_df

    def _set_weights(self):
        # TODO: Implement a special case for lambda=1
        """This method implements the interpolation of the weight function"""
        _dict = {}
        for lambda_param in LAMBDA:
            for similarity in self.similarity_measures:
                _pred_scores_df = self.full_raw_weights_df[self.prediction_scores] * (1 - lambda_param)
                _sim_df = self.full_raw_weights_df[similarity] * lambda_param
                _dict[f'simfunc-{similarity}+lambda-{lambda_param}'] = _pred_scores_df.add(_sim_df, axis=0)
        return _dict

    def _normalize_rows(self):
        """This method will normalize the weights of each row in the matrix W to be equal to 1 in order to make it
        a legal right stochastic matrix. The rows of matrix W are all the outgoing edges from src node i"""
        _dict = {}
        for params, df in self.dict_all_options.items():
            _list = []
            for src_qid, _df in df.groupby(level='src'):
                _norm_df = _df / _df.sum()
                _list.append(_norm_df)
            _dict[params] = pd.concat(_list)
        return _dict

    def __normalize_predictions(self):
        """This method will normalize the predictions scores """
        df = self.var_cv.full_set.reset_index()
        # Add topic column to the scores df
        for topic, _ in self.similarity_features_df.groupby('topic'):
            df.loc[df['qid'].str.startswith(topic), 'topic'] = topic
        df.set_index(['topic', 'qid'], inplace=True)
        z_n = df.groupby(['topic']).sum()
        norm_df = (df.groupby(['topic', 'qid']).sum() / z_n)
        return norm_df

    def __normalize_similarity(self):
        """This method will normalize the predictions scores """
        df = self.similarity_features_df
        z_n = df.groupby(['topic']).sum()
        norm_df = (df / z_n)
        return norm_df

    def calc_pagerank(self, epsilon=1e-04):
        """The method will calculate the PR scores for the entire set, with all the hyper parameters and write the
        results to files"""
        for hyper_params, full_df in self.dict_all_options_stochastic.items():
            sim_func, lambda_param = (s.split('-')[1] for s in hyper_params.split('+'))
            full_df_pr_list = []
            for pred_score in self.prediction_scores:
                print(f'Working on the combination {sim_func}: {pred_score} lambda={lambda_param}')
                _score_list = []
                for topic, _df in full_df[pred_score].groupby('topic'):
                    # Set all the PR values to be 1/N
                    _initial_pr = 1 / len(_df.index.unique('dest'))
                    # Create a Series that holds the PR scores for each query node
                    pr_sr = pd.Series(data=_initial_pr, index=_df.index.unique('dest'))
                    _pr_dict = {}
                    diff = 1
                    timeout = 100
                    while diff > epsilon and timeout > 0:
                        for dest_node, incoming_weights_df in _df.groupby('dest'):
                            _pr_dict[dest_node] = incoming_weights_df.mul(pr_sr, level='src').sum()
                        _pr_sr = pd.Series(_pr_dict)
                        _diff_sr = pr_sr.subtract(_pr_sr, level='dest').abs()
                        diff = _diff_sr.sum()
                        pr_sr = _pr_sr
                        timeout -= 1
                        # If the PR hasn't converged, it will print a message and continue
                        if timeout == 0:
                            print(
                                f'\n-------> The combination {sim_func}: {pred_score} lambda={lambda_param} Timed Out!')
                            print(
                                f'Topic {topic} has failed to converge, finished with: diff={diff} epsilon={epsilon}\n')
                    _score_list.append(pr_sr)
                res_sr = pd.concat(_score_list)
                res_sr.name = pred_score
                full_df_pr_list.append(res_sr)
                # self._write_results(res_df, sim_func, pred_score.split('_')[1], lambda_param)
            # _full_df_pr = pd.concat(full_df_pr_list, axis=1)

    def _write_results(self, res_df: pd.Series, sim_func, pred_score, lambda_param):
        dir_path = dp.ensure_dir(f'{self.output_dir}/raw/{sim_func}/{self.predictor}/predictions/')
        file_name = f'predictions-{pred_score}+lambda+{lambda_param}'
        res_df.to_csv(path=f'{dir_path}/{file_name}', index=True, sep=' ', float_format='%f')


def main(args):
    corpus = args.corpus
    predictor = args.predictor
    uef = args.uef
    load = args.nocache

    if uef:
        predictor = f'uef/{predictor}'

    # # Debugging
    # print('\n\n\n------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------\n\n\n')
    # predictor = input('What predictor should be used for debugging?\n')
    # corpus = 'ROBUST'

    cores = mp.cpu_count() - 1
    if predictor and corpus:
        pr_obj = PageRank(corpus, predictor, load=load)
        pr_obj.calc_pagerank()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
