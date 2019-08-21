import argparse
from glob import glob

import pandas as pd
import numpy as np

import dataparser as dp
from Timer.timer import Timer
import networkx as nx
import pprint
from crossval import CrossValidation
from queries_pre_process import add_topic_to_qdf

PREDICTORS = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'qf', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv', 'uef/qf']
SIMILARITY_MEASURES = ['Jac_coefficient', 'RBO_EXT_100', 'Top_10_Docs_overlap', 'RBO_FUSED_EXT_100']


def calc_stats(full_df: pd.DataFrame, ap_df: pd.DataFrame):
    max_ap_vars = ap_df.groupby('topic').max()
    min_ap_vars = ap_df.groupby('topic').min()
    best_results = []
    worst_results = []
    for col in full_df.set_index(['topic', 'qid']).columns:
        pr_df = full_df.loc[:, ['topic', 'qid', col]]
        best_result = {}
        worst_result = {}
        for topic, _df in pr_df.groupby('topic'):
            _max_var_ap = max_ap_vars.loc[topic].qid
            _min_var_ap = min_ap_vars.loc[topic].qid
            pr_of_max = _df.loc[_df['qid'] == _max_var_ap, col].values[0]
            pr_of_min = _df.loc[_df['qid'] == _min_var_ap, col].values[0]
            best_var_score = np.count_nonzero(_df[col] < pr_of_max) / len(_df)
            worst_var_score = np.count_nonzero(_df[col] > pr_of_min) / len(_df)
            best_result[topic] = {col: best_var_score}
            worst_result[topic] = {col: worst_var_score}
        best_results.append(pd.DataFrame.from_dict(best_result, orient='index'))
        worst_results.append(pd.DataFrame.from_dict(worst_result, orient='index'))
    best_df = pd.concat(best_results, axis=1)
    worst_df = pd.concat(worst_results, axis=1)
    return best_df, worst_df


def calc_scores(corpus, similarity, predictor):
    cv_folds = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/2_folds_30_repetitions.json')
    ap_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QLmap1000')
    pkl_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/test/pageRank/pkl_files/{predictor}')
    predictions_dir = dp.ensure_dir(
        f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/pageRank/raw/{similarity}/{predictor}/predictions')
    # results_dir = dp.ensure_dir(
    #     f'QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/pageRank/scores/{similarity}/{predictor}/predictions')
    ap_obj = dp.ResultsReader(ap_file, 'ap')
    ap_df = add_topic_to_qdf(ap_obj.data_df)
    cv_obj = CrossValidation(predictions_dir=predictions_dir, file_to_load=cv_folds)
    full_results_df = add_topic_to_qdf(cv_obj.full_set)
    best_file = f'{pkl_dir}/{similarity}_best_results.pkl'
    worst_file = f'{pkl_dir}/{similarity}_worst_results.pkl'
    try:
        best_df = pd.read_pickle(dp.ensure_file(best_file))
        worst_df = pd.read_pickle(dp.ensure_file(worst_file))
    except AssertionError:
        best_df, worst_df = calc_stats(full_results_df, ap_df)
        best_df.to_pickle(best_file)
        worst_df.to_pickle(worst_file)
    cv_obj.full_set = best_df
    best_score = cv_obj.calc_test_results()
    cv_obj.__delattr__('corrs_df')
    cv_obj.full_set = worst_df
    worst_score = cv_obj.calc_test_results()
    return best_score, worst_score


if __name__ == '__main__':
    # Debugging
    print('\n\n\n------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------\n\n\n')
    predictor = input('What predictor should be used for debugging?\n')
    corpus = 'ROBUST'
    similarity = 'Jac_coefficient'
    timer = Timer('Total time')
    calc_scores(corpus, similarity, predictor)
    timer.stop()
