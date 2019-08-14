import argparse
from glob import glob

import pandas as pd

import dataparser as dp
from Timer.timer import Timer
import networkx as nx
import pprint
from crossval import CrossValidation
from queries_pre_process import add_topic_to_qdf

PREDICTORS = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'qf', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv', 'uef/qf']
SIMILARITY_MEASURES = ['Jac_coefficient', 'RBO_EXT_100', 'Top_10_Docs_overlap', 'RBO_FUSED_EXT_100']


def calc_stats(full_df: pd.DataFrame, ap_df: pd.DataFrame):
    add_topic_to_qdf(full_df)
    for col in full_df.columns:
        pr_sr = full_df.loc[:, col]
        pr_sr.groupby()


def main(corpus, similarity, predictor):
    cv_folds = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/2_folds_30_repetitions.json')
    ap_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QLmap1000')
    predictions_dir = dp.ensure_dir(
        f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/pageRank/raw/{similarity}/{predictor}/predictions/')
    ap_obj = dp.ResultsReader(ap_file, 'ap')
    ap_df = ap_obj.data_df
    cv_obj = CrossValidation(predictions_dir=predictions_dir, file_to_load=cv_folds)
    full_results_df = cv_obj.full_set
    calc_stats(full_results_df, ap_df)


if __name__ == '__main__':
    corpus = 'ROBUST'
    similarity = 'Jac_coefficient'
    predictor = 'wig'
    main(corpus, similarity, predictor)
