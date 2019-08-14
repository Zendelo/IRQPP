import argparse
from glob import glob

import pandas as pd

import dataparser as dp
from Timer.timer import Timer
import networkx as nx
import pprint
from crossval import CrossValidation

PREDICTORS = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'qf', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv', 'uef/qf']
SIMILARITY_MEASURES = ['Jac_coefficient', 'RBO_EXT_100', 'Top_10_Docs_overlap', 'RBO_FUSED_EXT_100']


def main(corpus, similarity, predictor):
    cv_folds = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/2_folds_30_repetitions.json')
    ap_file = dp.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/raw/QLmap1000')
    predictions_dir = dp.ensure_dir(
        f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/pageRank/raw/{similarity}/{predictor}/predictions/')
    ap_obj = dp.ResultsReader(ap_file, 'ap')
    ap_df = ap_obj.data_df
    cv_obj = CrossValidation(predictions_dir=predictions_dir, ap_file=ap_file, file_to_load=cv_folds)


if __name__ == '__main__':
    corpus = 'ROBUST'
    similarity = 'Jac_coefficient'
    predictor = 'wig'
    main(corpus, similarity, predictor)
