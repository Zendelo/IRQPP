"""This code implements the GEO mean predictor from the paper:
Estimating Query Representativeness for Query-Performance Prediction
by Sondak et al."""

import argparse

import pandas as pd

from qpputils import dataparser as dp
from Timer import Timer

parser = argparse.ArgumentParser(description='RSD(wig) predictor',
                                 usage='Change the paths in the code in order to predict UQV/Base queries',
                                 epilog='Generates the RSD predictor scores')

parser.add_argument('-c', '--corpus', default=None, help='The corpus to be used', choices=['ROBUST', 'ClueWeb12B'])


def geo_mean(qdb: dp.QueriesTextParser, probabilities_df: pd.DataFrame):
    qdf = qdb.queries_df.set_index('qid')
    qdf['qlen'] = qdf['text'].str.split().apply(len)
    prob_qlen_df = probabilities_df.groupby('qid').count()
    prob_prod_df = probabilities_df.groupby('qid').prod()
    zeros_df = prob_qlen_df.subtract(qdf['qlen'], axis=0).applymap(lambda x: 0 if x < 0 else 1)
    df = prob_prod_df.mul(zeros_df)
    df = pd.concat([df, qdf['qlen']], axis=1, sort=True)
    df = df.apply(lambda x: x ** (1 / x.qlen), axis=1).drop('qlen', axis=1)
    return df


def write_predictions(df, corpus, uqv):
    if uqv:
        _dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/geo/predictions')
    else:
        _dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/geo/predictions')
    for col in df:
        file_name = f'{_dir}/predictions-{col}'
        df[col].to_csv(file_name, sep=" ", header=False, index=True, float_format='%f')


def main(args):
    corpus = args.corpus

    # corpus = 'ROBUST'

    if not corpus:
        return

    queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.txt')
    rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/RMprob')

    # queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries.txt')
    # rm_probabilities_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/RMprob')

    queries_obj = dp.QueriesTextParser(queries_file)
    rm_probabilities_df = dp.read_rm_prob_files(rm_probabilities_dir, number_of_docs=20000, clipping='*')

    uqv = True if 'uqv' in queries_file.split('/')[-1].lower() else False

    results_df = geo_mean(queries_obj, rm_probabilities_df)
    write_predictions(results_df, corpus, uqv)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
