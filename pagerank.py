import argparse
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations_with_replacement
from functools import partial

import numpy as np
import pandas as pd

import dataparser as dp
from RBO import rbo_dict
from topic_graph_features import features_loader
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Features for PageRank UQV query variations Generator',
                                 usage='python3.7 features.py -q queries.txt -c CORPUS',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])

# parser.add_argument('-g', '--group', help='group of queries to predict',
#                     choices=['top', 'low', 'medh', 'medl', 'title'])
# parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default=None,
#                     choices=['all', 'low', 'med', 'top'])
parser.add_argument('-l', '--load', default=None, type=str, help='features file to load')
parser.add_argument('--generate', help="generate new features file", action="store_true")
parser.add_argument('--predict', help="generate new predictions", action="store_true")


def main(args):
    corpus = args.corpus
    generate = args.generate
    # predict = args.predict
    # queries_group = args.group
    file_to_load = args.load
    # quantile = args.quantile

    # # Debugging
    # print('------------!!!!!!!---------- Debugging Mode ------------!!!!!!!----------')
    # testing_feat = QueryFeatureFactory('ROBUST')
    # norm_features_df = testing_feat.generate_features()
    # # norm_features_df.reset_index().to_json('query_features_{}_uqv.JSON'.format(corpus))

    cores = mp.cpu_count() - 1

    if generate:
        testing_feat = QueryFeatureFactory(corpus)
        norm_features_df = testing_feat.generate_features()

        _path = f'~/QppUqvProj/Results/{corpus}/test/pageRank'
        _path = dp.ensure_dir(_path)
        norm_features_df.reset_index().to_json(f'{_path}/PageRank_Features.JSON')

    elif file_to_load:
        features_df = features_loader(file_to_load, corpus)
        print(features_df)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
