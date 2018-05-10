#!/usr/bin/env python
import argparse
from subprocess import run

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

PARAMS = [5, 10, 25, 50, 100, 250, 500, 1000]
# PARAMS = [i*i*5 for i in range(1,15)]

parser = argparse.ArgumentParser(description='Correlation Evaluation script',
                                 usage='Use CV to optimize correlation',
                                 epilog='The files must have 2 columns, first for index and second for the values')

parser.add_argument('--predictor', metavar='predictor_file_path',
                    default='SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana', help='path to predictor executable file')
parser.add_argument('--parameters', metavar='parameters_file_path', default='clarity/clarityParam.xml',
                    help='path to predictor parameters file')
parser.add_argument('--testing', metavar='running_parameter', default='-documents=', choices=['-documents'],
                    help='The parameter to optimize')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='data/ROBUST/queries.xml',
                    help='path to queries xml file')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")


def pre_testing(predictor_exe, parameters_xml, test_param, queries):
    """This function will run the predictor using a shell command for different numbers of documents
     and save the output files to the dir tmp-testing"""

    run('mkdir -v tmp-testing', shell=True)
    pred = 'Fiana' if 'Fiana' in predictor_exe else 'Anna'
    run('mkdir -v tmp-testing/clarity-{}'.format(pred), shell=True)
    print('The temporary files will be saved in the directory tmp-testing')
    for i in PARAMS:
        print('\n ******** Running for: {} documents ******** \n'.format(i))
        output = 'tmp-testing/clarity-{}/predictions-{}'.format(pred, i)
        run('{} {} {}{} {} > {}'.format(predictor_exe, parameters_xml, test_param, i,
                                        queries, output), shell=True)


def calc_cor_files(first_file, second_file, test):
    first_df = pd.read_table(first_file, delim_whitespace=True, header=None, index_col=0, names=['x'])
    second_df = pd.read_table(second_file, delim_whitespace=True, header=None, index_col=0, names=['y'])
    return calc_cor_df(first_df, second_df, test)


def calc_cor_df(first_df, second_df, test):
    merged_df = pd.merge(first_df, second_df, left_index=True, right_index=True)
    return merged_df['x'].corr(merged_df['y'], method=test)


def main(args):
    predictor_exe = args.predictor
    parameters_xml = args.parameters
    test_parameter = args.testing
    queries = args.queries
    correlation_measure = args.measure

    pre_testing(predictor_exe, parameters_xml, test_parameter, queries)

    # print("\n Removing files \n")
    # run('rm -rfv tmp-testing', shell=True)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
