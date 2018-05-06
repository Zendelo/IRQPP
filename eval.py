#! /usr/bin/env python
from __future__ import print_function
import subprocess
from subprocess import run
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Correlation Evaluation script',
                                 usage='specify 2 files to evaluate correlation',
                                 epilog='The files must have 2 columns, first for index and second for the values')

parser.add_argument('--predictor', metavar='predictor_file_path',
                    default='SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana', help='path to predictor executable file')
parser.add_argument('--parameters', metavar='parameters_file_path', default='clarity/clarityParam.xml',
                    help='path to predictor parameters file')
parser.add_argument('--testing', metavar='running_parameter', default='-documents=', choices=['-documents'],
                    help='path to predictor parameters file')
parser.add_argument('--queries', default='data/ROBUST/queries.xml', help='path to queries xml file')
parser.add_argument('--labeled', default='baseline/QLmap1000', help='path to labeled list file')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )


def testing(predictor_exe, parameters_xml, test_param, queries, labeled, correlation_measure):
    run('mkdir -v tmp-testing', shell=True)
    print('The temporary files will be saved in the directory tmp-testing')

    for i in [5, 10, 25, 50, 100, 250, 500, 1000]:
        print('\n ******** Running for: {} documents ******** \n'.format(i))

        output = 'tmp-testing/predictions-{}'.format(i)
        run('{} {} {}{} {} > {}'.format(predictor_exe, parameters_xml, test_param, i,
                                        queries, output), shell=True)
        print('the {} correlation is: {}'.format(correlation_measure, calc_cor(output, labeled, correlation_measure)))

    print("\n Removing files \n")

    run('rm -rfv tmp-testing', shell=True)


def calc_cor(first_file, second_file, test):
    first_df = pd.read_table(first_file, delim_whitespace=True, header=None, index_col=0, names=['x'])
    second_df = pd.read_table(second_file, delim_whitespace=True, header=None, index_col=0, names=['y'])
    merged_df = pd.merge(first_df, second_df, left_index=True, right_index=True)
    return merged_df['x'].corr(merged_df['y'], method=test)


def main(args):
    predictor_exe = args.predictor
    parameters_xml = args.parameters
    test_parameter = args.testing
    labeled_file = args.labeled
    queries = args.queries
    correlation_measure = args.measure
    testing(predictor_exe, parameters_xml, test_parameter, queries, labeled_file, correlation_measure)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
