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
parser.add_argument('-t', '--test', default='pearson', type=str,
                    help='default test type is pearson', choices=['pearson', 'spearman', 'kendall'], )


def testing(predictor_exe, parameters_xml, test_param, queries):
    run('mkdir -v tmp-testing', shell=True)
    print('The temporary files will be saved in the directory tmp-testing')

    for i in [5, 10, 25, 50, 100, 250, 500, 1000]:
        print('\n ******** Running for: {} documents ******** \n'.format(i))
        #print('{0} {1} {2}{3} {4}> tmp-testing/testing-{3}'.format(predictor_exe, parameters_xml, test_param, i, queries))
        #exit()

        run('{0} {1} {2}{3} {4} > tmp-testing/testing-{3}'.format(predictor_exe, parameters_xml, test_param, i, queries), shell=True)
        run('python correlation.py baseline/QLmap1000 tmp-testing/testing-{}'.format(i), shell=True)

    print("\n Removing files \n")

    run('rm -rfv tmp-testing', shell=True, check=True)

    # SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana clarity/clarityParam.xml -documents=2 data/ROBUST/queries.xml


def print_cor(df, type):
    print('The {} correlation is: {:0.4f}'.format(type, df['x'].corr(df['y'], method=type)))


def main(args):
    predictor_exe = args.predictor
    parameters_xml = args.parameters
    test_parameter = args.testing
    queries = args.queries
    testing(predictor_exe, parameters_xml, test_parameter, queries)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
