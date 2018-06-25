#!/usr/bin/env python
import argparse
from subprocess import run

import pandas as pd

PREDICTORS = ['clarity', 'nqc', 'wig', 'qf']
NUM_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_CUT_OFF = [5, 10, 25, 50, 100]

parser = argparse.ArgumentParser(description='Full Results Pipeline Automation Generator',
                                 usage='Run / Load Results and generate table in LateX',
                                 epilog='Currently Beta Version')

parser.add_argument('--predictor', metavar='predictor_file_path',
                    default='SetupFiles-indri-5.6/clarity.m-1/Clarity-Fiana', help='path to predictor executable res')
parser.add_argument('--parameters', metavar='parameters_file_path', default='clarity/clarityParam.xml',
                    help='path to predictor parameters res')
parser.add_argument('--testing', metavar='running_parameter', default='-documents=', choices=['documents', 'fbDocs'],
                    help='The parameter to optimize')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='data/ROBUST/queries.xml',
                    help='path to queries xml res')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )


class GeneratePredictions:
    def __init__(self, queries, predictions_dir):
        self.queries = queries
        self.predictions_dir = predictions_dir

    def generate_clartiy(self):
        predictor_exe = '~/SetupFiles-indri-5.6/clarity.m-2/Clarity-Anna'
        parameters = '~/clarity/clarityParam.xml'
        running_param = '-fbDocs='
        predictions_dir = self.predictions_dir + 'clarity/predictions/'
        for n in NUM_DOCS:
            print('\n ******** Running for: {} documents ******** \n'.format(n))
            output = predictions_dir + 'predictions-{}'.format(n)
            run('{} {} {}{} {} > {}'.format(predictor_exe, parameters, running_param, n, self.queries, output),
                shell=True)

    def generate_wig(self):
        predictor_exe = 'python3.6 ~/repos/IRQPP/wig.py'
        parameters = '~/baseline/singleUQV/CE.res'
        running_param = '-fbDocs='
        predictions_dir = self.predictions_dir + 'clarity/predictions/'
        for n in NUM_DOCS:
            print('\n ******** Running for: {} documents ******** \n'.format(n))
            output = predictions_dir + 'predictions-{}'.format(n)
            run('{} {} {}{} {} > {}'.format(predictor_exe, parameters, running_param, n, self.queries, output),
                shell=True)
    #   ~/data/ROBUST/singleUQV/queries$k.xml ~/baseline/singleUQV/logqlc$k.res -d $i > ~/predictionsUQV/singleUQV/wig$k/predictions/predictions-$i
    def generate_nqc(self):
        pass

    def generate_qf(self):
        pass

    def generate_uef(self):
        pass


class CrossValidation:
    def __init__(self):
        pass


class GenerateTable:
    def __init__(self):
        pass


def main(args):
    predictor_exe = args.predictor
    parameters_xml = args.parameters
    test_parameter = args.testing
    queries = args.queries
    correlation_measure = args.measure


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
