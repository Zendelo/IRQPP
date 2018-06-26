#!/usr/bin/env python
import argparse
from subprocess import run
import multiprocessing

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
        self.cpu_cores = multiprocessing.cpu_count()

    def __run_predictor(self, predictions_dir, predictor_exe, parameters, running_param, lists=False):
        threads = '-threads={}'.format(self.cpu_cores - 1)
        if lists:
            res = 'list'
            print('\n ******** Generating Lists ******** \n')
        else:
            res = 'predictions'
            print('\n ******** Generating Predictions ******** \n')

        for n in NUM_DOCS:
            print('\n ******** Running for: {} documents ******** \n'.format(n))
            output = predictions_dir + '{}-{}'.format(res, n)
            run('{} {} {} {}{} {} > {}'.format(predictor_exe, parameters, threads, running_param, n, self.queries,
                                               output),
                shell=True)

    def generate_clartiy(self, predictions_dir=None):
        predictor_exe = '~/SetupFiles-indri-5.6/clarity.m-2/Clarity-Anna'
        parameters = '~/QppUqvProj/Results/ROBUST/test/clarityParam.xml'
        running_param = '-fbDocs='
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'clarity/predictions/'
        else:
            predictions_dir = predictions_dir + 'clarity/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_wig(self, predictions_dir=None):
        predictor_exe = 'python3.6 ~/repos/IRQPP/wig.py'
        parameters = '~/QppUqvProj/Results/ROBUST/test'
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'wig/predictions/'
        else:
            predictions_dir = predictions_dir + 'wig/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_nqc(self, predictions_dir=None):
        predictor_exe = 'python3.6 ~/repos/IRQPP/nqc.py'
        parameters = '~/QppUqvProj/Results/ROBUST/test'
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'nqc/predictions/'
        else:
            predictions_dir = predictions_dir + 'nqc/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_qf(self, predictions_dir=None):
        # TODO: modify the function to work with QF
        self._generate_lists_qf()
        predictor_exe = 'python3.6 ~/repos/IRQPP/qf.py'
        parameters = '~/QppUqvProj/Results/ROBUST/test'
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'qf/predictions/'
        else:
            predictions_dir = predictions_dir + 'qf/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def _generate_lists_qf(self):
        predictor_exe = '~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL'
        parameters = '~/QppUqvProj/Results/ROBUST/test/indriRunQF.xml'
        running_param = '-fbDocs='
        predictions_dir = self.predictions_dir + 'qf/lists/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param, lists=True)

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
