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

parser.add_argument('--predictor', metavar='predictor_name', default='clarity', help='predictor to run',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'uef'])
parser.add_argument('-r', '--predictions_dir', metavar='parameters_file_path', default='clarity/clarityParam.xml',
                    help='path to predictor parameters res')
parser.add_argument('--hyper', metavar='hyper_parameter', default='-documents=', choices=['documents', 'fbDocs'],
                    help='The parameter to optimize')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='data/ROBUST/queries.xml',
                    help='path to queries xml res')
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'])
parser.add_argument('-c', '--corpus', default='ROBUST', type=str,
                    help='corpus (index) to work with', choices=['ROBUST', 'ClueWebB12'])


class GeneratePredictions:
    def __init__(self, queries, predictions_dir, corpus='ROBUST'):
        """
        :param queries: queries XML file
        :param predictions_dir: default predictions results dir
        """
        self.queries = queries
        self.predictions_dir = predictions_dir
        self.corpus = corpus
        self.cpu_cores = max(multiprocessing.cpu_count() * 0.5, min(multiprocessing.cpu_count() - 1, 16))

    def __run_predictor(self, predictions_dir, predictor_exe, parameters, running_param, lists=False):
        threads = '-threads={}'.format(self.cpu_cores)
        if lists:
            res = 'list'
            print('\n ******** Generating Lists ******** \n')
        else:
            res = 'predictions'
            print('\n ******** Generating Predictions ******** \n')

        if 'indri' in predictor_exe.lower():
            for n in NUM_DOCS:
                print('\n ******** Running for: {} documents ******** \n'.format(n))
                output = predictions_dir + '{}-{}'.format(res, n)
                run('{} {} {} {}{} {} > {}'.format(predictor_exe, parameters, threads, running_param, n, self.queries,
                                                   output), shell=True)

        if 'qf' in predictor_exe.lower():
            lists_dir = predictions_dir.replace('predictions', 'lists')
            for n in NUM_DOCS:
                for k in LIST_CUT_OFF:
                    print('\n ******** Running for: {} documents + {} list cutoff ******** \n'.format(n, k))
                    output = predictions_dir + '{}-{}+{}'.format(res, n, k)
                    inlist = lists_dir + 'list-{}'.format(n)
                    run('{} {} {} {}{} > {}'.format(predictor_exe, parameters, inlist, running_param, k, output),
                        shell=True)
        else:
            # Assuming the predictor is NQC or WIG
            for n in NUM_DOCS:
                print('\n ******** Running for: {} documents ******** \n'.format(n))
                output = predictions_dir + '{}-{}'.format(res, n)
                run('{} {} {} {}{} > {}'.format(predictor_exe, parameters, self.queries, running_param, n, output),
                    shell=True)

    def generate_clartiy(self, predictions_dir=None):
        predictor_exe = '~/SetupFiles-indri-5.6/clarity.m-2/Clarity-Anna'
        parameters = '~/QppUqvProj/Results/{}/test/clarityParam.xml'.format(self.corpus)
        running_param = '-fbDocs='
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'clarity/predictions/'
        else:
            predictions_dir = predictions_dir + 'clarity/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_wig(self, predictions_dir=None):
        predictor_exe = 'python3.6 ~/repos/IRQPP/wig.py'
        ce_scores = '~/QppUqvProj/Results/{}/test/basic/CE.res'.format(self.corpus)
        qlc_scores = '~/QppUqvProj/Results/{}/test/basic/logqlc.res'.format(self.corpus)
        parameters = '{} {}'.format(ce_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'wig/predictions/'
        else:
            predictions_dir = predictions_dir + 'wig/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_nqc(self, predictions_dir=None):
        predictor_exe = 'python3.6 ~/repos/IRQPP/nqc.py'
        ce_scores = '~/QppUqvProj/Results/{}/test/basic/CE.res'.format(self.corpus)
        qlc_scores = '~/QppUqvProj/Results/{}/test/basic/logqlc.res'.format(self.corpus)
        parameters = '{} {}'.format(ce_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'nqc/predictions/'
        else:
            predictions_dir = predictions_dir + 'nqc/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_qf(self, predictions_dir=None):
        self._generate_lists_qf()
        predictor_exe = 'python3.6 ~/repos/IRQPP/qf.py'
        parameters = '~/QppUqvProj/Results/{}/test/basic/QL.res'.format(self.corpus)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'qf/predictions/'
        else:
            predictions_dir = predictions_dir + 'qf/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def _generate_lists_qf(self):
        predictor_exe = '~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL'
        parameters = '~/QppUqvProj/Results/{}/test/indriRunQF.xml'.format(self.corpus)
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
    predictions_dir = args.predictions_dir
    test_parameter = args.hyper
    queries = args.queries
    correlation_measure = args.measure
    corpus = args.corpus
    predict = GeneratePredictions(queries, predictions_dir, corpus)

    predict.generate_clartiy()
    predict.generate_nqc()
    predict.generate_wig()
    predict.generate_qf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
