#!/usr/bin/env python

import argparse
from subprocess import run
import multiprocessing
import os

# TODO: Add directories checks and creation
# os.path.exists('file or dir')
# os.path.isfile('file')
# os.path.isdir('dir')

# TODO: Create for UQV aggregations
# TODO: Create for UQV singles
# TODO: Create CV process and write the results to tables

PREDICTORS = ['clarity', 'nqc', 'wig', 'qf']
NUM_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_CUT_OFF = [5, 10, 25, 50, 100]

parser = argparse.ArgumentParser(description='Full Results Pipeline Automation Generator',
                                 usage='Run / Load Results and generate table in LateX',
                                 epilog='Currently Beta Version')

parser.add_argument('--predictor', metavar='predictor_name', help='predictor to run',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'uef', 'all'])
parser.add_argument('-r', '--predictions_dir', metavar='results_dir_path',
                    default='~/QppUqvProj/Results/ROBUST/basicPredictions/', help='path where to save results')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='~/data/ROBUST/queries.xml',
                    help='path to queries xml res')
parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--qtype', default='basic', type=str, help='The type of queries to run',
                    choices=['basic', 'single', 'aggregated'])
parser.add_argument('--generate', help="generate new predictions", action="store_true")


class GeneratePredictions:
    def __init__(self, queries, predictions_dir, corpus='ROBUST', qtype='basic'):
        """
        :param queries: queries XML file
        :param predictions_dir: default predictions results dir
        """
        self.queries = queries
        self.predictions_dir = predictions_dir
        self.corpus = corpus
        self.qtype = qtype
        self.cpu_cores = max(multiprocessing.cpu_count() * 0.5, min(multiprocessing.cpu_count() - 1, 16))

    @staticmethod
    def __run_indri_app(predictor_exe, parameters, threads, running_param, n, queries, output):
        ensure_dir(output)
        run('{} {} {} {}{} {} > {}'.format(predictor_exe, parameters, threads, running_param, n, queries, output),
            shell=True)

    @staticmethod
    def __run_py_predictor(predictor_exe, parameters, temporal_var, running_param, n, output):
        ensure_dir(output)
        run('{} {} {} {}{} > {}'.format(predictor_exe, parameters, temporal_var, running_param, n, output), shell=True)

    def __run_predictor(self, predictions_dir, predictor_exe, parameters, running_param, lists=False, queries=None):
        threads = '-threads={}'.format(self.cpu_cores)
        if queries is None:
            queries = self.queries
        if lists:
            res = 'list'
            print('\n ******** Generating Lists ******** \n')
        else:
            res = 'predictions'
            print('\n ******** Generating Predictions ******** \n')

        if 'indri' in predictor_exe.lower():
            # Running indri APP
            _queries = queries
            for n in NUM_DOCS:
                print('\n ******** Running for: {} documents ******** \n'.format(n))
                output = predictions_dir + '{}-{}'.format(res, n)
                if 'uef' in queries.lower():
                    # Assuming it's uef lists creation
                    queries = '{}-{}.xml'.format(_queries, n)
                ensure_file([predictor_exe, parameters, queries])
                self.__run_indri_app(predictor_exe, parameters, threads, running_param, n, queries, output)

        elif predictor_exe.endswith('qf.py'):
            lists_dir = predictions_dir.replace('predictions', 'lists')
            for n in NUM_DOCS:
                for k in LIST_CUT_OFF:
                    print('\n ******** Running for: {} documents + {} list cutoff ******** \n'.format(n, k))
                    output = predictions_dir + '{}-{}+{}'.format(res, n, k)
                    inlist = lists_dir + 'list-{}'.format(n)
                    ensure_file([predictor_exe.split()[1]] + [parameters, inlist])
                    self.__run_py_predictor(predictor_exe, parameters, inlist, running_param, k, output)

        elif predictor_exe.endswith('addWorkingsetdocs.py'):
            print('\n ******** Generating UEF query files ******** \n')
            for n in NUM_DOCS:
                output = predictions_dir + 'queriesUEF-{}.xml'.format(n)
                ensure_file([predictor_exe.split()[1]] + [parameters, queries])
                self.__run_py_predictor(predictor_exe, parameters, queries, running_param, n, output)

        elif predictor_exe.endswith(('nqc.py', 'wig.py')):
            for n in NUM_DOCS:
                print('\n ******** Running for: {} documents ******** \n'.format(n))
                output = predictions_dir + '{}-{}'.format(res, n)
                ensure_file([predictor_exe.split()[1]] + [parameters, queries])
                self.__run_py_predictor(predictor_exe, parameters, queries, running_param, n, output)

        elif predictor_exe.endswith('uef.py'):
            lists_dir = predictions_dir + 'lists/'
            for pred in PREDICTORS:
                for n in NUM_DOCS:
                    if pred != 'qf':
                        print('\n ******** Running for: {} documents ******** \n'.format(n))
                        output = predictions_dir + '{}/{}-{}'.format(pred, res, n)
                        inlist = lists_dir + 'list-{}'.format(n)
                        predictions = predictions_dir.replace('uef', pred) + 'predictions/{}-{}'.format(res, n)
                        params = '{} {}'.format(inlist, predictions)
                        ensure_file([predictor_exe.split()[1]] + [parameters, inlist, predictions])
                        self.__run_py_predictor(predictor_exe, parameters, params, running_param, n, output)
                    else:
                        for k in LIST_CUT_OFF:
                            print('\n ******** Running for: {} documents + {} list cutoff ******** \n'.format(n, k))
                            output = predictions_dir + '{}/{}-{}+{}'.format(pred, res, n, k)
                            inlist = lists_dir + 'list-{}'.format(n)
                            predictions = predictions_dir.replace('uef', pred)
                            predictions += 'predictions/{}-{}+{}'.format(res, n, k)
                            params = '{} {}'.format(inlist, predictions)
                            ensure_file([predictor_exe.split()[1]] + [parameters, inlist, predictions])
                            self.__run_py_predictor(predictor_exe, parameters, params, running_param, n, output)

    def generate_clartiy(self, predictions_dir=None):
        print('\n -- Clarity -- \n')
        predictor_exe = '~/SetupFiles-indri-5.6/clarity.m-2/Clarity-Anna'
        parameters = '~/QppUqvProj/Results/{}/test/clarityParam.xml'.format(self.corpus)
        running_param = '-fbDocs='
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'clarity/predictions/'
        else:
            predictions_dir = predictions_dir + 'clarity/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_wig(self, predictions_dir=None):
        print('\n -- WIG -- \n')
        predictor_exe = 'python3.6 ~/repos/IRQPP/wig.py'
        ce_scores = '~/QppUqvProj/Results/{}/test/{}/CE.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ce_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'wig/predictions/'
        else:
            predictions_dir = predictions_dir + 'wig/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_nqc(self, predictions_dir=None):
        print('\n -- NQC -- \n')
        predictor_exe = 'python3.6 ~/repos/IRQPP/nqc.py'
        ce_scores = '~/QppUqvProj/Results/{}/test/{}/CE.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ce_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'nqc/predictions/'
        else:
            predictions_dir = predictions_dir + 'nqc/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_qf(self, predictions_dir=None):
        print('\n -- QF -- \n')
        self._generate_lists_qf()
        predictor_exe = 'python3.6 ~/repos/IRQPP/qf.py'
        parameters = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
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
        """Assuming all the previous predictions exist, will generate the uef lists and predictions"""
        self._generate_lists_uef()
        predictor_exe = 'python3.6 ~/repos/IRQPP/uef/uef.py'
        parameters = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
        running_param = '-d '
        predictions_dir = self.predictions_dir + 'uef/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def _generate_lists_uef(self):
        predictor_exe = 'python3.6 ~/repos/IRQPP/addWorkingsetdocs.py'
        parameters = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
        running_param = '-d '
        predictions_dir = self.predictions_dir + 'uef/data/'
        queries = predictions_dir + 'queriesUEF'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)
        predictor_exe = '~/SetupFiles-indri-5.6/runqueryql/IndriRunQueryQL'
        parameters = '~/QppUqvProj/Results/{}/test/indriRunQF.xml'.format(self.corpus)
        running_param = '-fbDocs='
        predictions_dir = self.predictions_dir + 'uef/lists/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param, lists=True, queries=queries)


class CrossValidation:
    # TODO: Implement CV
    def __init__(self):
        pass

    def __run_predictor(self, predictions_dir, predictor_exe, parameters, running_param, lists=False):
        pass


class GenerateTable:
    def __init__(self):
        pass


def ensure_dir(file_path):
    # tilde expansion
    file_path = os.path.expanduser(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file(files):
    for file in files:
        # tilde expansion
        file_path = os.path.expanduser(file)
        assert os.path.isfile(file_path), "The file {} doesn't exist. Please create the file first".format(file)


def main(args):
    predictions_dir = args.predictions_dir
    queries = args.queries
    corpus = args.corpus
    queries_type = args.qtype
    generate = args.generate
    predictor = args.predictor

    predict = GeneratePredictions(queries, predictions_dir, corpus, queries_type)

    if predictor.lower() == 'clarity':
        if generate:
            predict.generate_clartiy()
    if predictor.lower() == 'nqc':
        if generate:
            predict.generate_nqc()
    if predictor.lower() == 'wig':
        if generate:
            predict.generate_wig()
    if predictor.lower() == 'qf':
        if generate:
            predict.generate_qf()
    if predictor.lower() == 'uef':
        if generate:
            predict.generate_uef()

    if predictor.lower() == 'all':
        if generate:
            predict.generate_clartiy()
            predict.generate_nqc()
            predict.generate_wig()
            predict.generate_qf()
            predict.generate_uef()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
