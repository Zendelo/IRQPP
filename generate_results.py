#!/usr/bin/env python

import argparse
import glob
import multiprocessing
import os
from subprocess import run
from collections import defaultdict
import pandas as pd

from Timer.timer import Timer
from crossval import CrossValidation

# TODO: Add directories checks and creation
# os.path.exists('file or dir')
# os.path.isfile('file')
# os.path.isdir('dir')
# TODO: Create for UQV aggregations
# TODO: Create for UQV singles

# TODO: Create CV process and write the results to tables
# TODO: Add a check that all necessary files exist on startup (to avoid later crash)

PREDICTORS = ['clarity', 'nqc', 'wig', 'qf']
NUM_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_CUT_OFF = [5, 10, 25, 50, 100]
AGGREGATE_FUNCTIONS = ['avg', 'max', 'med', 'min', 'std']
SINGLE_FUNCTIONS = ['max', 'min', 'medl', 'medh']
SPLITS = 2
REPEATS = 30

parser = argparse.ArgumentParser(description='Full Results Pipeline Automation Generator',
                                 usage='python3.6 generate_results.py --predictor PREDICTOR -c CORPUS -q QUERIES ',
                                 epilog='Currently Beta Version')

parser.add_argument('--predictor', metavar='predictor_name', help='predictor to run',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'uef', 'all'])
# parser.add_argument('-r', '--predictions_dir', metavar='results_dir_path',
#                     default='~/QppUqvProj/Results/ROBUST/basicPredictions/', help='path where to save results')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='~/data/ROBUST/queries.xml',
                    help='path to queries xml res')
parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--qtype', default='basic', type=str, help='The type of queries to run',
                    choices=['basic', 'single', 'aggregated'])
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )
parser.add_argument('--generate', help="generate new predictions", action="store_true")
parser.add_argument('--lists', help="generate new lists", action="store_true")


class GeneratePredictions:
    def __init__(self, queries, predictions_dir, corpus, qtype, lists):
        """
        :param queries: queries XML file
        :param predictions_dir: default predictions results dir
        """
        self.queries = queries
        self.predictions_dir = os.path.normpath(os.path.expanduser(predictions_dir)) + '/'
        self.corpus = corpus
        self.qtype = qtype if qtype == 'basic' else 'raw'
        self.gen_lists = lists
        self.cpu_cores = max(multiprocessing.cpu_count() * 0.5, min(multiprocessing.cpu_count(), 16))

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
                ensure_files([predictor_exe, parameters, queries])
                self.__run_indri_app(predictor_exe, parameters, threads, running_param, n, queries, output)

        elif predictor_exe.endswith('qf.py'):
            lists_dir = predictions_dir.replace('predictions', 'lists')
            for n in NUM_DOCS:
                for k in LIST_CUT_OFF:
                    print('\n ******** Running for: {} documents + {} list cutoff ******** \n'.format(n, k))
                    output = predictions_dir + '{}-{}+{}'.format(res, n, k)
                    inlist = lists_dir + 'list-{}'.format(n)
                    ensure_files([predictor_exe.split()[1]] + [parameters, inlist])
                    self.__run_py_predictor(predictor_exe, parameters, inlist, running_param, k, output)

        elif predictor_exe.endswith('addWorkingsetdocs.py'):
            print('\n ******** Generating UEF query files ******** \n')
            for n in NUM_DOCS:
                output = predictions_dir + 'queriesUEF-{}.xml'.format(n)
                ensure_files([predictor_exe.split()[1]] + [parameters, queries])
                self.__run_py_predictor(predictor_exe, parameters, queries, running_param, n, output)

        elif predictor_exe.endswith(('nqc.py', 'wig.py')):
            for n in NUM_DOCS:
                print('\n ******** Running for: {} documents ******** \n'.format(n))
                output = predictions_dir + '{}-{}'.format(res, n)
                ensure_files([predictor_exe.split()[1]] + [parameters, queries])
                self.__run_py_predictor(predictor_exe, parameters, queries, running_param, n, output)

        elif predictor_exe.endswith('uef.py'):
            lists_dir = predictions_dir + 'lists/'
            for pred in PREDICTORS:
                for n in NUM_DOCS:
                    if pred != 'qf':
                        print('\n ******** Running for: {} documents ******** \n'.format(n))
                        output = predictions_dir + '{}/predictions/{}-{}'.format(pred, res, n)
                        inlist = lists_dir + 'list-{}'.format(n)
                        predictions = predictions_dir.replace('uef', pred) + 'predictions/{}-{}'.format(res, n)
                        params = '{} {}'.format(inlist, predictions)
                        ensure_files([predictor_exe.split()[1]] + [parameters, inlist, predictions])
                        self.__run_py_predictor(predictor_exe, parameters, params, running_param, n, output)
                    else:
                        for k in LIST_CUT_OFF:
                            print('\n ******** Running for: {} documents + {} list cutoff ******** \n'.format(n, k))
                            output = predictions_dir + '{}/predictions/{}-{}+{}'.format(pred, res, n, k)
                            inlist = lists_dir + 'list-{}'.format(n)
                            predictions = predictions_dir.replace('uef', pred)
                            predictions += 'predictions/{}-{}+{}'.format(res, n, k)
                            params = '{} {}'.format(inlist, predictions)
                            ensure_files([predictor_exe.split()[1]] + [parameters, inlist, predictions])
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
        ql_scores = '~/QppUqvProj/Results/{}/test/{}/CE.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ql_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'wig/predictions/'
        else:
            predictions_dir = predictions_dir + 'wig/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_nqc(self, predictions_dir=None):
        print('\n -- NQC -- \n')
        predictor_exe = 'python3.6 ~/repos/IRQPP/nqc.py'
        ql_scores = '~/QppUqvProj/Results/{}/test/{}/CE.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ql_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'nqc/predictions/'
        else:
            predictions_dir = predictions_dir + 'nqc/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_qf(self, predictions_dir=None):
        print('\n -- QF -- \n')
        if self.gen_lists:
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
        if self.gen_lists:
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

    def calc_aggregations(self, predictor):
        print('----- Calculating {} aggregated predictions results -----'.format(predictor))
        script = 'python3.6 ~/repos/IRQPP/aggregateUQV.py'
        raw_dir = os.path.normpath('{}/{}/predictions'.format(self.predictions_dir, predictor))
        res_files = glob.glob('{}/*predictions*'.format(raw_dir))
        for file in res_files:
            for func in AGGREGATE_FUNCTIONS:
                predictions_dir = self.predictions_dir.replace('raw', 'aggregated')
                n = file.split('-')[-1]
                output = '{}/{}/{}/predictions/predictions-{}'.format(predictions_dir, func, predictor, n)
                ensure_dir(output)
                ensure_files([script.split(' ')[1], file])
                run('{} -p {} -f {} > {}'.format(script, file, func, output), shell=True)

    def calc_singles(self, predictor):
        print('----- Calculating {} single predictions results -----'.format(predictor))
        script = 'python3.6 ~/repos/IRQPP/singleUQV.py'
        raw_dir = os.path.normpath('{}/{}/predictions'.format(self.predictions_dir, predictor))
        map_raw = '~/QppUqvProj/Results/{}/test/{}/QLmap1000'.format(self.corpus, self.qtype)
        res_files = glob.glob('{}/*predictions*'.format(raw_dir))
        for file in res_files:
            for func in SINGLE_FUNCTIONS:
                predictions_dir = self.predictions_dir.replace('raw', 'single')
                n = file.split('-')[-1]
                output = '{}/{}/{}/predictions/predictions-{}'.format(predictions_dir, func, predictor, n)
                ensure_dir(output)
                ensure_files([script.split(' ')[1], file])
                run('{} {} {} -f {} > {}'.format(script, map_raw, file, func, output), shell=True)


class CrossVal:
    def __init__(self, base_dir, cv_map_file, correlation_measure):
        self.base_dir = os.path.normpath(os.path.expanduser(base_dir))
        self.test_dir = os.path.normpath('{}/test/'.format(self.base_dir))
        self.cv_map_f = ensure_file(cv_map_file)
        self.corr_measure = correlation_measure

    def calc_aggregated(self, aggregation):
        test_dir = os.path.normpath('{}/aggregated'.format(self.test_dir))
        predictions_dir = '{}/uqvPredictions/aggregated/{}'.format(self.base_dir, aggregation)
        _results = defaultdict()
        for p in PREDICTORS:
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            _p_res = list()
            for agg in AGGREGATE_FUNCTIONS:
                ap_score = ensure_file('{}/map1000-{}'.format(test_dir, agg))
                cv_obj = CrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=predictions_dir,
                                         file_to_load=self.cv_map_f, load=True, test=self.corr_measure,
                                         ap_file=ap_score)
                mean = cv_obj.calc_test_results()
                _p_res.append(mean)
            sr = pd.Series(_p_res)
            sr.name = p
            _results[p] = sr
        print(pd.DataFrame.from_dict(_results))
        print(pd.DataFrame.from_dict(_results, orient='index'))
        print(pd.DataFrame.from_records(_results))
        return pd.DataFrame.from_dict(_results, orient='index')

    def create_table(self):
        _list = []
        for agg in AGGREGATE_FUNCTIONS:
            _df = self.calc_aggregated(agg)
            _df.columns = AGGREGATE_FUNCTIONS
            _list.append(_df)
        print(pd.concat(_list))


class GenerateTable:
    def __init__(self):
        pass


def ensure_dir(file_path):
    # tilde expansion
    file_path = os.path.expanduser(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_files(files):
    for file in files:
        for _file in file.split(' '):
            ensure_file(_file)


def ensure_file(file):
    """Ensure a single file exists, returns the full path of the file if True"""
    # tilde expansion
    file_path = os.path.expanduser(file)
    assert os.path.isfile(file_path), "The file {} doesn't exist. Please create the file first".format(file)
    return file_path


def main(args):
    generate_functions = {'clarity': GeneratePredictions.generate_clartiy,
                          'nqc': GeneratePredictions.generate_nqc,
                          'wig': GeneratePredictions.generate_wig,
                          'qf': GeneratePredictions.generate_qf,
                          'uef': GeneratePredictions.generate_uef}

    calc_functions = {'single': GeneratePredictions.calc_singles,
                      'aggregated': GeneratePredictions.calc_aggregations}

    queries = args.queries
    corpus = args.corpus
    queries_type = args.qtype
    corr_measure = args.measure
    generate = args.generate
    predictor = args.predictor
    generate_lists = args.lists
    predictions_dir = '~/QppUqvProj/Results/{}/'.format(corpus)
    cv_map_file = '{}/test/2_folds_30_repetitions.json'.format(predictions_dir)

    if queries_type == 'aggregated' or queries_type == 'single':
        predictions_dir = '{}/uqvPredictions/raw'.format(predictions_dir)
    else:
        predictions_dir = '{}/basicPredictions'.format(predictions_dir)

    predict = GeneratePredictions(queries, predictions_dir, corpus, queries_type, generate_lists)

    if generate:
        # Special case for generating results
        if predictor == 'all':
            for pred in PREDICTORS + ['uef']:
                generation_timer = Timer('{} generating'.format(pred))
                method = generate_functions.get(pred, None)
                assert method is not None, 'No applicable generate function found for {}'.format(pred)
                method(predict)
                generation_timer.stop()
        else:
            generation_timer = Timer('{} generating'.format(predictor))
            method = generate_functions.get(predictor, None)
            assert method is not None, 'No applicable generate function found for {}'.format(predictor)
            method(predict)
            generation_timer.stop()

    if predictor == 'all':
        if queries_type != 'basic':
            for pred in PREDICTORS:
                method = calc_functions.get(queries_type, None)
                assert method is not None, 'No applicable calculation function found for {}'.format(queries_type)
                method(predict, pred)
                method(predict, 'uef/{}'.format(pred))
    else:
        if queries_type != 'basic':
            method = calc_functions.get(queries_type, None)
            assert method is not None, 'No applicable calculation function found for {}'.format(queries_type)
            if predictor == 'uef':
                method(predict, 'uef/{}'.format(predictor))
            else:
                method(predict, predictor)
    base_dir = '~/QppUqvProj/Results/{}'.format(corpus)
    cv = CrossVal(base_dir=base_dir, cv_map_file=cv_map_file, correlation_measure='pearson')
    cv.calc_aggregated('avg')


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
