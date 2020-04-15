#!/usr/bin/env python3

import argparse
import glob
import os
from collections import defaultdict
from subprocess import run
import multiprocessing as mp
from functools import partial

import pandas as pd

from Timer import Timer
from crossval import InterTopicCrossValidation
import pageRank.pr_eval as pr

# TODO: Add a check that all necessary files exist on startup (to avoid later crush)
# TODO: Add parallelization where possible
# TODO: Create a single set_paths class method and replace all the paths with parameters
# TODO: Replace the print functions with logger output

# PREDICTORS = ['clarity', 'nqc', 'wig', 'smv', 'qf', 'rsd']
PREDICTORS = ['clarity', 'nqc', 'wig', 'smv', 'qf']
UEF_PREDICTORS = ['uef/{}'.format(p) for p in PREDICTORS]
UEF_PREDICTORS.remove('uef/smv')
# UEF_PREDICTORS.remove('uef/rsd')
PRE_RET_PREDICTORS = ['preret/AvgIDF', 'preret/AvgSCQTFIDF', 'preret/AvgVarTFIDF', 'preret/MaxIDF',
                      'preret/MaxSCQTFIDF', 'preret/MaxVarTFIDF']
PREDICTORS = PRE_RET_PREDICTORS + PREDICTORS
# PREDICTORS = ['preret/AvgSCQTFIDF', 'preret/MaxIDF', 'wig']
# UEF_PREDICTORS = ['uef/clarity']
# SIM_REF_PREDICTORS = {'jcP': 'Jaccard', 'topDocsP': 'TopDocs', 'rboP': 'RBO', 'FrboP': 'RBO-F', 'geo': 'GEO'}
SIM_REF_PREDICTORS = {'jcP': 'Jaccard', 'topDocsP': 'TopDocs', 'rboP': 'RBO', 'FrboP': 'RBO-F', 'geo': 'GEO'}
NUM_DOCS = [5, 10, 25, 50, 100, 250, 500, 1000]
LIST_CUT_OFF = [5, 10, 25, 50, 100]
QUANTILES = ['all', 'high', 'low-0', 'cref']
# AGGREGATE_FUNCTIONS = ['avg', 'max', 'med', 'min', 'std', 'combsum']
AGGREGATE_FUNCTIONS = ['avg', 'max', 'med', 'std']
# SINGLE_FUNCTIONS = ['title', 'top', 'low', 'medh']
SINGLE_FUNCTIONS = ['top', 'low', 'medh', 'title']
CORR_MEASURES = ['pearson', 'kendall']
REFERENCE_FUNCTIONS = ['uni', 'jac', 'sim', 'rbo', 'rbof', 'geo']
REFERENCE_TITLES = ['Uniform', 'Jaccard', 'TopDocs', 'RBO', 'RBO-F', 'GEO']
# QUERY_GROUPS = {'title': 'Title', 'top': 'MaxAP', 'low': 'MinAP', 'medh': 'MedHiAP', 'medl': 'MedLoAP'}
QUERY_GROUPS = {'top': 'MaxAP', 'low': 'MinAP', 'medh': 'MedHiAP', 'title': 'Title'}
SPLITS = 2
REPEATS = 30

parser = argparse.ArgumentParser(description='Full Results Pipeline Automation Generator',
                                 usage='python3.6 generate_results.py --predictor PREDICTOR -c CORPUS -q QUERIES ',
                                 epilog='Currently Beta Version')

parser.add_argument('--predictor', help='predictor to run', choices=PREDICTORS + ['uef', 'all'])
# parser.add_argument('-r', '--predictions_dir', metavar='results_dir_path',
#                     default='~/QppUqvProj/Results/ROBUST/basicPredictions/', help='path where to save results')
parser.add_argument('-q', '--queries', metavar='queries.xml', default='~/data/ROBUST/queries.xml',
                    help='path to queries xml res')
parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('--qtype', default='basic', type=str, help='The type of queries to run',
                    choices=['basic', 'single', 'aggregated', 'fusion'])
parser.add_argument('-m', '--measure', default='pearson', type=str,
                    help='default correlation measure type is pearson', choices=['pearson', 'spearman', 'kendall'], )
parser.add_argument('-t', '--table', type=str, default='all', help='the LaTeX table to be printed',
                    choices=['basic', 'single', 'aggregated', 'fusion', 'referenceLists', 'SimRefPred', 'pageRank',
                             'all'])
parser.add_argument('--generate', help="generate new predictions", action="store_true")
parser.add_argument('--lists', help="generate new lists", action="store_true")
parser.add_argument('--svm', help="generate SVM predictions", action="store_true")
parser.add_argument('--calc', help="calc new UQV predictions", action="store_true")
parser.add_argument('--oracle', help="calc new UQV predictions", action="store_true")


class GeneratePredictions:
    def __init__(self, queries, predictions_dir, corpus, qtype, lists):
        """
        :param queries: queries XML file
        :param predictions_dir: default predictions results dir
        """
        self.queries = queries
        self.predictions_dir = os.path.normpath(os.path.expanduser(predictions_dir)) + '/'
        self.corpus = corpus
        self.qtype = qtype if (qtype == 'basic' or qtype == 'fusion') else 'raw'
        self.gen_lists = lists
        self.cpu_cores = max(mp.cpu_count() * 0.5, min(mp.cpu_count(), 16))

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

        elif predictor_exe.endswith(('nqc.py', 'wig.py', 'smv.py')):
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
        ql_scores = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
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
        ql_scores = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ql_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'nqc/predictions/'
        else:
            predictions_dir = predictions_dir + 'nqc/predictions/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def generate_smv(self, predictions_dir=None):
        print('\n -- SMV -- \n')
        predictor_exe = 'python3 ~/repos/IRQPP/smv.py'
        ql_scores = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
        qlc_scores = '~/QppUqvProj/Results/{}/test/{}/logqlc.res'.format(self.corpus, self.qtype)
        parameters = '{} {}'.format(ql_scores, qlc_scores)
        running_param = '-d '
        if predictions_dir is None:
            predictions_dir = self.predictions_dir + 'smv/predictions/'
        else:
            predictions_dir = predictions_dir + 'smv/predictions/'
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
        predictor_exe = 'python3 ~/repos/IRQPP/uef/uef.py'
        parameters = '~/QppUqvProj/Results/{}/test/{}/QL.res'.format(self.corpus, self.qtype)
        running_param = '-d '
        predictions_dir = self.predictions_dir + 'uef/'
        self.__run_predictor(predictions_dir, predictor_exe, parameters, running_param)

    def _generate_lists_uef(self):
        predictor_exe = 'python3 ~/repos/IRQPP/addWorkingsetdocs.py'
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
        script = 'python3.7 ~/repos/IRQPP/aggregateUQV.py'
        raw_dir = os.path.normpath('{}/{}/predictions'.format(self.predictions_dir, predictor))
        res_files = glob.glob('{}/*predictions*'.format(raw_dir))
        for file in res_files:
            for func in AGGREGATE_FUNCTIONS + ['min']:
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

        for p in PREDICTORS + UEF_PREDICTORS:
            # dir of aggregated prediction results
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            # dir of aggregated uef prediction results
            # _uef_predictions_dir = os.path.normpath('{}/uef/{}/predictions'.format(predictions_dir, p))
            # list to save non uef results for a specific predictor with different AP files
            _p_res = list()
            # list to save uef results
            # _uef_p_res = list()
            _index = list()
            for agg in AGGREGATE_FUNCTIONS + ['combsum']:
                ap_score = ensure_file('{}/map1000-{}'.format(test_dir, agg))
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, test=self.corr_measure,
                                                   ap_file=ap_score)
                # uef_cv_obj = CrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_uef_predictions_dir,
                #                              file_to_load=self.cv_map_f, load=True, test=self.corr_measure,
                #                              ap_file=ap_score)
                mean = cv_obj.calc_test_results()
                # uef_mean = uef_cv_obj.calc_test_results()
                # _p_res.append('${}$'.format(mean))
                _p_res.append(mean)
                # _uef_p_res.append('${}$'.format(uef_mean))
                _index.append(agg)

            sr = pd.Series(_p_res)
            # uef_sr = pd.Series(_uef_p_res)
            sr.name = p
            sr.index = _index
            # uef_p = 'uef({})'.format(p)
            # uef_sr.name = uef_p
            # uef_sr.index = _index
            _results[p] = sr
            # _results[uef_p] = uef_sr

        res_df = pd.DataFrame.from_dict(_results, orient='index')
        # _uef_predictors = ['uef({})'.format(p) for p in PREDICTORS]
        res_df = res_df.reindex(index=PREDICTORS + UEF_PREDICTORS)
        res_df.index.name = 'predictor'
        res_df.index = res_df.index.str.upper()
        res_df.reset_index(inplace=True)
        res_df.insert(loc=0, column='predictor-agg', value=aggregation)
        # res_df.columns = ['predictor-agg', 'predictor'] + AGGREGATE_FUNCTIONS + ['combsum']
        res_df = res_df.reindex(['predictor-agg', 'predictor'] + AGGREGATE_FUNCTIONS + ['combsum'], axis='columns')
        return res_df

    def calc_fusion(self):
        test_dir = os.path.normpath('{}/aggregated'.format(self.test_dir))
        predictions_dir = '{}/uqvPredictions/fusion'.format(self.base_dir)
        _results = defaultdict()

        for p in ['nqc', 'wig']:
            # dir of aggregated prediction results
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            # dir of aggregated uef prediction results
            _uef_predictions_dir = os.path.normpath('{}/uef/{}/predictions'.format(predictions_dir, p))
            # list to save non uef results for a specific predictor with different AP files
            _p_res = list()
            # list to save uef results
            _uef_p_res = list()
            _index = list()
            for agg in AGGREGATE_FUNCTIONS + ['combsum']:
                ap_score = ensure_file('{}/map1000-{}'.format(test_dir, agg))
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, test=self.corr_measure,
                                                   ap_file=ap_score)
                # uef_cv_obj = CrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_uef_predictions_dir,
                #                              file_to_load=self.cv_map_f, load=True, test=self.corr_measure,
                #                              ap_file=ap_score)
                mean = cv_obj.calc_test_results()
                # uef_mean = uef_cv_obj.calc_test_results()
                _p_res.append('${}$'.format(mean))
                _index.append(agg)
                # _uef_p_res.append('${}$'.format(uef_mean))

            sr = pd.Series(_p_res)
            uef_sr = pd.Series(_uef_p_res)
            sr.name = p
            sr.index = _index
            uef_p = 'uef({})'.format(p)
            uef_sr.name = uef_p
            uef_sr.index = _index
            _results[p] = sr
            _results[uef_p] = uef_sr

        res_df = pd.DataFrame.from_dict(_results, orient='index')
        _uef_predictors = ['uef({})'.format(p) for p in PREDICTORS]
        res_df = res_df.reindex(index=PREDICTORS + _uef_predictors)
        res_df.index.names = ['predictor-agg', 'predictor']
        res_df.index = res_df.index.str.upper()
        res_df.reset_index(inplace=True)
        res_df.insert(loc=0, column='pred-agg', value='combsum')
        # res_df.columns = ['predictor-agg', 'predictor'] + AGGREGATE_FUNCTIONS + ['combsum']
        res_df = res_df.reindex(['predictor-agg', 'predictor'] + AGGREGATE_FUNCTIONS + ['combsum'], axis='columns')
        return res_df

    def calc_single(self, single_f):
        test_dir = os.path.normpath('{}/ref'.format(self.test_dir))
        predictions_dir = '{}/basicPredictions/{}'.format(self.base_dir, single_f)
        ap_score = ensure_file('{}/QLmap1000-{}'.format(test_dir, single_f))

        _results = defaultdict()

        for p in PREDICTORS + UEF_PREDICTORS:
            # dir of aggregated prediction results
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            # dir of aggregated uef prediction results
            # _uef_predictions_dir = os.path.normpath('{}/uef/{}/predictions'.format(predictions_dir, p))
            # list to save non uef results for a specific predictor with different AP files
            _p_res = list()
            # list to save uef results
            # _uef_p_res = list()
            _index = list()
            for measure in CORR_MEASURES:
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, test=measure,
                                                   ap_file=ap_score)
                # uef_cv_obj = CrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_uef_predictions_dir,
                #                              file_to_load=self.cv_map_f, load=True, test=measure,
                #                              ap_file=ap_score)
                mean = cv_obj.calc_test_results()
                # uef_mean = uef_cv_obj.calc_test_results()
                _p_res.append(float(mean))
                # _uef_p_res.append('${}$'.format(uef_mean))
                _index.append(measure)

            sr = pd.Series(_p_res)
            # uef_sr = pd.Series(_uef_p_res)
            sr.name = p
            sr.index = _index
            # uef_p = 'uef({})'.format(p)
            # uef_sr.name = uef_p
            # uef_sr.index = _index
            _results[p] = sr
            # _results[uef_p] = uef_sr

        res_df = pd.DataFrame.from_dict(_results, orient='index')
        # _uef_predictors = ['uef({})'.format(p) for p in PREDICTORS]
        res_df = res_df.reindex(index=PREDICTORS + UEF_PREDICTORS)
        res_df.index = res_df.index.str.upper()
        res_df.index.name = 'Predictor'
        res_df.reset_index(inplace=True)
        # res_df.columns = ['Predictor'] + CORR_MEASURES
        res_df = res_df.reindex(['Predictor'] + CORR_MEASURES, axis='columns')
        res_df.insert(loc=0, column='Function', value=single_f)
        return res_df

    def calc_reference_per_predictor(self, predictor, query_group, oracle=False):
        max_list = []
        _results = defaultdict()

        ap_file = os.path.normpath(f'{self.test_dir}/ref/QLmap1000-{query_group}')
        ensure_file(ap_file)
        base_pred_dir = f'{self.base_dir}/basicPredictions/{query_group}'

        """This part calculates and adds title queries column"""
        _base_predictions_dir = os.path.normpath('{}/{}/predictions'.format(base_pred_dir, predictor))
        cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_base_predictions_dir,
                                           folds_map_file=self.cv_map_f, load=True, ap_file=ap_file, test=self.corr_measure)
        _mean = cv_obj.calc_test_results()
        max_list.append(_mean)

        for quant in QUANTILES:

            if oracle:
                predictions_dir = f'{self.base_dir}/uqvPredictions/referenceLists/{query_group}/{quant}_vars/oracle'
            else:
                predictions_dir = f'{self.base_dir}/uqvPredictions/referenceLists/{query_group}/{quant}_vars/general'
            # list to save results for a specific predictor with different quantile variations
            _quant_res = [_mean]
            _index = [QUERY_GROUPS[query_group]]

            for ref_func, func_name in zip(REFERENCE_FUNCTIONS, REFERENCE_TITLES):
                _predictions_dir = os.path.normpath(f'{predictions_dir}/{ref_func}/{predictor}/predictions')
                _uef_predictions_dir = os.path.normpath(f'{predictions_dir}/{ref_func}/uef/{predictor}/predictions')
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, ap_file=ap_file, test=self.corr_measure)
                mean = cv_obj.calc_test_results()
                max_list.append(mean)
                _quant_res.append(mean)
                _index.append(func_name)

            sr = pd.Series(_quant_res)
            sr.index = _index
            sr.name = quant
            _results[quant] = sr

        res_df = pd.DataFrame.from_dict(_results, orient='index')
        res_df = res_df.reindex(index=QUANTILES)
        res_df.index = res_df.index.str.title()
        res_df.index.name = 'Quantile'
        res_df.reset_index(inplace=True)
        res_df = res_df.reindex(['Quantile', QUERY_GROUPS[query_group]] + REFERENCE_TITLES, axis='columns')
        res_df.insert(loc=0, column='Predictor', value=predictor.upper())
        return res_df, max(max_list)

    def calc_sim_ref_per_group(self, qgroup):
        max_list = []
        _results = defaultdict()
        ref_dir = f'{self.base_dir}/uqvPredictions/referenceLists'
        ap_file = os.path.normpath(f'{self.test_dir}/ref/QLmap1000-{qgroup}')
        ensure_file(ap_file)
        for quant in QUANTILES:
            # list to save results for a specific predictor with different quantile variations
            _quant_res = list()
            _index = list()
            for predictor in SIM_REF_PREDICTORS:
                _predictions_dir = os.path.normpath(
                    f'{ref_dir}/{qgroup}/{quant}_vars/sim_as_pred/{predictor}/predictions')
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, ap_file=ap_file, test=self.corr_measure)
                mean = cv_obj.calc_test_results()
                max_list.append(mean)
                _quant_res.append(mean)
                _index.append(SIM_REF_PREDICTORS[predictor])
            sr = pd.Series(_quant_res)
            sr.name = quant
            sr.index = _index
            _results[quant] = sr
        res_df = pd.DataFrame.from_dict(_results, orient='index')
        res_df = res_df.reindex(index=QUANTILES)
        res_df.index = res_df.index.str.title()
        res_df.index.name = 'Quantile'
        res_df.reset_index(inplace=True)
        res_df.insert(loc=0, column='Uniform', value='-')
        res_df = res_df.reindex(['Quantile'] + REFERENCE_TITLES, axis='columns')
        res_df.insert(loc=1, column=QUERY_GROUPS[qgroup], value='-')
        res_df.insert(loc=0, column='Predictor', value='SimilarityOnly')
        return res_df, max(max_list)

    def calc_sim_ref_per_predictor(self, predictor):
        _results = defaultdict()
        ref_dir = f'{self.base_dir}/uqvPredictions/referenceLists'
        for quant in ['all', 'med', 'top', 'low', 'low-0']:
            # list to save results for a specific predictor with different quantile variations
            _quant_res = list()
            _index = list()
            for qgroup, query_group in QUERY_GROUPS.items():
                _predictions_dir = os.path.normpath(
                    f'{ref_dir}/{qgroup}/{quant}_vars/sim_as_pred/{predictor}/predictions')
                ap_file = os.path.normpath(f'{self.test_dir}/ref/QLmap1000-{qgroup}')
                ensure_file(ap_file)
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, ap_file=ap_file, test=self.corr_measure)
                mean = cv_obj.calc_test_results()
                _quant_res.append('${}$'.format(mean))
                _index.append(query_group)
            sr = pd.Series(_quant_res)
            sr.name = quant
            sr.index = _index
            _results[quant] = sr
        res_df = pd.DataFrame.from_dict(_results, orient='index')
        res_df = res_df.reindex(index=['all', 'med', 'top', 'low', 'low-0'])
        res_df.index = res_df.index.str.title()
        res_df.index.name = 'Quantile'
        res_df.reset_index(inplace=True)
        res_df = res_df.reindex(['Quantile'] + _index, axis='columns')
        # res_df.rename(QUERY_GROUPS)
        res_df.insert(loc=0, column='Predictor', value=predictor)
        return res_df

    def calc_basic(self):
        test_dir = os.path.normpath('{}/basic'.format(self.test_dir))
        predictions_dir = os.path.normpath('{}/basicPredictions/title/'.format(self.base_dir))
        ap_score = ensure_file('{}/QLmap1000'.format(test_dir))

        _results = defaultdict()

        for p in PREDICTORS:
            # dir of aggregated prediction results
            _predictions_dir = os.path.normpath('{}/{}/predictions'.format(predictions_dir, p))
            # dir of aggregated uef prediction results
            _uef_predictions_dir = os.path.normpath('{}/uef/{}/predictions'.format(predictions_dir, p))
            # list to save non uef results for a specific predictor with different AP files
            _p_res = list()
            # list to save uef results
            _uef_p_res = list()
            for measure in CORR_MEASURES:
                cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_predictions_dir,
                                                   folds_map_file=self.cv_map_f, load=True, test=measure,
                                                   ap_file=ap_score)
                uef_cv_obj = InterTopicCrossValidation(k=SPLITS, rep=REPEATS, predictions_dir=_uef_predictions_dir,
                                                       folds_map_file=self.cv_map_f, load=True, test=measure,
                                                       ap_file=ap_score)
                mean = cv_obj.calc_test_results()
                uef_mean = uef_cv_obj.calc_test_results()
                _p_res.append('${}$'.format(mean))
                _uef_p_res.append('${}$'.format(uef_mean))

            sr = pd.Series(_p_res)
            uef_sr = pd.Series(_uef_p_res)
            sr.name = p
            uef_p = 'uef({})'.format(p)
            uef_sr.name = uef_p
            _results[p] = sr
            _results[uef_p] = uef_sr

        res_df = pd.DataFrame.from_dict(_results, orient='index')
        _uef_predictors = ['uef({})'.format(p) for p in PREDICTORS]
        res_df = res_df.reindex(index=PREDICTORS + _uef_predictors)
        res_df.index = res_df.index.str.upper()
        res_df.reset_index(inplace=True)
        res_df.columns = ['predictor'] + CORR_MEASURES
        res_df.insert(loc=0, column='Function', value='basic')
        return res_df

    def calc_pagerank_scores(self):
        corpus = 'ROBUST' if 'ROBUST' in self.base_dir else 'ClueWeb12B'
        results = defaultdict()
        for predictor in PREDICTORS:
            scores_best = {}
            scores_worst = {}
            for similarity in ['Jac_coefficient', 'RBO_EXT_100', 'RBO_FUSED_EXT_100', 'Top_10_Docs_overlap']:
                _score_best = pr.best_worst_metric(corpus, similarity, predictor, metric='best', load=True)
                _score_worst = pr.best_worst_metric(corpus, similarity, predictor, metric='worst', load=True)
                scores_best[similarity] = _score_best
                scores_worst[similarity] = _score_worst
            b_sr = pd.Series(scores_best)
            w_sr = pd.Series(scores_worst)
            b_sr.name = 'score_best'
            w_sr.name = 'score_worst'
            results[predictor] = pd.DataFrame([b_sr, w_sr])
        res_df = pd.concat(results)
        # _uef_predictors = ['uef({})'.format(p) for p in PREDICTORS]
        # res_df = res_df.reindex(index=PREDICTORS)
        # res_df.index = res_df.index.str.upper()
        res_df.reset_index(inplace=True)
        # res_df.columns = ['predictor'] + CORR_MEASURES
        # res_df.insert(loc=0, column='Function', value='basic')
        return res_df


class GenerateTable:
    """The class implements methods to print LaTeX tables"""

    def __init__(self, cv: CrossVal, corpus):
        self.cv = cv
        self.corpus = corpus

    def print_agg_latex_table(self):
        _list = []
        print('\\begin{table}[ht!]')
        print('\\begin{center}')
        print('\\caption{{ {} UQV aggregated {} Correlations}}'.format(self.corpus, self.cv.corr_measure.capitalize()))

        _agg = AGGREGATE_FUNCTIONS[0]
        _df = self.cv.calc_aggregated(_agg)
        _list.append(_df)
        table = _df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                             index_names=False, column_format='clcccccc')
        table = table.replace('\\end{tabular}', '')
        # This will replace the 8 last substrings, using Extended Slices with negative int to copy in reverse order
        table = table[::-1].replace('{}'.format(_agg[::-1]), '', 8)[::-1]
        table = table.replace('\\toprule', '\\toprule \n & & \\multicolumn{5}{c}{AP-aggregations} \\\\')
        table = table.replace('predictor-agg', '\\multirow{{8}}{{*}}{{{}}}'.format(_agg))
        print(table)

        for agg in AGGREGATE_FUNCTIONS[1:] + ['min']:
            _df = self.cv.calc_aggregated(agg)
            _list.append(_df)
            table = _df.to_latex(header=False, multirow=False, multicolumn=False, index=False, escape=False,
                                 index_names=False, column_format='clcccccc')
            table = table.replace('\\begin{tabular}{clcccccc}', '')
            table = table.replace('\\end{tabular}', '')
            table = table.replace('{}'.format(agg), '')
            table = table.replace('\\toprule', '\\multirow{{8}}{{*}}{{{}}}'.format(agg))
            print(table)
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\end{table}')
        full_df = pd.concat(_list, axis=0)
        print(full_df)
        full_df.to_pickle(f'{self.corpus}_aggr_queries_full_results_DF.pkl')

    def print_fused_latex_table(self):
        _df = self.cv.calc_fusion()
        print(_df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                           index_names=False, column_format='lccc'))

    def print_sing_latex_table(self):
        _list = []
        print('\\begin{table}[ht!]')
        print('\\begin{center}')
        print('\\caption{{ {} UQV single picked queries}}'.format(self.corpus))
        _sing = SINGLE_FUNCTIONS[0]
        _df = self.cv.calc_single(_sing)
        _list.append(_df)

        table = _df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                             index_names=False, column_format='llccc')

        table = table.replace('\\toprule', '\\toprule \n  &&  \\multicolumn{3}{c}{Correlation method} \\\\')
        table = table.replace('Predictor', '& Predictor')
        table = table.replace(f'{_sing} &', '&')
        table = table.replace('\\midrule', f'\\midrule \n \\multirow{{8}}{{*}}{_sing}')
        table = table.replace('\\end{tabular}', '')
        print(table)
        for sing in SINGLE_FUNCTIONS[1:]:
            _df = self.cv.calc_single(sing)
            _list.append(_df)
            table = _df.to_latex(header=False, multirow=False, multicolumn=False, index=False, escape=False,
                                 index_names=False, column_format='llccc')
            table = table.replace('\\begin{tabular}{llccc}', '')
            table = table.replace('\\end{tabular}', '')
            table = table.replace('{} &'.format(sing), '&')
            table = table.replace('\\toprule', f'\\multirow{{8}}{{*}}{{{sing.title()}AP}}')
            print(table)
        print('\\end{tabular}')
        print('\\end{center}')
        print('\\end{table}')
        full_df = pd.concat(_list, axis=0)
        print(full_df)
        full_df.to_pickle(f'{self.corpus}_single_queries_full_results_DF.pkl')

    def print_basic_latex_table(self):
        _df = self.cv.calc_basic()
        print(_df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                           index_names=False, column_format='lccc'))

    def print_sim_ref_latex_table(self):
        """This functions will print a table of all the reference lists similarity predictors for all query groups with
        all the qunatile variations
        """
        corr_measure = self.cv.corr_measure.capitalize()
        print('\n\\begin{table}[ht!]')
        print('\\begin{center}')
        print(
            '\\caption{{ {} QPP-Reference lists similarity functions as predictors {} Correlations for all queries}}'.format(
                self.corpus, corr_measure, ))
        _predictor = SIM_REF_PREDICTORS[0]
        _df = self.cv.calc_sim_ref_per_predictor(_predictor)
        table = _df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                             index_names=False, column_format='lccccccc')
        table = table.replace('\\end{tabular}', '')
        # This will replace the 8 last substrings, using Extended Slices with negative int to copy in reverse order
        # table = table[::-1].replace('{}'.format(_predictor.upper()[::-1]), '', 4)[::-1]
        table = table.replace(f'{_predictor}', '')
        table = table.replace('\\midrule', f'\\midrule \n \\multirow{{5}}{{*}}{{{_predictor}}}')
        table = table.replace('\\toprule', f'\\toprule \n & & \\multicolumn{{5}}{{c}}{{Queries Group}} \\\\')
        print(table, end='')
        for predictor in SIM_REF_PREDICTORS[1:]:
            _df = self.cv.calc_sim_ref_per_predictor(predictor)
            table = _df.to_latex(header=False, multirow=False, multicolumn=False, index=False, escape=False,
                                 index_names=False, column_format='lcccccc')
            table = table.replace('\\begin{tabular}{lcccccc}', '')
            table = table.replace('\\end{tabular}', '')
            table = table.replace(f'{predictor}', '')
            table = table.replace('\\toprule', '\\multirow{{5}}{{*}}{{{}}}'.format(predictor))
            print(table, end='')

        print('\\end{tabular}')
        print('\\end{center}')
        print('\\end{table} \n')

    def print_ref_latex_table(self, oracle=False):
        """This functions will print tables with all the qunatile variations, with all the predictors inside. Good for
         comparing the influence of the variations quantile, it prints a separate table for each group of queries"""
        corr_measure = self.cv.corr_measure.capitalize()
        _predictor = PREDICTORS[0]
        for qgroup, queries_group in QUERY_GROUPS.items():
            _list = []
            tables_max_vals = []
            print('\n\\begin{table}[ht!]')
            print('\\begin{center}')
            print(
                '\\caption{{ {} QPP-Reference lists {} Correlations for {} queries}}'.format(self.corpus, corr_measure,
                                                                                             queries_group))
            _df, _max = self.cv.calc_reference_per_predictor(_predictor, qgroup, oracle)
            _list.append(_df)
            tables_max_vals.append(_max)
            table = _df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                                 index_names=False, column_format='lccccccc')
            table = table.replace('\\end{tabular}', '')
            # This will replace the 8 last substrings, using Extended Slices with negative int to copy in reverse order
            # table = table[::-1].replace('{}'.format(_predictor.upper()[::-1]), '', 4)[::-1]
            table = table.replace(f'{_predictor.upper()}', '')
            table = table.replace('\\midrule', f'\\midrule \n \\multirow{{5}}{{*}}{{{_predictor.upper()}}}')
            table = table.replace('\\toprule', f'\\toprule \n & & \\multicolumn{{5}}{{c}}{{Similarity Functions}} \\\\')
            print(table, end='')
            # with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            #     result = pool.map(partial(self.cv.calc_reference_per_predictor, query_group=qgroup, oracle=oracle),
            #                       PREDICTORS[1:] + UEF_PREDICTORS)
            # for (_df, _max), predictor in zip(result, PREDICTORS[1:] + UEF_PREDICTORS):
            for predictor in PREDICTORS[1:] + UEF_PREDICTORS:
                _df, _max = self.cv.calc_reference_per_predictor(predictor, qgroup, oracle)
                _list.append(_df)
                tables_max_vals.append(_max)
                table = _df.to_latex(header=False, multirow=False, multicolumn=False, index=False, escape=False,
                                     index_names=False, column_format='lcccccc')
                table = table.replace('\\begin{tabular}{lcccccc}', '')
                table = table.replace('\\end{tabular}\n', '')
                table = table.replace(f'{predictor.upper()}', '')
                table = table.replace('\\toprule', '\\multirow{{5}}{{*}}{{{}}}'.format(predictor.upper()))
                print(table, end='')

            _df, _max = self.cv.calc_sim_ref_per_group(qgroup)
            _list.append(_df)

            full_res_df = pd.concat(_list)
            if oracle:
                full_res_df.to_pickle(
                    f'{self.corpus}_{queries_group}_{self.cv.corr_measure}_queries_oracle_results_DF.pkl')
            else:
                full_res_df.to_pickle(
                    f'{self.corpus}_{queries_group}_{self.cv.corr_measure}_queries_full_results_DF.pkl')

            tables_max_vals.append(_max)
            table = _df.to_latex(header=False, multirow=False, multicolumn=False, index=False, escape=False,
                                 index_names=False, column_format='lcccccc')
            table = table.replace('\\begin{tabular}{lcccccc}', '')
            table = table.replace('\\end{tabular}', '')
            table = table.replace('SimilarityOnly', '')
            table = table.replace('\\toprule', '\\multirow{5}{*}{SimilarityOnly}')
            print(table, end='')
            print('\\end{tabular}')
            print('\\end{center}')
            print(f'The maximum value in the table is: {max(tables_max_vals)}')
            print('\\end{table} \n')

    def print_pagerank_latex_table(self):
        _df = self.cv.calc_pagerank_scores()
        _df.to_pickle(f'{self.corpus}_pagerank_scores_df.pkl')
        print(_df.to_latex(header=True, multirow=False, multicolumn=False, index=False, escape=False,
                           index_names=False, column_format='lccc'))


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
                          'smv': GeneratePredictions.generate_smv,
                          'qf': GeneratePredictions.generate_qf,
                          'uef': GeneratePredictions.generate_uef}

    calc_functions = {'single': GeneratePredictions.calc_singles,
                      'aggregated': GeneratePredictions.calc_aggregations}

    table_functions = {'basic': GenerateTable.print_basic_latex_table, 'single': GenerateTable.print_sing_latex_table,
                       'aggregated': GenerateTable.print_agg_latex_table,
                       'fusion': GenerateTable.print_fused_latex_table,
                       'referenceLists': GenerateTable.print_ref_latex_table,
                       'SimRefPred': GenerateTable.print_sim_ref_latex_table,
                       'pageRank': GenerateTable.print_pagerank_latex_table}

    queries = args.queries
    corpus = args.corpus
    queries_type = args.qtype
    corr_measure = args.measure
    generate = args.generate
    predictor = args.predictor
    generate_lists = args.lists
    calc_predictions = args.calc
    generate_svm = args.svm
    table = args.table
    # Stores true if oracle tables should be printed for the QPP-Reference similarity model
    oracle = args.oracle

    # Debugging - should be in comment when not debugging !
    # print('\n------+++^+++------ Debugging !! ------+++^+++------\n')
    # corpus = 'ROBUST'
    # table = 'aggregated'
    # predictor = 'all'
    # queries_type = 'aggregated'
    # generate = True
    # calc_predictions = True

    base_dir = '~/QppUqvProj/Results/{}'.format(corpus)
    cv_map_file = '{}/test/2_folds_30_repetitions.json'.format(base_dir)

    if queries_type == 'aggregated' or queries_type == 'single':
        predictions_dir = '{}/uqvPredictions/raw'.format(base_dir)
    elif queries_type == 'fusion':
        predictions_dir = '{}/uqvPredictions/fusion'.format(base_dir)
    else:
        predictions_dir = '{}/basicPredictions/title'.format(base_dir)

    predict = GeneratePredictions(queries, predictions_dir, corpus, queries_type, generate_lists)

    if generate:
        # Special case for generating results
        assert predictor is not None, 'No predictor was chosen for results generation'
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

    if calc_predictions:
        assert predictor is not None, 'No predictor was chosen for results calculation'
        if predictor == 'all':
            if queries_type == 'single' or queries_type == 'aggregated':
                for pred in PREDICTORS + UEF_PREDICTORS:
                    method = calc_functions.get(queries_type, None)
                    assert method is not None, 'No applicable calculation function found for {}'.format(queries_type)
                    method(predict, pred)
                    # method(predict, 'uef/{}'.format(pred))
        else:
            if queries_type == 'single' or queries_type == 'aggregated':
                method = calc_functions.get(queries_type, None)
                assert method is not None, 'No applicable calculation function found for {}'.format(queries_type)
                if predictor == 'uef':
                    for p in PREDICTORS:
                        method(predict, 'uef/{}'.format(p))
                else:
                    method(predict, predictor)

    cv = CrossVal(base_dir=base_dir, cv_map_file=cv_map_file, correlation_measure=corr_measure)
    lat = GenerateTable(cv, corpus)

    if generate_svm:
        assert predictor is not None, 'No predictor was chosen for SVM prediction'
        pass

    if table == 'all':
        print('{} UQV aggregated predictions LaTeX table: \n'.format(corpus))
        lat.print_agg_latex_table()

        print('{} UQV single pick predictions LaTeX table: \n'.format(corpus))
        lat.print_sing_latex_table()

        print('{} basic predictions LaTeX table:'.format(corpus))
        lat.print_basic_latex_table()
    else:
        creation_timer = Timer('{} {} predictions LaTeX table:'.format(corpus, table))
        method = table_functions.get(table, None)
        assert method is not None, 'No applicable table function found for {}'.format(table)
        method(lat)
        creation_timer.stop()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
