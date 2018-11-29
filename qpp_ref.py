import argparse
import glob
import itertools
import multiprocessing as mp
import os
from functools import partial
from subprocess import run

import numpy as np
import pandas as pd

import dataparser as dp
from Timer.timer import Timer
from crossval import CrossValidation
from query_features import features_loader

parser = argparse.ArgumentParser(description='Query Prediction Using Reference lists',
                                 usage='python3.6 qpp_ref.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the files')

parser.add_argument('-c', '--corpus', type=str, default=None, help='corpus to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-p', '--predictor', default=None, type=str, help='Choose the predictor to use',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'all'])
parser.add_argument('--uef', help='Add this if the predictor is in uef framework', action="store_true")
parser.add_argument('-g', '--group', help='group of queries to predict',
                    choices=['top', 'low', 'title', 'medh', 'medl'])
parser.add_argument('--quantile', help='quantile of query variants to use for prediction', default='all',
                    choices=['all', 'low', 'med', 'top'])
parser.add_argument('--corr_measure', default='pearson', type=str, choices=['pearson', 'spearman', 'kendall'],
                    help='features JSON file to load')
parser.add_argument('--generate', help='use ltr to generate SVM-Rank predictions, or calc to calc predictions',
                    choices=['ltr', 'calc'])

LAMBDA = np.linspace(start=0, stop=1, num=11)
C_PARAMETERS = [0.01, 0.1, 1, 10]


class QueryPredictionRef:
    """The class reads a queries intended for prediction, it's named inside the class as queries_group or qgroup - e.g
    "title" / "top" ..
    Also reads a file with the variations without the queries to be predicted, and a file with the features constructed
    for the relevant queries with the relevant variations"""

    def __init__(self, predictor, corpus, qgroup, vars_quantile):
        self.__set_paths(corpus, predictor, qgroup, vars_quantile)
        _q2p_obj = dp.QueriesTextParser(self.queries2predict_file, 'uqv')
        self.var_cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.vars_results_dir)
        _vars_results_df = self.var_cv.full_set
        # Initialize the base prediction results of the queries to be predicted
        if qgroup == 'title':
            _base_cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.base_results_dir)
            self.base_results_df = _base_cv.full_set
        else:
            self.base_results_df = dp.convert_vid_to_qid(_vars_results_df.loc[_q2p_obj.queries_dict.keys()])

        self.base_results_df.rename_axis('topic', inplace=True)
        # # The next function is used to save results in basic predictions format of the given queries set
        # write_basic_predictions(self.base_results_df, corpus, qgroup, predictor)
        # exit()
        self.query_vars = dp.QueriesTextParser(self.query_vars_file, 'uqv')
        _quantile_vars = dp.QueriesTextParser(self.quantile_vars_file, 'uqv')
        _features_df = features_loader(self.features, corpus)
        self.features_df = self.__initialize_features_df(_quantile_vars, _features_df)
        self.var_scores_df = self.__initialize_var_scores_df(_features_df.reset_index()[['topic', 'qid']],
                                                             _vars_results_df)

    @classmethod
    def __set_paths(cls, corpus, predictor, qgroup, vars_quantile):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        cls.predictor = predictor

        _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/'
        _base_dir = os.path.normpath(os.path.expanduser(_base_dir))
        cls.vars_results_dir = '{}/raw/{}/predictions/'.format(_base_dir, cls.predictor)

        if qgroup == 'title':
            _orig_dir = f'~/QppUqvProj/Results/{corpus}/basicPredictions/title'
            _orig_dir = os.path.normpath(os.path.expanduser(_orig_dir))
            cls.base_results_dir = f'{_orig_dir}/{predictor}/predictions/'

        cls.output_dir = f'{_base_dir}/referenceLists/{qgroup}/{vars_quantile}_vars/general/'
        dp.ensure_dir(cls.output_dir)

        _test_dir = f'~/QppUqvProj/Results/{corpus}/test/'
        _test_dir = os.path.normpath(os.path.expanduser(_test_dir))
        cls.folds = f'{_test_dir}/2_folds_30_repetitions.json'
        dp.ensure_file(cls.folds)

        cls.ap_file = f'{_test_dir}/ref/QLmap1000-{qgroup}'

        # cls.features = '{}/raw/query_features_{}_uqv_legal.JSON'.format(_test_dir, corpus)
        # cls.features = f'{_test_dir}/ref/{qgroup}_query_features_{corpus}_uqv.JSON'
        cls.features = f'{_test_dir}/ref/{qgroup}_query_{vars_quantile}_variations_features_{corpus}_uqv.JSON'
        # dp.ensure_file(cls.features)

        _query_vars = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_wo_{qgroup}.txt'
        cls.query_vars_file = os.path.normpath(os.path.expanduser(_query_vars))
        dp.ensure_file(cls.query_vars_file)

        _queries2predict = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_{qgroup}.txt'
        cls.queries2predict_file = os.path.normpath(os.path.expanduser(_queries2predict))
        dp.ensure_file(cls.queries2predict_file)

        if vars_quantile == 'all':
            cls.quantile_vars_file = cls.query_vars_file
        else:
            _quantile_vars = f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_{vars_quantile}_variants.txt'
            cls.quantile_vars_file = os.path.normpath(os.path.expanduser(_quantile_vars))
            dp.ensure_file(cls.quantile_vars_file)

    def __initialize_features_df(self, quantile_vars, features_df):
        """This method filters from features df only the ones conjunction with the relevant variations"""
        _quant_vars = quantile_vars.queries_df['qid'].tolist()
        _vars_list = self.query_vars.queries_df.loc[self.query_vars.queries_df['qid'].isin(_quant_vars)]['qid']
        _features_df = features_df.reset_index()
        _features_df = _features_df.loc[_features_df['qid'].isin(_vars_list)]
        _features_df.set_index(['topic', 'qid'], inplace=True)
        return _features_df

    def __initialize_var_scores_df(self, topic_df, vars_results_df):
        """This method filters from query variations df only the ones relevant for prediction"""
        _var_scores_df = pd.merge(topic_df, vars_results_df, on='qid')
        _var_scores_df = _var_scores_df.loc[_var_scores_df['qid'].isin(self.features_df.index.get_level_values('qid'))]
        _var_scores_df.set_index(['topic', 'qid'], inplace=True)
        return _var_scores_df

    def calc_queries(self):
        for lambda_param in LAMBDA:
            _jac_res_df = self.__calc_jac(self.var_scores_df, self.features_df['Jac_coefficient'],
                                          lambda_param)
            self.write_results(_jac_res_df, 'jac', lambda_param)

            _sim_res_df = self.__calc_top_sim(self.var_scores_df, self.features_df['Top_10_Docs_overlap'],
                                              lambda_param)
            self.write_results(_sim_res_df, 'sim', lambda_param)

            _rbo_res_df = self.__calc_rbo(self.var_scores_df, self.features_df['RBO_EXT_100'], lambda_param)
            self.write_results(_rbo_res_df, 'rbo', lambda_param)

            _rbof_res_df = self.__calc_rbof(self.var_scores_df, self.features_df['RBO_FUSED_EXT_100'],
                                            lambda_param)
            self.write_results(_rbof_res_df, 'rbof', lambda_param)

            _uni_res_df = self.__calc_uni(self.var_scores_df, lambda_param)
            self.write_results(_uni_res_df, 'uni', lambda_param)

    def __calc_uni(self, var_scores_df, lambda_param):
        return lambda_param * self.base_results_df + (1 - lambda_param) * var_scores_df.groupby('topic').mean()

    def __calc_jac(self, var_scores_df, features_df, lambda_param):
        _result_df = var_scores_df.multiply(features_df, axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_top_sim(self, var_scores_df, features_df, lambda_param):
        _result_df = var_scores_df.multiply(features_df, axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_rbo(self, var_scores_df, features_df, lambda_param):
        _result_df = var_scores_df.multiply(features_df, axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_rbof(self, var_scores_df, features_df, lambda_param):
        _result_df = var_scores_df.multiply(features_df, axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def calc_integrated(self, score_param):
        """The function receives a column name from the scores df, in the shape of score_n
        Basically it implements the calculations specified in section 3.1.1 in the paper and returns a DF with the
        feature values (and the basic predictions score)"""
        _df = self.features_df.multiply(self.var_scores_df[score_param], axis=0, level='qid')
        _df = _df.groupby('topic').sum()
        _res_df = _df.join(self.base_results_df[score_param], on='topic')
        return _res_df

    def write_results(self, df, simfunc, lambda_param):
        for col in df.columns:
            _file_path = f'{self.output_dir}{simfunc}/{self.predictor}/predictions/'
            dp.ensure_dir(_file_path)
            _file_name = col.replace('score_', 'predictions-')
            file_name = f'{_file_path}{_file_name}+lambda+{lambda_param}'
            df[col].to_csv(file_name, sep=" ", header=False, index=True)


class LearningSimFunctions:
    """This class creates data sets and splits them into train and test, afterwards running svmRank to learn"""

    def __init__(self, qpp_ref: QueryPredictionRef, corr_measure='pearson'):
        self.corr_measure = corr_measure
        _predictor = qpp_ref.predictor
        self.features_df = qpp_ref.features_df
        self.results_df = qpp_ref.var_scores_df
        _ap_file = qpp_ref.ap_file
        self.ap_obj = dp.ResultsReader(_ap_file, 'ap')
        self.folds_df = qpp_ref.var_cv.data_sets_map.transpose()
        self.output_dir = f'{qpp_ref.output_dir}ltr/{_predictor}/'
        dp.ensure_dir(self.output_dir)
        self.calc_features_df = qpp_ref.calc_integrated
        self.feature_names = self.features_df.columns.tolist()
        self.cpu_cores = mp.cpu_count() - 1

    def _create_data_set(self, param):
        feat_df = self.calc_features_df(param)
        # feat_df = feat_df.apply(np.log)
        feat_df = feat_df.merge(self.ap_obj.data_df, left_index=True, right_index=True)
        feat_df.insert(0, 'qid', 'qid:1')
        return feat_df

    def _split_data_set(self, dataset_df, set_id, subset):
        set_id = int(set_id)
        train = np.array(self.folds_df.loc[set_id, subset]['train']).astype(str)
        test = np.array(self.folds_df.loc[set_id, subset]['test']).astype(str)
        return dataset_df.loc[train], dataset_df.loc[test]

    def _df_to_str(self, df, param):
        formatters = {}
        s = f'{1}' + ':{:f}'
        formatters[param] = s.format
        for i, feat in enumerate(self.feature_names, start=2):
            s = f'{i}' + ':{:f}'
            formatters[feat] = s.format
        _str_df = df.to_string(columns=['ap', 'qid', param] + self.feature_names, index=False, index_names=False,
                               header=False, float_format='%f', formatters=formatters)
        return _str_df

    def generate_data_sets_fine_tune(self):
        """This method will create the data sets with all the available hyper parameters of the qpp predictions"""
        # TODO: add prompt with list of files before delete
        # run(f'rm -rfv {self.output_dir}*', shell=True)
        for set_id in self.folds_df.index:
            for subset in ['a', 'b']:
                for col in self.results_df.columns:
                    h = col.split('_')[-1]
                    features_df = self._create_data_set(col)
                    train_df, test_df = self._split_data_set(features_df, set_id, subset)
                    train_str = self._df_to_str(train_df, col)
                    test_str = self._df_to_str(test_df, col)
                    self.write_str_to_file(train_str, f'train_{set_id}_{subset}-d_{h}.dat')
                    self.write_str_to_file(test_str, f'test_{set_id}_{subset}-d_{h}.dat')

    def generate_data_sets(self):
        """This method will create the data sets with a single hyper parameter for the qpp predictions, which will be
        chosen based on the best result on the train set"""
        run(f'rm -rfv {self.output_dir}*', shell=True)
        for set_id in self.folds_df.index:
            for subset in ['a', 'b']:
                param = self.parameters_df.loc[set_id][subset]
                features_df = self._create_data_set(param)
                train_df, test_df = self._split_data_set(features_df, set_id, subset)
                train_str = self._df_to_str(train_df)
                test_str = self._df_to_str(test_df)
                self.write_str_to_file(train_str, f'train_{set_id}_{subset}.dat')
                self.write_str_to_file(test_str, f'test_{set_id}_{subset}.dat')

    def write_str_to_file(self, string, file_name):
        datasets_dir = f'{self.output_dir}datasets'
        dp.ensure_dir(datasets_dir)
        with open(f'{datasets_dir}/{file_name}', "w") as text_file:
            print(string, file=text_file)

    def run_svm_fine_tune(self):
        models_dir = f'{self.output_dir}models'
        dp.ensure_dir(models_dir)
        classification_dir = f'{self.output_dir}classifications'
        dp.ensure_dir(classification_dir)
        dp.empty_dir(models_dir)
        dp.empty_dir(classification_dir)
        train_sets = glob.glob(f'{self.output_dir}datasets/train*')
        args_list = list(itertools.product(C_PARAMETERS, train_sets))
        if not mp.current_process().daemon:
            with mp.Pool(processes=self.cpu_cores) as pool:
                pool.starmap(partial(svm_sub_procedure, models_dir=models_dir, classification_dir=classification_dir),
                             args_list)
        else:
            for c, train_sets in args_list:
                svm_sub_procedure(c, train_sets, models_dir=models_dir, classification_dir=classification_dir)

    def run_svm(self):
        c = '1'
        svm_learn = 'svmRank/svm_rank_learn'
        svm_classify = '~/svmRank/svm_rank_classify'
        models_dir = self.output_dir.replace('datasets', 'models')
        dp.ensure_dir(models_dir)
        classification_dir = self.output_dir.replace('datasets', 'classifications')
        run(f'rm -rfv {models_dir}*', shell=True)
        run(f'rm -rfv {classification_dir}*', shell=True)
        dp.ensure_dir(classification_dir)
        for set_id in range(1, 31):
            for subset in ['a', 'b']:
                run('{0} -c {1} {2}/train_{3}_{4}.dat {5}/model_{3}_{4}'.format(svm_learn, c, self.output_dir, set_id,
                                                                                subset, models_dir), shell=True)
                run('{0} {1}/test_{2}_{3}.dat {4}/model_{2}_{3} {5}/predictions_{2}_{3}'.format(svm_classify,
                                                                                                self.output_dir, set_id,
                                                                                                subset, models_dir,
                                                                                                classification_dir),
                    shell=True)

    @staticmethod
    def _df_from_files(files):
        _list = []
        for file in files:
            _str = file.split('_', 1)[-1]
            _params = _str.strip('.cls').split('-', 1)[-1]
            _df = pd.read_csv(file, header=None, names=[_params])
            _list.append(_df)
        return pd.concat(_list, axis=1)

    def cross_val_fine_tune(self):
        classification_dir = f'{self.output_dir}classifications/'
        _list = []
        # _dict = {}
        for set_id in range(1, 31):
            _pair = []
            for subset in ['a', 'b']:
                train_files = glob.glob(classification_dir + f'train_{set_id}_{subset}-*')
                _train_df = self._df_from_files(train_files)
                _train_topics = np.array(self.folds_df.loc[set_id, subset]['train']).astype(str)
                _train_df.insert(loc=0, column='qid', value=_train_topics)
                _train_df.set_index('qid', inplace=True)
                _ap_df = self.ap_obj.data_df.loc[_train_topics]
                _df = _train_df.merge(_ap_df, how='outer', on='qid')
                _correlation_df = _df.corr(method=self.corr_measure)
                _corr = _correlation_df.drop('ap')['ap']
                max_train_param = _corr.idxmax()
                _test_file = classification_dir + f'test_{set_id}_{subset}-{max_train_param}.cls'
                _test_df = pd.read_csv(_test_file, header=None, names=['score'])
                _test_topics = np.array(self.folds_df.loc[set_id, subset]['test']).astype(str)
                _test_df.insert(loc=0, column='qid', value=_test_topics)
                _test_df.set_index('qid', inplace=True)
                _ap_df = self.ap_obj.data_df.loc[_test_topics]
                _df = _test_df.merge(_ap_df, how='outer', on='qid')
                _correlation = _df['score'].corr(_df['ap'], method=self.corr_measure)
                _pair.append(_correlation)
            _list.append(np.mean(_pair))
        print('mean: {:.3f}'.format(np.mean(_list)))
        return 'mean: {:.3f}'.format(np.mean(_list))

    def cross_val(self):
        classification_dir = self.output_dir.replace('datasets', 'classifications')
        _list = []
        for set_id in range(1, 31):
            _pair = []
            for subset in ['a', 'b']:
                _res_df = pd.read_csv(f'{classification_dir}/predictions_{set_id}_{subset}', header=None,
                                      names=['score'])
                _test_topics = np.array(self.folds_df[set_id][subset]['test']).astype(str)
                _res_df.insert(loc=0, column='qid', value=_test_topics)
                _res_df.set_index('qid', inplace=True)
                _ap_df = self.ap_obj.data_df.loc[_test_topics]
                _df = _res_df.merge(_ap_df, how='outer', on='qid')
                _correlation = _df['score'].corr(_df['ap'], method=self.corr_measure)
                _pair.append(_correlation)
            _list.append(np.mean(_pair))
        print('mean: {:.3f}'.format(np.mean(_list)))


def svm_sub_procedure(c, trainset, models_dir, classification_dir):
    svm_learn = '~/svmRank/svm_rank_learn'
    svm_classify = '~/svmRank/svm_rank_classify'
    testset = trainset.replace('train', 'test')
    _model_params = trainset.strip('.dat').split('train_', 1)[-1]
    _model_path = f'{models_dir}/model_{_model_params}_c_{c}'
    _cls_train_path = f'{classification_dir}/train_{_model_params}_c_{c}.cls'
    _cls_test_path = f'{classification_dir}/test_{_model_params}_c_{c}.cls'
    run('{0} -v 0 -c {1} {2} {3}'.format(svm_learn, c, trainset, _model_path), shell=True)
    run('{0} -v 0 {1} {2} {3}'.format(svm_classify, trainset, _model_path, _cls_train_path), shell=True)
    run('{0} -v 0 {1} {2} {3}'.format(svm_classify, testset, _model_path, _cls_test_path), shell=True)


def write_basic_predictions(df: pd.DataFrame, corpus, qgroup, predictor):
    """The function is used to save results in basic predictions format of a given queries set"""
    for col in df.columns:
        _file_path = f'~/QppUqvProj/Results/{corpus}/basicPredictions/{qgroup}/{predictor}/predictions/'
        dp.ensure_dir(os.path.normpath(os.path.expanduser(_file_path)))
        _file_name = col.replace('score_', 'predictions-')
        file_name = f'{_file_path}{_file_name}'
        df[col].to_csv(file_name, sep=" ", header=False, index=True)


def run_calc_process(pred, corpus, queries_group, quantile):
    qpp_ref = QueryPredictionRef(pred, corpus, queries_group, quantile)
    qpp_ref.calc_queries()
    return qpp_ref


def run_ltr_process(pred, corpus, queries_group, quantile, corr_measure):
    qpp_ref = QueryPredictionRef(pred, corpus, queries_group, quantile)
    qpp_ref_ltr = LearningSimFunctions(qpp_ref, corr_measure)
    qpp_ref_ltr.generate_data_sets_fine_tune()
    qpp_ref_ltr.run_svm_fine_tune()
    return qpp_ref_ltr


def main(args):
    corpus = args.corpus
    predictor = args.predictor
    queries_group = args.group
    quantile = args.quantile
    # agg_func = args.aggregate
    uef = args.uef
    corr_measure = args.corr_measure
    generate = args.generate
    # fine_tune = args.fine

    # # Debug
    # predictor = 'wig'
    # corpus = 'ROBUST'
    # quantile = 'all'
    # queries_group = 'title'
    # generate = 'ltr'

    assert predictor is not None, 'No predictor was chosen'
    assert corpus is not None, 'No corpus was chosen'

    if predictor == 'all':
        cores = mp.cpu_count() - 1
        if generate == 'calc':
            with mp.Pool(processes=cores) as pool:
                qpp_ref = pool.map(
                    partial(run_calc_process, corpus=corpus, queries_group=queries_group, quantile=quantile),
                    PREDICTORS)
        elif generate == 'ltr':
            with mp.Pool(processes=cores) as pool:
                qpp_ref_ltr = pool.map(
                    partial(run_ltr_process, corpus=corpus, queries_group=queries_group, quantile=quantile,
                            corr_measure=corr_measure), PREDICTORS_WO_QF)
            for pred in PREDICTORS_QF:
                run_ltr_process(pred, corpus, queries_group, quantile, corr_measure)
    else:
        if uef:
            predictor = f'uef/{predictor}'

        qpp_ref = QueryPredictionRef(predictor, corpus, queries_group, quantile)
        qpp_ref_ltr = LearningSimFunctions(qpp_ref, corr_measure)
        if generate == 'calc':
            run_calc_process(predictor, corpus, queries_group, quantile)
        elif generate == 'ltr':
            run_ltr_process(predictor, corpus, queries_group, quantile, corr_measure)
        qpp_ref_ltr.cross_val_fine_tune()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    PREDICTORS_WO_QF = ['clarity', 'wig', 'nqc', 'uef/clarity', 'uef/wig', 'uef/nqc']
    PREDICTORS_QF = ['qf', 'uef/qf']
    PREDICTORS = PREDICTORS_WO_QF + PREDICTORS_QF
    main(args)
    overall_timer.stop()
