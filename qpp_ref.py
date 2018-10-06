import pandas as pd
import numpy as np
import os
import dataparser as dp
from crossval import CrossValidation
from features import features_loader
import argparse
from Timer.timer import Timer

parser = argparse.ArgumentParser(description='Query Prediction Using Reference lists',
                                 usage='python3.6 qpp_ref.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the files')

parser.add_argument('-c', '--corpus', type=str, default=None, help='corpus to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-a', '--aggregate', default='avg', type=str, help='Aggregate function')
parser.add_argument('-p', '--predictor', default=None, type=str, help='full CV results JSON file to load',
                    choices=['clarity', 'wig', 'nqc', 'qf'])
parser.add_argument('--uef', help='Add this if the predictor is in uef framework', action="store_true")
parser.add_argument('--corr_measure', default='pearson', type=str, choices=['pearson', 'spearman', 'kendall'],
                    help='features JSON file to load')
parser.add_argument('--generate', help='Add this to generate new results, make sure to RM the previous results',
                    action="store_true")
# parser.add_argument('--fine',
#                     help='Add this to generate new results, with fine tuning of parameters (may cause overfitting)',
#                     action="store_true")

LAMBDA = np.linspace(start=0.1, stop=0.9, num=8)


class QueryPredictionRef:

    def __init__(self, predictor, corpus, corr_measure='pearson'):
        self.__set_paths(corpus, predictor)
        self.working_dir = self.vars_results_dir.replace('predictions', 'ltr')
        self.var_cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.vars_results_dir, test=corr_measure)
        self.base_cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.base_results_dir,
                                       test=corr_measure)
        self.folds_df = self.base_cv.data_sets_map
        self.vars_results_df = self.var_cv.full_set
        self.base_results_df = self.base_cv.full_set.rename_axis('topic')
        self.feature_names = ['Jac_coefficient', 'Top_10_Docs_overlap', 'RBO_EXT_100', 'RBO_FUSED_EXT_100']  # LTR-few
        features_df = features_loader(self.features, corpus)
        self.features_df = features_df.filter(items=self.feature_names)
        self.query_vars = dp.QueriesTextParser(self.query_vars_file, 'uqv')

    @classmethod
    def __set_paths(cls, corpus, predictor):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        cls.predictor = predictor
        _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/'
        _base_dir = os.path.normpath(os.path.expanduser(_base_dir))
        cls.vars_results_dir = '{}/raw/{}/predictions/'.format(_base_dir, cls.predictor)
        cls.output_dir = '{}/referenceLists/'.format(_base_dir)
        dp.ensure_dir(cls.output_dir)
        _test_dir = f'~/QppUqvProj/Results/{corpus}/test/'
        _test_dir = os.path.normpath(os.path.expanduser(_test_dir))
        cls.folds = '{}/2_folds_30_repetitions.json'.format(_test_dir)
        dp.ensure_file(cls.folds)
        cls.features = '{}/raw/query_features_{}_uqv.JSON'.format(_test_dir, corpus)
        dp.ensure_file(cls.features)
        _orig_dir = f'~/QppUqvProj/Results/{corpus}/basicPredictions/'
        _orig_dir = os.path.normpath(os.path.expanduser(_orig_dir))
        cls.base_results_dir = f'{_orig_dir}/{predictor}/predictions/'
        _query_vars = f'~/QppUqvProj/data/{corpus}/queries_ROBUST_UQV_only.txt'
        cls.query_vars_file = os.path.normpath(os.path.expanduser(_query_vars))
        dp.ensure_file(cls.query_vars_file)

    def calc_queries(self):
        _topic_df = self.features_df.reset_index()[['topic', 'qid']]
        _vars_list = self.query_vars.queries_df['qid']
        _var_scores_df = pd.merge(_topic_df, self.vars_results_df, on='qid')
        _features_df = self.features_df.reset_index()
        _features_df = _features_df.loc[_features_df['qid'].isin(_vars_list)]
        _var_scores_df = _var_scores_df.loc[_var_scores_df['qid'].isin(_vars_list)]
        for lambda_param in LAMBDA:
            _jac_res_df = self.__calc_jac(_var_scores_df, _features_df[['topic', 'qid', 'Jac_coefficient']],
                                          lambda_param)
            self.write_results(_jac_res_df, 'jac', lambda_param)

            _sim_res_df = self.__calc_top_sim(_var_scores_df, _features_df[['topic', 'qid', 'Top_10_Docs_overlap']],
                                              lambda_param)
            self.write_results(_sim_res_df, 'sim', lambda_param)

            _rbo_res_df = self.__calc_rbo(_var_scores_df, _features_df[['topic', 'qid', 'RBO_EXT_100']], lambda_param)
            self.write_results(_rbo_res_df, 'rbo', lambda_param)

            _rbof_res_df = self.__calc_rbof(_var_scores_df, _features_df[['topic', 'qid', 'RBO_FUSED_EXT_100']],
                                            lambda_param)
            self.write_results(_rbo_res_df, 'rbof', lambda_param)

            _uni_res_df = self.__calc_uni(_var_scores_df, lambda_param)
            self.write_results(_uni_res_df, 'uni', lambda_param)

    def __calc_uni(self, var_scores_df, lambda_param):
        return lambda_param * self.base_results_df + (1 - lambda_param) * var_scores_df.groupby('topic').mean()

    def __calc_jac(self, var_scores_df, features_df, lambda_param):
        _var_scores_df = var_scores_df.set_index(['topic', 'qid'])
        _features_df = features_df.set_index(['topic', 'qid'])
        _result_df = _var_scores_df.multiply(_features_df['Jac_coefficient'], axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_top_sim(self, var_scores_df, features_df, lambda_param):
        _var_scores_df = var_scores_df.set_index(['topic', 'qid'])
        _features_df = features_df.set_index(['topic', 'qid'])
        _result_df = _var_scores_df.multiply(_features_df['Top_10_Docs_overlap'], axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_rbo(self, var_scores_df, features_df, lambda_param):
        _var_scores_df = var_scores_df.set_index(['topic', 'qid'])
        _features_df = features_df.set_index(['topic', 'qid'])
        _result_df = _var_scores_df.multiply(_features_df['RBO_EXT_100'], axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def __calc_rbof(self, var_scores_df, features_df, lambda_param):
        _var_scores_df = var_scores_df.set_index(['topic', 'qid'])
        _features_df = features_df.set_index(['topic', 'qid'])
        _result_df = _var_scores_df.multiply(_features_df['RBO_FUSED_EXT_100'], axis=0, level='qid')
        _result_df = _result_df.groupby('topic').sum()
        return lambda_param * self.base_results_df + (1 - lambda_param) * _result_df

    def write_results(self, df, simfunc, lambda_param):
        for col in df.columns:
            _file_path = f'{self.output_dir}{simfunc}/{self.predictor}/predictions/'
            dp.ensure_dir(_file_path)
            _file_name = col.replace('score_', 'predictions-')
            file_name = f'{_file_path}{_file_name}+lambda+{lambda_param}'
            df[col].to_csv(file_name, sep=" ", header=False, index=True)


def main(args):
    corpus = args.corpus
    predictor = args.predictor
    # agg_func = args.aggregate
    uef = args.uef
    corr_measure = args.corr_measure
    # generate = args.generate
    # fine_tune = args.fine

    assert predictor is not None, 'No predictor was chosen'
    assert corpus is not None, 'No corpus was chosen'

    if uef:
        predictor = f'uef/{predictor}'

    y = QueryPredictionRef(predictor, corpus, corr_measure=corr_measure)
    y.calc_queries()


#     QppUqvProj/data/ROBUST/queries_ROBUST_UQV_only.txt


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
