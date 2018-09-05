import argparse
import os
import glob
from subprocess import run

import numpy as np
import pandas as pd

from Timer.timer import Timer
from crossval import CrossValidation
from dataparser import ResultsReader, ensure_dir
from features import features_loader

# TODO: implement the UEF addition

parser = argparse.ArgumentParser(description='LTR (SVMRank) data sets Generator',
                                 usage='python3.6 learningsets.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', type=str, help='corpus to work with', choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-a', '--aggregate', default=None, type=str, help='Aggregate function')
parser.add_argument('-p', '--predictor', default=None, type=str, help='full CV results JSON file to load',
                    choices=['clarity', 'wig', 'nqc', 'qf'])
parser.add_argument('--uef', help='Add this if the predictor is in uef framework', action="store_true")
parser.add_argument('--corr_measure', default='pearson', type=str, choices=['pearson', 'spearman', 'kendall'],
                    help='features JSON file to load')


class LearningDataSets:

    def __init__(self, predictor, corpus, corr_measure='pearson', aggregation='avg', uef=False):
        self.__set_paths(corpus, predictor, aggregation)
        self.ap_obj = ResultsReader(self.ap_file, 'ap')
        self.working_dir = self.results_dir.replace('predictions', 'ltr')
        self.cv = CrossValidation(file_to_load=self.folds, predictions_dir=self.results_dir, test=corr_measure)
        self.folds_df = self.cv.data_sets_map
        _parameters = os.path.normpath(os.path.expanduser(self.parameters))
        self.parameters_df = self.cv.read_eval_results(_parameters)
        self.results_df = self.cv.full_set
        # self.feature_names = ['Jac_coefficient', 'Top_10_Docs_overlap', 'RBO_EXT_100', 'RBO_EXT_1000',
        #                       'RBO_FUSED_EXT_100', 'RBO_FUSED_EXT_1000']
        self.feature_names = ['Jac_coefficient', 'Top_10_Docs_overlap', 'RBO_EXT_100', 'RBO_FUSED_EXT_100']
        features_df = features_loader(self.features, corpus)
        self.features_df = features_df.filter(items=self.feature_names)

    @classmethod
    def __set_paths(cls, corpus, predictor, agg):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""

        _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/'
        _base_dir = os.path.normpath(os.path.expanduser(_base_dir))
        cls.parameters = '{}/aggregated/{}/{}/evaluation/full_results_vector_for_2_folds_30_repetitions_{}.json'.format(
            _base_dir, agg, predictor, agg)
        cls.results_dir = '{}/raw/{}/predictions/'.format(_base_dir, predictor)
        cls.output_dir = '{}/aggregated/{}/{}/ltr/datasets/'.format(_base_dir, agg, predictor)
        ensure_dir(cls.output_dir)
        _test_dir = f'~/QppUqvProj/Results/{corpus}/test/'
        _test_dir = os.path.normpath(os.path.expanduser(_test_dir))
        cls.folds = '{}/2_folds_30_repetitions.json'.format(_test_dir)
        cls.features = '{}/raw/norm_features_{}_uqv.JSON'.format(_test_dir, corpus)
        cls.ap_file = '{}/aggregated/map1000-{}'.format(_test_dir, agg)

    def _create_data_set(self, param):
        predictor_resutls = self.results_df[f'score_{param}']
        feat_df = self.features_df.multiply(predictor_resutls, axis=0, level='qid')
        feat_df = feat_df.groupby('topic').sum()
        feat_df = feat_df.apply(np.log)
        feat_df = feat_df.merge(self.ap_obj.data_df, left_index=True, right_index=True)
        feat_df.insert(0, 'qid', 'qid:1')
        return feat_df

    def _split_data_set(self, dataset_df, set_id, subset):
        set_id = int(set_id)
        train = np.array(self.folds_df[set_id][subset]['train']).astype(str)
        test = np.array(self.folds_df[set_id][subset]['test']).astype(str)
        return dataset_df.loc[train], dataset_df.loc[test]

    def _df_to_str(self, df):
        formatters = {}
        for i, feat in enumerate(self.feature_names):
            j = i + 1
            s = f'{j}' + ':{:f}'
            formatters[feat] = s.format

        _df = df.to_string(columns=['ap', 'qid', ] + self.feature_names, index=False, index_names=False, header=False,
                           float_format='%f', formatters=formatters)
        return _df

    def generate_data_sets(self):
        for set_id in self.parameters_df.index:
            for subset in ['a', 'b']:
                param = self.parameters_df.loc[set_id][subset]
                features_df = self._create_data_set(param)
                train_df, test_df = self._split_data_set(features_df, set_id, subset)
                train_str = self._df_to_str(train_df)
                test_str = self._df_to_str(test_df)
                self.write_str_to_file(train_str, f'train_{set_id}_{subset}.dat')
                self.write_str_to_file(test_str, f'test_{set_id}_{subset}.dat')

    def write_str_to_file(self, string, file_name):
        with open(self.output_dir + file_name, "w") as text_file:
            print(string, file=text_file)

    def run_svm(self):
        c = '1'
        svm_learn = 'svmRank/svm_rank_learn'
        svm_classify = '~/svmRank/svm_rank_classify'
        models_dir = self.output_dir.replace('datasets', 'models')
        ensure_dir(models_dir)
        classification_dir = self.output_dir.replace('datasets', 'classifications')
        ensure_dir(classification_dir)
        for set_id in range(1, 31):
            for subset in ['a', 'b']:
                run('{0} -c {1} {2}/train_{3}_{4}.dat {5}/model_{3}_{4}'.format(svm_learn, c, self.output_dir, set_id,
                                                                                subset, models_dir), shell=True)
                run('{0} {1}/test_{2}_{3}.dat {4}/model_{2}_{3} {5}/predictions_{2}_{3}'.format(svm_classify,
                                                                                                self.output_dir, set_id,
                                                                                                subset, models_dir,
                                                                                                classification_dir),
                    shell=True)

    def cross_val(self):
        classification_dir = self.output_dir.replace('datasets', 'classifications')
        # all_files = glob.glob(classification_dir + "/*predictions*")
        _list = []
        for set_id in range(1, 31):
            _pair = []
            for subset in ['a', 'b']:
                # set_id, subset = file_.split('_')[-2:]
                _res_df = pd.read_csv(f'{classification_dir}/predictions_{set_id}_{subset}', header=None)
                _test_topics = np.array(self.folds_df[set_id][subset]['test']).astype(str)
                _res_df.insert(loc=0, column='qid', value=_test_topics)
                _res_df.set_index('qid', inplace=True)
                _ap_df = self.ap_obj.data_df.loc[_test_topics]['ap']
                _correlation = _res_df.corrwith(_ap_df)[0]
                _pair.append(_correlation)
            _list.append(np.mean(_pair))
        print('mean: {:.4f}'.format(np.mean(_list)))


def main(args):
    corpus = args.corpus
    predictor = args.predictor
    agg_func = args.aggregate
    uef = args.uef
    corr_measure = args.corr_measure

    y = LearningDataSets(predictor, corpus)
    y.generate_data_sets()
    y.run_svm()
    y.cross_val()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
