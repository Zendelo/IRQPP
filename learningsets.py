import argparse

import numpy as np

from Timer.timer import Timer
from crossval import CrossValidation
from dataparser import ResultsReader
from features import features_loader

parser = argparse.ArgumentParser(description='LTR (SVMRank) data sets Generator',
                                 usage='python3.6 learningsets.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', type=str, help='corpus to work with', choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-r', '--results', default=None, type=str, help='QPP results dir for the relevant UQV topics')
parser.add_argument('-a', '--ap', default=None, type=str, help='AP results file as "ground truth"')
parser.add_argument('--folds', default=None, type=str, help='folds JSON file to load')
parser.add_argument('-p', '--parameters', default=None, type=str, help='full CV results JSON file to load')
parser.add_argument('-f', '--features', default=None, type=str, help='features JSON file to load')
parser.add_argument('--corr_measure', default='pearson', type=str, choices=['ROBUST', 'ClueWeb12B'],
                    help='features JSON file to load')
parser.add_argument('--generate', help="generate new predictions", action="store_true")


class LearningDataSets:
    def __init__(self, folds, parameters, features, corpus, results_dir, ap_file, corr_measure='pearson'):
        # self.corpus = corpus
        # self.results_dir = results_dir
        self.ap_obj = ResultsReader(ap_file, 'ap')
        self.cv = CrossValidation(file_to_load=folds, predictions_dir=results_dir, test=corr_measure)
        self.folds_df = self.cv.data_sets_map
        self.parameters_df = self.cv.read_eval_results(parameters)
        self.results_df = self.cv.full_set
        self.feature_names = ['Jac_coefficient', 'Top_10_Docs_overlap', 'RBO_EXT_100', 'RBO_EXT_1000',
                              'RBO_FUSED_EXT_100', 'RBO_FUSED_EXT_1000']
        features_df = features_loader(features, corpus)
        self.features_df = features_df.filter(items=self.feature_names)

        # print(f'features df \n {self.features_df}')
        # print(f'folds df \n {self.folds_df}')
        # print(f'patameters df \n {self.parameters_df}')
        # print(f'results df \n {self.results_df}')
        # print(f'ap df \n{self.ap_obj.data_df}')

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
        _df = df.to_string(columns=['ap', 'qid', ] + self.feature_names, index=False,
                           index_names=False, header=False,
                           formatters={'Jac_coefficient': '1:{}'.format, 'Top_10_Docs_overlap': '2:{}'.format,
                                       'RBO_EXT_100': '3:{}'.format,
                                       'RBO_EXT_1000': '4:{}'.format,
                                       'RBO_FUSED_EXT_100': '5:{}'.format,
                                       'RBO_FUSED_EXT_1000': '6:{}'.format})
        return _df

    @staticmethod
    def write_str_to_file(string, file_name):
        with open(file_name, "w") as text_file:
            print(string, file=text_file)


def main(args):
    corpus = args.corpus
    parameters = args.parameters
    folds = args.folds
    features = args.features
    qpp_results = args.results
    map_file = args.ap

    y = LearningDataSets(folds, parameters, features, corpus, qpp_results, map_file)
    y.generate_data_sets()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
