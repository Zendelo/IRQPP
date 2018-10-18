import pandas as pd
import numpy as np
import os
import dataparser as dp
from crossval import CrossValidation
from features import features_loader
import argparse
from Timer.timer import Timer
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Query Prediction Using Reference lists',
                                 usage='python3.6 qpp_ref.py -c CORPUS ... <parameter files>',
                                 epilog='Unless --generate is given, will try loading the files')

parser.add_argument('-c', '--corpus', type=str, default=None, help='choose the corpus',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-p', '--predictor', type=str, default=None, help='choose the predictor',
                    choices=['clarity', 'wig', 'nqc', 'qf'])
parser.add_argument('-f', '--function', type=str, default=None, help='choose the similarity function',
                    choices=['uni', 'rbo', 'jac', 'rbof', 'overlap'])

parser.add_argument('--generate', help='Add this to generate new results, make sure to RM the previous results',
                    action="store_true")
# parser.add_argument('--fine',
#                     help='Add this to generate new results, with fine tuning of parameters (may cause overfitting)',
#                     action="store_true")

LAMBDA = np.linspace(start=0, stop=1, num=11)
MARKERS = ['-^', '-v', '-D', '-x', '-h', '-H', 'p-', 's-', '--v', '--1', '--2', '--D', '--x', '--h', '--H', '^-.',
           '-.v', '1-.', '2-.', '-.D', '-.x', '-.h', '-.H', '3-.', '4-.', 's-.', 'p-.', '+-.', '*-.']


class GenerateResults:
    def __init__(self, predictor, corpus, ref_feature, corr_measure='pearson'):
        self.__set_paths(corpus, predictor, ref_feature)
        self.corr_measure = corr_measure

    @classmethod
    def __set_paths(cls, corpus, predictor, ref_feature):
        """This method sets the default paths of the files and the working directories, it assumes the standard naming
         convention of the project"""
        cls.predictor = predictor
        cls.ref_feature = ref_feature
        cls.corpus = corpus
        _base_dir = f'~/QppUqvProj/Results/{corpus}/uqvPredictions/'
        _base_dir = os.path.normpath(os.path.expanduser(_base_dir))
        cls.results_dir = f'{_base_dir}/referenceLists/{ref_feature}/{predictor}/predictions/'
        _ap_file = f'~/QppUqvProj/Results/{corpus}/test/basic/QLmap1000'
        cls.true_ap_file = os.path.normpath(os.path.expanduser(_ap_file))

    def generate_full_set_results(self):
        _dict = defaultdict(list)
        plots_dict = {}
        prediction_files = glob(self.results_dir + 'predictions-*')
        for result in prediction_files:
            _params = result.split('-')[-1].split('lambda')
            _predictor_param = _params[0].strip('+')
            _lambda_val = _params[-1].strip('+')
            _temp_df = dp.ResultsReader(result, 'predictions').data_df
            _df = _temp_df.rename(columns={"score": f'lambda-{_lambda_val}'})
            _dict[f'{self.predictor}-{_predictor_param}'].append(_df)
        for pred, _list in _dict.items():
            ap_df = dp.ResultsReader(self.true_ap_file, 'ap').data_df
            _list.append(ap_df)
            _df = pd.concat(_list, axis=1)
            _df = _df.reindex_axis(sorted(_df.columns), axis=1)
            plots_dict[pred] = _df.corr()['ap'].drop('ap')
        plots_df = pd.DataFrame(plots_dict)
        plots_df.index.name = 'lambda'
        plots_df.rename(index=lambda x: f'{x.lstrip("lambda-")}', inplace=True)
        plots_df.rename(index=float, inplace=True)
        # plots_df.reindex_axis(sorted(plots_df.columns), axis=0)
        plots_df.plot(
            title=f'{self.corr_measure.title()} correlation {self.ref_feature} similarity on {self.corpus}',
            grid=True, fontsize=10, style=MARKERS, markersize=10)
        plots_df.to_csv(f'{self.predictor}_{self.corr_measure}_AP_fullset_{self.ref_feature}')
        plt.ylabel(f'{self.corr_measure}')
        plt.xlabel('Lambda values')
        plt.show()


def main(args):
    corpus = args.corpus
    predictor = args.predictor
    sim_function = args.function
    # generate = args.generate

    # # Debugging
    # corpus = 'ROBUST'
    # predictor = 'wig'
    # sim_function = 'rbo'

    res_gen = GenerateResults(predictor, corpus, sim_function)
    res_gen.generate_full_set_results()


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
