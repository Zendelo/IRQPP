import os

import numpy as np
import pandas as pd
from Timer import Timer, timer
import argparse
from inspect import signature

try:
    import dataparser as dp
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    script_dir = sys.path[0]
    # Adding the parent directory to the path
    sys.path.append(str(Path(script_dir).parent))
    import dataparser as dp
from crossval import InterTopicCrossValidation
from queries_pre_process import add_topic_to_qdf

PREDICTORS = ['clarity', 'wig', 'nqc', 'smv', 'rsd', 'qf', 'uef/clarity', 'uef/wig', 'uef/nqc', 'uef/smv', 'uef/qf']
SIMILARITY_MEASURES = ['Jac_coefficient', 'RBO_EXT_100', 'Top_10_Docs_overlap', 'RBO_FUSED_EXT_100']

parser = argparse.ArgumentParser(description='PageRank UQV Evaluation', usage='python3.7 pr_eval.py -c CORPUS')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])


def calc_best_worst(full_df: pd.DataFrame, ap_df: pd.DataFrame, metric_direction):
    bw_param = max if metric_direction == 'best' else min
    _ap_vars = ap_df.loc[ap_df.groupby('topic')['ap'].transform(bw_param) == ap_df['ap']].set_index('topic')
    _results = []
    for col in full_df.set_index(['topic', 'qid']).columns:
        pr_df = full_df.loc[:, ['topic', 'qid', col]]
        _result = {}
        for topic, _df in pr_df.groupby('topic'):
            _var_ap = _ap_vars.loc[topic].qid
            if type(_var_ap) is str:
                _pr_val = _df.loc[_df['qid'] == _var_ap, col].values[0]
            else:
                _pr_val = np.mean(_df.loc[_df['qid'].isin(_var_ap), col].values)
            if metric_direction == 'best':
                _var_score = np.count_nonzero(_df[col] < _pr_val) / len(_df)
            else:
                _var_score = np.count_nonzero(_df[col] > _pr_val) / len(_df)
            _result[topic] = {col: _var_score}
        _results.append(pd.DataFrame.from_dict(_result, orient='index'))
    df = pd.concat(_results, axis=1)
    return df


def set_basic_paths(corpus):
    res_dir, data_dir = dp.set_environment_paths()
    cv_folds = dp.ensure_file(f'{res_dir}/{corpus}/test/2_folds_30_repetitions.json')
    ap_file = dp.ensure_file(f'{res_dir}/{corpus}/test/raw/QLmap1000')
    pkl_dir = dp.ensure_dir(f'{res_dir}/{corpus}/test/pageRank/pkl_files')
    return {'res_dir': res_dir, 'data_dir': data_dir, 'pkl_dir': pkl_dir, 'cv_folds': cv_folds, 'ap_file': ap_file}


def init_eval(corpus, similarity, predictor):
    pth_dict = set_basic_paths(corpus)
    predictor_pkl_dir = dp.ensure_dir(f"{pth_dict['pkl_dir']}/{predictor}")
    predictions_dir = dp.ensure_dir(
        f'{pth_dict["res_dir"]}/{corpus}/uqvPredictions/referenceLists/pageRank/raw/{similarity}/{predictor}/predictions')
    ap_obj = dp.ResultsReader(pth_dict['ap_file'], 'ap')
    ap_df = add_topic_to_qdf(ap_obj.data_df)
    cv_obj = InterTopicCrossValidation(predictions_dir=predictions_dir, folds_map_file=pth_dict['cv_folds'])
    full_results_df = add_topic_to_qdf(cv_obj.full_set)
    return {'predictor_pkl_dir': predictor_pkl_dir, 'ap_obj': ap_obj, 'ap_df': ap_df,
            'full_results_df': full_results_df, 'cv_obj': cv_obj}


@timer
def best_worst_metric(corpus, similarity, predictor, metric, load=False):
    assert metric == 'best' or metric == 'worst', f'The function expects a known metric. {metric} was passed'
    pkl_dir, ap_obj, ap_df, full_results_df, cv_obj = init_eval(corpus, similarity, predictor).values()
    _file = f'{pkl_dir}/{similarity}_{metric}_results.pkl'
    if load:
        _df = load_exec(_file, calc_best_worst, (full_results_df, ap_df, metric))
    else:
        _df = calc_best_worst(full_results_df, ap_df, metric)
        _df.to_pickle(_file)
    return calc_s(cv_obj, _df)


def calc_s(cv_obj: InterTopicCrossValidation, full_scores_df: pd.DataFrame):
    if hasattr(cv_obj, 'corr_df'):
        cv_obj.__delattr__('corr_df')
    cv_obj.full_set = full_scores_df
    score = cv_obj.calc_test_results()
    return float(score)


def load_exec(file_to_load, function_to_exec, args=None):
    """
    The function tries to load a pandas DataFrame from Pickle file, if the file doesn't exist it will use the function
    that was passed as an argument to generate a new DataFrame.
    The assumptions are that the pickle file is a pandas DataFrame (if exists) and
    the passed function returns a pandas DataFrame
    :param file_to_load: path to the file that should loaded
    :param function_to_exec: the function that will be executed in case the file doesn't exist
    :param args: single or list of arguments to pass to the exec function
    :return: pd.DataFrame
    """
    try:
        _df = pd.read_pickle(dp.ensure_file(file_to_load))
    except AssertionError:
        # Checking the signature of the function
        sig = signature(function_to_exec)
        if bool(sig.parameters):
            if len(sig.parameters) > 1:
                _df = function_to_exec(*args)
            else:
                _df = function_to_exec(args)
        else:
            _df = function_to_exec()
        _df.to_pickle(file_to_load)
    return _df


def minmax_ap_metric(corpus, similarity, predictor, minmax):
    pkl_dir, ap_obj, raw_ap_df, full_pr_df, cv_obj = init_eval(corpus, similarity, predictor).values()
    _list = []
    for col in full_pr_df.set_index(['topic', 'qid']).columns:
        grpby = full_pr_df.loc[:, ['topic', 'qid', col]].set_index('qid').groupby('topic')[col]
        _qids = grpby.idxmax() if minmax == 'max' else grpby.idxmin()
        _df = raw_ap_df.loc[raw_ap_df.qid.isin(_qids)].set_index('topic')['ap']
        _list.append(_df.rename(col))
    full_ap_df = pd.concat(_list, axis=1)
    return calc_s(cv_obj, full_ap_df) if minmax == 'max' else -calc_s(cv_obj, -full_ap_df)


def interactive_parameters():
    # TODO: cover all options for the interactive choice with retries instead of fixed values
    corpus = input('What corpus should be used for evaluation?\n')
    while corpus != 'ROBUST' and corpus != 'ClueWeb12B':
        print(f'Unknown corpus: {corpus}\ntry ROBUST or ClueWeb12B instead')
        corpus = input('What corpus should be used for evaluation?\n')
    predictor = input('What predictor should be used for evaluation?\n')
    while predictor not in PREDICTORS:
        print(f'Unknown predictor: {predictor}\n try one of the available predictors instead, e.g.\n{PREDICTORS}')
        predictor = input('What predictor should be used for evaluation?\n')
    similarity = input('What similarity should be used for evaluation?\n')
    while similarity not in SIMILARITY_MEASURES:
        print(
            f'Unknown similarity: {similarity}\n try one of the available similarities instead, e.g.\n{SIMILARITY_MEASURES}')
        similarity = input('What similarity should be used for evaluation?\n')
    return corpus, similarity, predictor


def run_all(metric_func, args):
    # TODO: change it to parallel runs
    results = []
    for corpus in {'ROBUST', 'ClueWeb12B'}:
        for similarity in SIMILARITY_MEASURES:
            for predictor in PREDICTORS:
                results.append(metric_func(corpus, similarity, predictor, *args))
    return results


def main(args, load_results=True, interact=True):
    corpus = args.corpus
    pths_dict = set_basic_paths(corpus)
    if interact:
        print('\n\n\n------------!!!!!!!---------- Interactive Mode ------------!!!!!!!----------\n\n\n')
        while True:
            corpus, similarity, predictor = interactive_parameters()
            max_ap_score = minmax_ap_metric(corpus, similarity, predictor, 'max')
            min_ap_score = minmax_ap_metric(corpus, similarity, predictor, 'min')
            print(f'The MAP score using {predictor} for max PR queries: {max_ap_score}\n'
                  f'The MAP score using {predictor} for min PR queries: {min_ap_score}')
            best_score = best_worst_metric(corpus, similarity, predictor, metric='best', load=True)
            worst_score = best_worst_metric(corpus, similarity, predictor, metric='worst', load=True)
            print(f'Predicting using {predictor} best var: {best_score}\n'
                  f'Predicting using {predictor} worst var {worst_score}')
    elif load_results:
        res_df = load_exec(os.path.join(pths_dict['pkl_dir'], f'{corpus}_PageRank_results_table.pkl'), fun, corpus)
        print(
            res_df.to_latex(header=True, multirow=True, multicolumn=False, index=True, escape=False, index_names=True))
    else:
        results_df = fun(corpus)
        results_df.to_pickle(os.path.join(pths_dict['pkl_dir'], f'{corpus}_PageRank_results_table.pkl'))


def fun(corpus):
    res_dir, data_dir = dp.set_environment_paths()
    raw_ap_df = dp.add_topic_to_qdf(
        dp.ResultsReader(dp.ensure_file(f'{res_dir}/{corpus}/test/raw/QLmap1000'), 'ap').data_df)
    grp_obj = raw_ap_df.groupby('topic')['ap']
    avg_ap_df = grp_obj.mean()
    max_ap_df = grp_obj.max()
    min_ap_df = grp_obj.min()
    print(f'Average real MAP: {avg_ap_df.mean():.4f}')
    print(f'Maximum real MAP: {max_ap_df.mean():.4f}')
    print(f'Min real MAP: {min_ap_df.mean():.4f}')
    result = {}
    for similarity in SIMILARITY_MEASURES:
        print(f'Similarity {similarity}:')
        for predictor in PREDICTORS:
            max_ap_score = minmax_ap_metric(corpus, similarity, predictor, 'max')
            min_ap_score = minmax_ap_metric(corpus, similarity, predictor, 'min')
            print(f'The MAP score using {predictor} for max PR queries: {max_ap_score}\n'
                  f'The MAP score using {predictor} for min PR queries: {min_ap_score}')
            best_score = best_worst_metric(corpus, similarity, predictor, metric='best', load=True)
            worst_score = best_worst_metric(corpus, similarity, predictor, metric='worst', load=True)
            print(f'Predicting using {predictor} best var: {best_score}\n'
                  f'Predicting using {predictor} worst var {worst_score}')
            result[similarity, predictor] = {'Max MAP': max_ap_score, 'Min MAP': min_ap_score,
                                             'Best Score': best_score,
                                             'Worst Score': worst_score}
    df = pd.DataFrame.from_dict(result, orient='index')
    print(df.to_latex(header=True, multirow=True, multicolumn=False, index=True, escape=False, index_names=True))
    return df


if __name__ == '__main__':
    args = parser.parse_args()
    timer = Timer('Total time')
    main(args)
    timer.stop()
