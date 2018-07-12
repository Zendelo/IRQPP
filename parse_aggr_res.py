import pandas as pd
import csv
from collections import defaultdict
import argparse
from functools import reduce
import numpy as np

parser = argparse.ArgumentParser(description='UQV aggregation script',
                                 usage='Create new UQV scores',
                                 epilog='ROBUST version')

parser.add_argument('results', metavar='Aggregate_Results_File', default=None, help='path to agg res file')
parser.add_argument('--ap', default=False, action='store_true')


def read_file(file):
    temp_dict = defaultdict(list)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            else:
                row = row[0].split()
                # print(row)
                if row[0].lower() == 'predictor':
                    predictor = row[1]
                    continue
                elif row[0].lower().startswith('ap-'):
                    ap = row[0].split('-')[1]
                    pred = row[1].split('-')[1]
                    continue
                elif row[0].lower().startswith('mean'):
                    mean = row[2]
                    continue
                elif row[0].lower().startswith('var'):
                    var = row[2]
                    continue
                elif row[0].lower().startswith('stand'):
                    std = row[3]
                # temp_dict[predictor].append([ap, pred, mean, var, std])
                temp_dict[predictor].append({'ap': ap, 'predictor': pred, 'mean': mean, 'var': var, 'std': std})
    f.close()
    return temp_dict


def read_big_file(file):
    temp_dict = defaultdict(list)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            else:
                row = row[0].split()
                # print(row)
                if row[0].lower() == 'predictor':
                    predictor = row[1]
                    continue
                elif row[0].lower().startswith('ap-'):
                    ap = row[0].split('-')[1]
                    pred = row[1].split('-')[1]
                    continue
                elif row[0].lower().startswith('mean'):
                    mean = row[2]
                    continue
                elif row[0].lower().startswith('var'):
                    var = row[2]
                    continue
                elif row[0].lower().startswith('stand'):
                    std = row[3]
                # temp_dict[predictor].append([ap, pred, mean, var, std])
                temp_dict[predictor].append({'ap': ap, 'predictor': pred, 'mean': mean, 'var': var, 'std': std})
    f.close()
    return temp_dict


def read_ap_file(file):
    temp = list()
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            else:
                row = row[0].split()
                # print(row)
                if row[0].lower().startswith('ap-'):
                    ap = row[0].split('-')[1]
                    pred = row[1].split('-')[1]
                    continue
                else:
                    mean = row[0]
                temp.append({'ap': ap, 'predictor': pred, 'mean': mean})
    f.close()
    return temp


def print_latex(res_dict, p=None):
    df = pd.DataFrame.from_records(res_dict, columns=['ap', 'predictor', 'mean'])
    df = df.set_index(['ap'])
    df = df.sort_index()
    df = df.sort_values(['predictor'])
    avg_df = df.loc['avg']
    max_df = df.loc['max']
    med_df = df.loc['med']
    min_df = df.loc['min']
    std_df = df.loc['std']
    x = reduce(lambda left, right: pd.merge(left, right, on='predictor'), [avg_df, max_df, med_df, min_df, std_df])
    x = x.set_index('predictor')
    x.columns = ['avg', 'max', 'med', 'min', 'std']
    # print(x.to_latex())
    return x.to_dict(orient='dict')


def bold_max_col(res_dict):
    """Boldface the maximum value for each AP aggregate"""
    max_cols = defaultdict()
    for agg in aggregates_list:
        max_pred = defaultdict()
        max_val = 0
        for pred in predictors_list:
            max_agg = max(res_dict[pred][agg], key=res_dict[pred][agg].get)
            _max_val = float(res_dict[pred][agg][max_agg])
            if _max_val > max_val:
                max_val = _max_val
                max_pred = {'aggregate': max_agg, 'predictor': pred, 'corr': max_val}
        res_dict[max_pred['predictor']][agg][max_pred['aggregate']] = '\\textbf{{{}}}'.format(max_pred['corr'])


def print_big_latex(res_dict):
    # test = list()
    test = defaultdict()
    avg_dict = defaultdict(list)
    max_dict = defaultdict(list)
    med_dict = defaultdict(list)
    min_dict = defaultdict(list)
    std_dict = defaultdict(list)
    # pred_list = ['clarity', 'wig', 'nqc', 'qf']
    agg_list = aggregates_list
    for pred in agg_list:
        for ap in agg_list:
            test[ap, pred] = {'clarity': '${}$'.format(res_dict['clarity'][ap][pred]),
                              'wig': '${}$'.format(res_dict['wig'][ap][pred]),
                              'nqc': '${}$'.format(res_dict['nqc'][ap][pred]),
                              'qf': '${}$'.format(res_dict['qf'][ap][pred])}
    # print(pd.DataFrame.from_dict(test['avg', 'avg'], orient='index'))
    col = 0
    print('\\begin{tabular}{lccccc}')
    print('\\toprule')
    print('{AP} &     avg &     max &     med &     min &     std \\\\')
    print('predictor &         &         &         &         &         \\\\')
    print('\\midrule')

    for pred in agg_list:
        for ap in agg_list:
            # print('pred {}'.format(pred), 'ap {}'.format(ap))
            x = pd.DataFrame.from_dict(test[ap, pred], orient='index')

            if col % 5 == 0:
                print('{}   &'.format(pred))
                latex = x.to_latex(header=False, multirow=True, index=True, escape=False)
            else:
                latex = x.to_latex(header=False, multirow=True, index=False, escape=False)
            latex = latex.replace('\\toprule', '')
            latex = latex.replace('\\bottomrule', '')
            print(latex)
            col += 1

            if col % 5 == 0:
                print('\\\\')
                if pred != 'std':
                    print('\\hline')
            else:
                print('&')

    print('\\bottomrule')
    print('\\end{tabular}')


def main(args: parser):
    results_file = args.results
    ap_file = args.ap
    if ap_file:
        res_dict = read_ap_file(results_file)
        print_latex(res_dict)

    else:
        _dict = defaultdict()
        res_dic = read_file(results_file)
        for p in res_dic:
            _dict[p] = print_latex(res_dic[p], p)
        bold_max_col(_dict)
        print_big_latex(_dict)


if __name__ == '__main__':
    predictors_list = ['clarity', 'wig', 'nqc', 'qf']
    aggregates_list = ['avg', 'max', 'med', 'min', 'std']
    args = parser.parse_args()
    main(args)
