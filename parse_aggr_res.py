import pandas as pd
import csv
from collections import defaultdict
import argparse
from functools import reduce

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


def print_latex(res_dict):
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
    print(x.to_latex())


def main(args: parser):
    results_file = args.results
    ap_file = args.ap
    if ap_file:
        res_dict = read_ap_file(results_file)
        print_latex(res_dict)

    else:
        res_dic = read_file(results_file)
        for p in res_dic:
            print(p)
            print_latex(res_dic[p])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
