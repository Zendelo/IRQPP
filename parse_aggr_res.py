import pandas as pd
import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='UQV aggregation script',
                                 usage='Create new UQV scores',
                                 epilog='ROBUST version')

parser.add_argument('results', metavar='Aggregate_Results_File', default=None, help='path to agg res file')


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


def main(args: parser):
    results_file = args.results
    res_dic = read_file(results_file)
    for p in res_dic:
        print(p)
        print(pd.DataFrame.from_records(res_dic[p], columns=['ap', 'predictor', 'mean', 'std']).to_latex(index=False))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
