import pandas as pd
import csv
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='UQV aggregation script',
                                 usage='Create new UQV scores',
                                 epilog='ROBUST version')

parser.add_argument('results', metavar='Aggregate_Results_File', default=None, help='path to agg res file')


def read_file(file):
    df = pd.read_table(file, delim_whitespace=True, index_col=0, names=['score'])
    # print(df.head())
    print('Mean : {:0.4f}'.format(float(df.mean(axis=0))))
    print('Var : {:0.4f}'.format(float(df.var(0))))
    print('STD : {:0.4f}'.format(float(df.std(0))))


def main(args: parser):
    results_file = args.results
    read_file(results_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
