#! /usr/bin/env python

import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='Correlation Evaluation script',
                                 usage='specify 2 files to evaluate correlation',
                                 epilog='The files must have 2 columns, first for index and second for the values')

parser.add_argument('first', metavar='first_file_path', help='first evaluation file')
parser.add_argument('second', metavar='second_file_path', help='second evaluation file')
parser.add_argument('-t', '--test', default='pearson', type=str,
                    help='default test type is pearson', choices=['pearson', 'spearman', 'kendall'], )


def print_cor(df, type):
    print('The {} correlation is: {:0.4f}'.format(type, df['x'].corr(df['y'], method=type)))


def main(args):
    first_file = args.first
    second_file = args.second

    first_df = pd.read_table(first_file, delim_whitespace=True, header=None, index_col=0, names=['x'])
    second_df = pd.read_table(second_file, delim_whitespace=True, header=None, index_col=0, names=['y'])
    merged_df = pd.merge(first_df, second_df, left_index=True, right_index=True)
    print_cor(merged_df, args.test)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
