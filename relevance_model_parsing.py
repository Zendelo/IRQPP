import pandas as pd
import numpy as np
import dataparser as dp
from glob import glob


def separate_tables(raw_file):
    raw_df = pd.read_table(raw_file, names=['qid', 'term', 'probability', 'clipping'], sep=' ')
    full_rm_df = raw_df.loc[raw_df['clipping'].isna()].drop('clipping', axis=1)
    clipped_df = raw_df.loc[~raw_df['clipping'].isna()].drop('clipping', axis=1)

    raw_file = raw_file.rsplit('/', 1)
    _file_name = raw_file[-1]
    _dir = dp.ensure_dir(raw_file[0].replace('raw_data', 'data'))
    fullrm_file = f'{_file_name}+c0'
    clipped_file = f'{_file_name}+c100'

    clipped_df.to_csv(f'{_dir}/{clipped_file}', sep=" ", header=False, index=False, float_format='%f')
    full_rm_df.to_csv(f'{_dir}/{fullrm_file}', sep=" ", header=False, index=False, float_format='%f')


def main():
    # corpus = 'ROBUST'
    corpus = 'ClueWeb12B'

    # raw_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/raw/rsd/raw_data')
    raw_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/basicPredictions/title/rsd/raw_data')

    raw_files = glob(f'{raw_dir}/probabilities-*')
    for raw_file in raw_files:
        separate_tables(raw_file)


if __name__ == '__main__':
    main()
