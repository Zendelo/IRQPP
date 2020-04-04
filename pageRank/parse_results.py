import argparse
from glob import glob

import pandas as pd

import dataparser as dp
from Timer import Timer

PREDICTORS = ['clarity', 'nqc', 'wig', 'qf', 'uef/clarity', 'uef/nqc', 'uef/wig', 'uef/qf']
SIMILARITY_FUNCTIONS = ['Jac_coefficient', 'RBO_EXT_100', 'Top_10_Docs_overlap', 'RBO_FUSED_EXT_100']

parser = argparse.ArgumentParser(description='PageRank UQV Generator',
                                 usage='python3.7 pagerank.py -c CORPUS',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-c', '--corpus', default='ROBUST', type=str, help='corpus (index) to work with',
                    choices=['ROBUST', 'ClueWeb12B'])
parser.add_argument('-p', '--predictor', default=None, type=str, help='Choose the predictor to use',
                    choices=['clarity', 'wig', 'nqc', 'qf', 'all'])
parser.add_argument('--uef', help="use the uef version of the predictor", action="store_true")
parser.add_argument('--nocache', help="add this option to avoid loading from cache", action="store_false")


def read_results(results_path):
    for sim in SIMILARITY_FUNCTIONS:
        for predictor in PREDICTORS:
            _results_dir = f'{results_path}/{sim}/{predictor}/predictions'
            full_df = read_into_df(_results_dir)
            write_basic_predictions(full_df, _results_dir.replace('raw', 'title'), sim)
    return None


def read_into_df(results_dir):
    result_files = glob(f'{results_dir}/predictions-*')
    _list = []
    for res_file in result_files:
        df = dp.ResultsReader(res_file, 'predictions').data_df
        df_name = res_file.split('/')[-1]
        df.rename(columns={'score': df_name}, inplace=True)
        _list.append(df)
    res_df = pd.concat(_list, axis=1)
    return res_df


def write_basic_predictions(df: pd.DataFrame, output_dir, similarity: str, qgroup='title') -> None:
    """The function is used to save results in basic predictions format of a given queries set. e.g. 'qid': score"""
    output_dir = dp.ensure_dir(output_dir)
    for col in df.columns:
        file_name = f'{output_dir}/{col}'
        df[col].to_csv(file_name, sep=' ', header=False, index=True, float_format='%f')


def filter_top_queries(qdf: pd.DataFrame, apdb: dp.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        top_var = _apdf.loc[q_vars].idxmax()
        _list.append(top_var[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_title_queries(resutls_df: pd.DataFrame, full_obj: dp.QueriesTextParser, title_obj: dp.QueriesTextParser):
    # TODO: Check for bugs
    full_df = full_obj.queries_df
    _title_vid = full_df.loc[full_df['text'].isin(title_obj.queries_df['text'])]['qid']
    _df = resutls_df.loc[_title_vid]
    return _df


def convert_vid_to_qid(df: pd.DataFrame):
    df.rename(index=lambda x: f'{x.split("-")[0]}', inplace=True)
    return df


def main(args):
    corpus = args.corpus
    results_dir = dp.ensure_dir(f'~/QppUqvProj/Results/{corpus}/uqvPredictions/referenceLists/pageRank/raw')
    title_queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_title.txt')
    title_queries_obj = dp.QueriesTextParser(title_queries_file)
    full_queries_file = dp.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_UQV_full.txt')
    full_queries_obj = dp.QueriesTextParser(full_queries_file, 'uqv')

    for sim in SIMILARITY_FUNCTIONS:
        for predictor in PREDICTORS:
            _results_dir = f'{results_dir}/{sim}/{predictor}/predictions'
            full_df = read_into_df(_results_dir)
            title_vid_df = filter_title_queries(full_df, full_queries_obj, title_queries_obj)
            title_qid_df = convert_vid_to_qid(title_vid_df)
            write_basic_predictions(title_qid_df, _results_dir.replace('raw', 'title'), sim)


if __name__ == '__main__':
    args = parser.parse_args()
    overall_timer = Timer('Total runtime')
    main(args)
    overall_timer.stop()
