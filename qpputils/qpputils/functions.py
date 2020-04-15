import multiprocessing as mp
import os
from glob import glob
from subprocess import run

import pandas as pd


def ensure_file(file):
    """Ensure a single file exists, returns the full path of the file if True or throws an Assertion error if not"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file))
    assert os.path.isfile(file_path), "The file {} doesn't exist. Please create the file first".format(file)
    return file_path


def ensure_dir(file_path):
    """The function ensures the dir exists, if it doesn't it creates it and returns the path"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
    return directory


def empty_dir(dir_path, force=False):
    if force:
        run(f'rm -v {dir_path}/*', shell=True)
    else:
        files = os.listdir(dir_path)
        if len(files) and not mp.current_process().daemon:
            answer = input(
                f'The directory {dir_path} contains {len(files)} files, do you want to remove them?\n [yes\\No] ')
            if answer.lower() == 'yes':
                run(f'rm -v {dir_path}/*', shell=True)


def convert_vid_to_qid(df: pd.DataFrame):
    if df.index.name != 'qid' and df.index.name != 'topic':
        if 'qid' in df.columns:
            _df = df.set_index('qid')
        elif 'topic' in df.columns:
            _df = df.set_index('topic')
        else:
            assert False, "The DF doesn't has qid or topic"
    else:
        _df = df
    _df.rename(index=lambda x: f'{x.split("-")[0]}', inplace=True)
    return _df


def add_topic_to_qdf(qdf: pd.DataFrame):
    """This functions will add a topic column to the queries DF"""
    if 'topic' not in qdf.columns:
        if 'qid' in qdf.columns:
            qdf = qdf.assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
        else:
            qdf = qdf.reset_index().assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
    return qdf


def read_rm_prob_files(data_dir, number_of_docs, clipping='*'):
    """The function creates a DF from files, the probabilities are p(w|RM1) for all query words
    If a query term doesn't appear in the file, it's implies p(w|R)=0"""
    data_files = glob(f'{data_dir}/probabilities-{number_of_docs}+{clipping}')
    if len(data_files) < 1:
        data_files = glob(f'{data_dir}/probabilities-{number_of_docs}')
    _list = []
    for _file in data_files:
        _col = f'{_file.rsplit("/")[-1].rsplit("-")[-1]}'
        _df = pd.read_csv(_file, names=['qid', 'term', _col], sep=' ')
        _df = _df.astype({'qid': str}).set_index(['qid', 'term'])
        _list.append(_df)
    return pd.concat(_list, axis=1).fillna(0)


def set_environment_paths(base_path=None):
    base_path = base_path if base_path else os.path.dirname(os.path.abspath(__file__))
    results_dir = ensure_dir(f'{base_path}/QppUqvProj/Results')
    data_dir = ensure_dir(f'{base_path}/QppUqvProj/data')
    return results_dir, data_dir


def char_range(a, z):
    """Creates a generator that iterates the characters from `c1` to `c2`, inclusive."""
    # ord returns the ASCII value, chr returns the char of ASCII value
    for c in range(ord(a), ord(z) + 1):
        yield chr(c)
