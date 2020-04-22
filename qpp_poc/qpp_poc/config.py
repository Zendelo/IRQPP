import os
from typing import NamedTuple, Tuple
import logging

import qpputils as qu
import toml


class Config:
    config = toml.load('/research/local/olz/IRQPP/qpp_poc/qpp_poc/config.toml')

    # Index dump paths
    env = config.get('environment')
    INDEX_DIR = qu.ensure_dir(env.get('index_dir'), create_if_not=False)
    TEXT_INV = qu.ensure_file(os.path.join(INDEX_DIR, 'text.inv'))
    DICT_TXT = qu.ensure_file(os.path.join(INDEX_DIR, 'dict_new.txt'))
    DOC_LENS = qu.ensure_file(os.path.join(INDEX_DIR, 'doc_lens.txt'))
    DOC_NAMES = qu.ensure_file(os.path.join(INDEX_DIR, 'doc_names.txt'))
    INDEX_GLOBALS = qu.ensure_file(os.path.join(INDEX_DIR, 'global.txt'))

    QUERIES_FILE = env.get('queries_file')

    # Defaults
    defaults = config.get('defaults')
    MU = defaults.get('mu')
    NUM_DOCS = defaults.get('max_result_size')

    def __init_logger(self, logger):
        if logger:
            return logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


# Special project types

# TODO: add a parser tp the types that can receive a str object and parse it to the relevant fields
class Posting(NamedTuple):
    doc_id: int
    tf: int


class TermPosting(NamedTuple):
    term: str
    cf: int
    df: int
    posting_list: Tuple[Posting]


class TermRecord(NamedTuple):
    term: str
    id: int
    cf: int
    df: int


class TermFrequency(NamedTuple):
    term: str
    doc_id: int
    tf: int


class ResultPair(NamedTuple):
    doc_id: str
    score: float
