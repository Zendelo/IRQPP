import qpputils as qu
import toml
import os


class Config:
    config = toml.load('./config.toml')

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
