import argparse

import pandas as pd

import dataparser as dt

# TODO: add logging and qrels file generation for UQV
# TODO: Change the remove duplicates function to remove only inter topic duplicates

parser = argparse.ArgumentParser(description='Script for query files pre-processing',
                                 epilog='Use this script with Caution')

parser.add_argument('-t', '--queries', default=None, metavar='queries.txt', help='path to UQV queries txt file')
parser.add_argument('--remove', default=None, metavar='queries.txt',
                    help='path to queries txt file that will be removed from the final file NON UQV ONLY')


def add_original_queries():
    pass


def remove_duplicates(qdb: dt.QueriesTextParser):
    _list = []
    for topic, q_vars in qdb.query_vars.items():
        _list.append(qdb.queries_df.loc[qdb.queries_df['qid'].isin(q_vars)].drop_duplicates('text'))
    return pd.concat(_list)


def alternate_remove_duplicates(qdb: dt.QueriesTextParser):
    """Different commands, same result"""
    _dup_list = []
    for topic, q_vars in qdb.query_vars.items():
        _dup_list.extend(qdb.queries_df.loc[qdb.queries_df['qid'].isin(q_vars)].duplicated('text'))
    return qdb.queries_df[~qdb.queries_df['qid'].isin(qdb.queries_df.loc[_dup_list]['qid'])]


def remove_q1_from_q2(qdf_1: pd.DataFrame, qdf_2: pd.DataFrame):
    """This function will remove from qdf_2 the queries that exist in qdf_1 """
    return qdf_2.loc[~qdf_2['text'].isin(qdf_1['text'])]


def save_txt_queries(q_df: pd.DataFrame):
    q_df.to_csv('queries_new.txt', sep=":", header=False, index=False)


def main(args):
    queries_txt_file = args.queries
    queries_to_remove = args.remove

    if queries_txt_file is not None:
        qdb = dt.QueriesTextParser(queries_txt_file, 'uqv')
        queries_df = remove_duplicates(qdb)
        if queries_to_remove:
            qdb_rm = dt.QueriesTextParser(queries_to_remove)
            queries_df = remove_q1_from_q2(qdb_rm.queries_df, queries_df)
        save_txt_queries(queries_df)
        query_xml = dt.QueriesXMLParser(queries_df)
        query_xml.print_queries_xml()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
