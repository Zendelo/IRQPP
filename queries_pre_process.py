import argparse

import pandas as pd

import dataparser as dt

# TODO: add logging and qrels file generation for UQV

parser = argparse.ArgumentParser(description='Features for UQV query variations Generator',
                                 usage='python3.6 features.py -q queries.txt -c CORPUS -r QL.res ',
                                 epilog='Unless --generate is given, will try loading the file')

parser.add_argument('-t', '--queries', default=None, metavar='queries.txt', help='path to UQV queries txt file')


def remove_duplicates(qdb: dt.QueriesTextParser):
    return qdb.queries_df.drop_duplicates('text')


def save_txt_queries(q_df: pd.DataFrame):
    q_df.to_csv('queries_new.txt', sep=":", header=False, index=False)


def main(args):
    queries_txt_file = args.queries

    if queries_txt_file is not None:
        qdb = dt.QueriesTextParser(queries_txt_file, 'uqv')
        queries_df = remove_duplicates(qdb)
        save_txt_queries(queries_df)
        query_xml = dt.QueriesXMLParser(queries_df)
        query_xml.print_queries_xml()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
