import argparse
from statistics import median_high, median_low
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import dataparser as dt

# TODO: add logging and qrels file generation for UQV

QUERY_GROUPS = {'top': 'MaxAP', 'low': 'MinAP', 'medh': 'MedHiAP', 'medl': 'MedLoAP'}
QUANTILES = {'med': 'Med', 'top': 'Top', 'low': 'Low'}

parser = argparse.ArgumentParser(description='Script for query files pre-processing',
                                 epilog='Use this script with Caution')

parser.add_argument('-t', '--queries', default=None, metavar='queries.txt', help='path to UQV queries txt file')
parser.add_argument('--remove', default=None, metavar='queries.txt',
                    help='path to queries txt file that will be removed from the final file NON UQV ONLY')
parser.add_argument('--group', default='title', choices=['low', 'top', 'medh', 'medl'],
                    help='Return only the <> performing queries of each topic')
parser.add_argument('--quant', default=None, choices=['low', 'top', 'med'],
                    help='Return a quantile of the variants for each topic')
parser.add_argument('--ap', default=None, metavar='QLmap1000', help='path to queries AP results file')
parser.add_argument('--stats', action='store_true', help='Print statistics')


def add_original_queries(uqv_obj: dt.QueriesTextParser):
    """Don't use this function ! not tested"""
    original_obj = dt.QueriesTextParser('QppUqvProj/data/ROBUST/queries.txt')
    uqv_df = uqv_obj.queries_df.set_index('qid')
    original_df = original_obj.queries_df.set_index('qid')
    for topic, vars in uqv_obj.query_vars.items():
        uqv_df.loc[vars, 'topic'] = topic
    missing_list = []
    for topic, topic_df in uqv_df.groupby('topic'):
        if original_df.loc[original_df['text'].isin(topic_df['text'])].empty:
            missing_list.append(topic)
    missing_df = pd.DataFrame({'qid': '341-9-1', 'text': original_obj.queries_dict['341'], 'topic': '341'}, index=[0])
    uqv_df = uqv_df.append(missing_df.set_index('qid'))
    return uqv_df.sort_index().drop(columns='topic').reset_index()


def convert_vid_to_qid(df: pd.DataFrame):
    _df = df.set_index('qid')
    _df.rename(index=lambda x: f'{x.split("-")[0]}', inplace=True)
    return _df.reset_index()


def filter_quant_variants(qdf: pd.DataFrame, apdb: dt.ResultsReader, q):
    """This function returns a df with QID: TEXT of the queries inside a quantile"""
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        if 0 in q:
            # For the low quantile, 0 AP variants are removed
            _df = _df[_df['ap'] > 0]
        q_vals = _df.quantile(q=q)
        _qvars = _df.loc[(_df['ap'] >= q_vals['ap'].min()) & (_df['ap'] <= q_vals['ap'].max())]
        _list.extend(_qvars.index.tolist())
    _res_df = qdf.loc[qdf['qid'].isin(_list)]
    return _res_df


def filter_top_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        top_var = _apdf.loc[q_vars].idxmax()
        _list.append(top_var[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_low_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        # remove 0 ap variants
        _df = _df[_df['ap'] > 0]
        low_var = _df.idxmin()
        _list.append(low_var[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_medh_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        _med = median_high(_df['ap'])
        med_var = _df.loc[_df['ap'] == _med]
        _list.append(med_var.index[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


def filter_medl_queries(qdf: pd.DataFrame, apdb: dt.ResultsReader):
    _apdf = apdb.data_df
    _list = []
    for topic, q_vars in apdb.query_vars.items():
        _df = _apdf.loc[q_vars]
        _med = median_low(_df['ap'])
        med_var = _df.loc[_df['ap'] == _med]
        _list.append(med_var.index[0])
    _df = qdf.loc[qdf['qid'].isin(_list)]
    return _df


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


def remove_q1_from_q2(rm_df: pd.DataFrame, qdb: dt.QueriesTextParser):
    """This function will remove from queries_df in qdb the queries that exist in rm_df """
    _dup_list = []
    full_df = qdb.queries_df.set_index('qid')
    queries_to_remove = convert_vid_to_qid(rm_df).set_index('qid').to_dict(orient='index')
    for topic, q_vars in qdb.query_vars.items():
        # _dup_list.extend(full_df.loc[full_df['text'] == query_text]['qid'])
        topic_df = full_df.loc[q_vars]
        _dup_list.extend(topic_df.loc[topic_df['text'] == queries_to_remove[topic]['text']].index.tolist())
    return full_df.drop(index=_dup_list).reset_index()


def write_queries_to_files(q_df: pd.DataFrame, corpus, queries_group='title', quantile=None, remove=None):
    if quantile:
        file_name = f'queries_{corpus}_UQV_{quantile}_variants'
    elif remove:
        title = input('What queries were removed? \n')
        file_name = f'queries_{corpus}_UQV_wo_{title}'
    else:
        file_name = f'queries_{corpus}_{queries_group}'

    q_df.to_csv(f'{file_name}.txt', sep=":", header=False, index=False)
    query_xml = dt.QueriesXMLWriter(q_df)
    query_xml.print_queries_xml_file(f'{file_name}.xml')


def add_format(s):
    s = '${:.4f}$'.format(s)
    return s


def plot_robust_histograms(quant_variants_dict):
    for quant, vars_df in quant_variants_dict.items():
        if quant == 'all':
            bins = np.arange(4, 60) - 0.5
            xticks = np.arange(4, 60)
            yticks = np.arange(0, 80, 5)
        else:
            bins = np.arange(20) - 0.5
            xticks = np.arange(20)
            yticks = np.arange(0, 115, 5)
        vars_df.groupby('topic')['text'].count().plot(title=f'Number of vars in {quant} quantile ROBUST', kind='hist',
                                                      bins=bins)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylabel('Number of topics')
        plt.xlabel('Number of Variants per topic')
        # plt.grid(True)
        plt.show()


def plot_cw_histograms(quant_variants_dict):
    for quant, vars_df in quant_variants_dict.items():
        if quant == 'all':
            bins = np.arange(12, 96) - 0.5
            xticks = np.arange(10, 98, 2)
            yticks = np.arange(7)
        else:
            bins = np.arange(40) - 0.5
            xticks = np.arange(40)
            yticks = np.arange(15)
        vars_df.groupby('topic')['text'].count().plot(title=f'Number of vars in {quant} quantile CW12B', kind='hist',
                                                      bins=bins)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.ylabel('Number of topics')
        plt.xlabel('Number of Variants per topic')
        # plt.grid(True)
        plt.show()


def calc_statistics(qdf: pd.DataFrame, apdb: dt.ResultsReader, title_queries_df: pd.DataFrame,
                    title_ap: dt.ResultsReader, filter_functions_dict: dict, quantiles_dict: dict, corpus):
    """
    This function constructs:
    QUERY_GROUPS={'title'" 'Title', 'top': 'MaxAP', 'low': 'MinAP', 'medh': 'MedHiAP', 'medl': 'MedLoAP'}
    QUANTILES = {'all': 'All', 'med': 'Med', 'top': 'Top', 'low': 'Low'}
    queries_groups_dict: {group: df}
    quant_variants_dict: {quantile: df}
    """
    # Add topic column to qdf
    for topic, q_vars in apdb.query_vars.items():
        qdf.loc[qdf['qid'].isin(q_vars), 'topic'] = topic
    # Create queries_groups_dict
    _title_df = pd.merge(title_queries_df, title_ap.data_df, on='qid')
    queries_groups_dict = {'title': _title_df.set_index('qid')}
    for qgroup in QUERY_GROUPS:
        _df = filter_functions_dict[qgroup](qdf, apdb)
        queries_groups_dict[qgroup] = _df.merge(apdb.data_df, on='qid').set_index('qid')
    QUERY_GROUPS['title'] = 'Title'
    # Create quant_variants_dict
    _all_vars_df = pd.merge(qdf, apdb.data_df, on='qid')
    quant_variants_dict = {'all': _all_vars_df.set_index('qid')}
    for quant in QUANTILES:
        _df = filter_quant_variants(qdf, apdb, quantiles_dict[quant])
        quant_variants_dict[quant] = _df.merge(apdb.data_df, on='qid').set_index('qid')
    QUANTILES['all'] = 'All'

    _map_dict = {}
    _wo_removal_dict = {}
    for qgroup, group_df in queries_groups_dict.items():
        single_map = group_df['ap'].mean()
        _dict = {'Single': single_map}
        for quant, vars_df in quant_variants_dict.items():
            _raw_map = vars_df['ap'].mean()
            _wo_removal_dict[QUANTILES[quant]] = _raw_map
            # Remove queries group from the quantile variations (after the quantile was filtered)
            quant_wo_group_df = remove_q1_from_q2(group_df, vars_df)
            _map_wo_group = quant_wo_group_df['ap'].mean()
            _dict[QUANTILES[quant]] = _map_wo_group
        _map_dict[QUERY_GROUPS[qgroup]] = _dict
    _map_dict['W/O Removal'] = _wo_removal_dict
    stats_df = pd.DataFrame.from_dict(_map_dict, orient='index')
    formatters = [add_format] * len(stats_df.columns)
    print(stats_df.to_latex(formatters=formatters, escape=False))

    plot_robust_histograms(quant_variants_dict) if corpus == 'ROBUST' else plot_cw_histograms(quant_variants_dict)


def main(args):
    queries_txt_file = args.queries
    queries_to_remove = args.remove
    ap_file = args.ap
    queries_group = args.group
    quant_variants = args.quant
    stats = args.stats

    filter_functions_dict = {'top': filter_top_queries, 'low': filter_low_queries, 'medl': filter_medl_queries,
                             'medh': filter_medh_queries}
    quantiles_dict = {'low': [0, 0.33], 'med': [0.33, 0.66], 'top': [0.66, 1]}

    # # Uncomment for Debugging !!!!!
    # print('\n\n\n----------!!!!!!!!!!!!--------- Debugging Mode ----------!!!!!!!!!!!!---------\n\n\n')
    # # quant_variants = 'low'
    # ap_file = dt.ensure_file('~/QppUqvProj/Results/ROBUST/test/raw/QLmap1000')
    # queries_txt_file = dt.ensure_file('~/QppUqvProj/data/ROBUST/queries_ROBUST_UQV_full.txt')
    # stats = True

    corpus = 'ROBUST' if 'ROBUST' in queries_txt_file else 'ClueWeb12B'
    if queries_txt_file:
        qdb = dt.QueriesTextParser(queries_txt_file, 'uqv')
        qdb.queries_df = remove_duplicates(qdb)
        if queries_to_remove:
            qdb_rm = dt.QueriesTextParser(queries_to_remove)
            qdb.queries_df = remove_q1_from_q2(qdb_rm.queries_df, qdb)
        if ap_file:
            apdb = dt.ResultsReader(ap_file, 'ap')
            if queries_group != 'title':
                qdb.queries_df = filter_functions_dict[queries_group](qdb.queries_df, apdb)
            elif quant_variants:
                qdb.queries_df = filter_quant_variants(qdb.queries_df, apdb, quantiles_dict[quant_variants])
            if stats:
                title_queries_file = dt.ensure_file(f'~/QppUqvProj/data/{corpus}/queries_{corpus}_title.txt')
                title_queries_df = dt.QueriesTextParser(title_queries_file).queries_df
                title_ap_file = dt.ensure_file(f'~/QppUqvProj/Results/{corpus}/test/basic/QLmap1000')
                title_ap = dt.ResultsReader(title_ap_file, 'ap')
                calc_statistics(qdb.queries_df, apdb, title_queries_df, title_ap, filter_functions_dict, quantiles_dict,
                                corpus)
                return

        # # In order to convert the vid (variants ID) to qid, uncomment next line
        # queries_df = convert_vid_to_qid(queries_df)

        write_queries_to_files(qdb.queries_df, corpus=corpus, queries_group=queries_group, quantile=quant_variants,
                               remove=queries_to_remove)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
