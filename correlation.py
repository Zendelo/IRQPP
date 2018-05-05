import pandas as pd


def print_cor(df, type):
    print('The {} Correlation is: {:0.4f}'.format(type, df['clarity'].corr(df['ap.1000'], method=type)))


def main():
    params = [5,10,25,50,100,250,500,1000]
    # clarity_file_path = 'clarity-Fiana.res'
    # clarity_file_path = 'clarity-Anna.res'
    ap_file_path = 'QLmap1000'

    for i in params:
        print('\n ********* For {} documents *********'.format(i))
        clarity_file_path = 'clarity-Fiana-{}-docs.res'.format(i)
        clarity_df = pd.read_table(clarity_file_path, delim_whitespace=True, index_col='qid')
        ap_df = pd.read_table(ap_file_path, delim_whitespace=True, index_col='qid')
        merged_df = pd.merge(clarity_df, ap_df, left_index=True, right_index=True)
        print_cor(merged_df, 'pearson')
        print_cor(merged_df, 'spearman')
        print_cor(merged_df, 'kendall')
        print('************** end of {} ************** \n'.format(i))


if __name__ == '__main__':
    main()
