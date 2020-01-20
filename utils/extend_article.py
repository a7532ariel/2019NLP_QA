import os
import tqdm
import pandas as pd

columns = ['id', 'article', 'question', 'op1', 'op2', 'op3', 'op4', 'label']

data_dir = '../data'
csv_name = 'train_merge.csv'


def extend_article(df):
    """
    Concat every two articles in order to extend them.
    :param df: input data frame
    :return: same data frame with the extended articles
    """
    for i in tqdm.trange(len(df) // 2):
        row_A = df.iloc[2*i]
        row_B = df.iloc[2*i + 1]
        long_article = row_A['article'] + row_B['article']
        df.at[2*i, 'article'] = long_article
        df.at[2*i+1, 'article'] = long_article
    return df


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    new_df = extend_article(df)
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    new_df.to_csv(os.path.join(data_dir, csv_name.split('.')[0] + '_long_article.csv'), index=False)
