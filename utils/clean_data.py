import os
import tqdm
import pandas as pd

test = False
squad = True
dir_path = '../data/TrainingData'
# dir_path = '../data/TestingData'
columns = ['id', '文章', '問題', '選項1', '選項2', '選項3', '選項4', '答案']
dirty = ['\n', '\r', ' ']

if squad:
    save_dir = '../data/TrainingData/clean_squad'
else:
    save_dir = '../data/TrainingData/clean'

# save_dir = '../data/TestingData'


def clean_str(x):
    for d in dirty:
        x = x.replace(d, '')
    return x


def create_new_row(old_row, idx, context, ans_idx=0, test=False):
    dictionary = {
        'id': idx,
        '文章': clean_str(context),
        '問題': clean_str(old_row[columns.index('問題')]),
        '選項1': clean_str(str(old_row[columns.index('選項1')])),
        '選項2': clean_str(str(old_row[columns.index('選項2')])),
        '選項3': clean_str(str(old_row[columns.index('選項3')])),
        '選項4': clean_str(str(old_row[columns.index('選項4')])),
    }
    if not test:
        dictionary['答案'] = ans_idx
    return pd.Series(dictionary)


def clean_df(df, real_idx=0):
    out_df = pd.DataFrame(columns=columns)
    print('Length before cleaning: {}'.format(len(df)))

    last_context = None
    ans_idx = None
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = row[columns.index('文章')]
        if pd.isna(context):
            context = last_context
        else:
            last_context = context

        if not test and squad:
            ''' for squad, use data only if answer can be found in context'''
            ans_idx = row[columns.index('答案')]
            if type(ans_idx) == str:  # fix the answer idx
                ans_idx = int(ans_idx[-1])
            ans = str(row[columns.index('選項{}'.format(ans_idx))])
            if ans not in context:
                continue

        r = create_new_row(row, real_idx, context, ans_idx, test)
        out_df = out_df.append(r, ignore_index=True)
        real_idx += 1
    return out_df


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    real_idx = 0
    for filename in os.listdir(dir_path):
        if filename.split('.')[-1] != 'xlsx':
            continue
        filepath = os.path.join(dir_path, filename)
        print('\nCleaning {}...'.format(filepath))

        df = pd.read_excel(filepath)
        out_df, real_idx = clean_df(df, real_idx)

        print('Length after cleaning: {}'.format(len(out_df)))
        out_df.to_csv(os.path.join(save_dir, filename.split('.')[0] + '.csv'), index=False)

    # combining all the subset data
    if not test:
        dfs = []
        for name in os.listdir(save_dir):
            if name.split('.')[-1] != 'csv':
                continue
            df = pd.read_csv(os.path.join(save_dir, name))
            dfs.append(df)
        result = pd.concat(dfs)
        print('\nLength of the train set: {}'.format(len(result)))
        result.to_csv(os.path.join(save_dir, 'full_train.csv'), index=False)
