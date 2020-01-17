import os
import json
import pandas as pd

test = False
data_dir = '../../data/TrainingData/clean_squad/'
csv_name = 'full_train.csv'
columns = ['id', '文章', '問題', '選項1', '選項2', '選項3', '選項4', '答案']


def create_qa_item(row):
    if not test:
        answer_text = row['選項{}'.format(row['答案'])]
        answer_start = row['文章'].index(answer_text)
        answers = [{
            'text': answer_text,
            'answer_start': answer_start,
            'is_impossible': False,
        }]

        qa_item = {
            'question': row['問題'],
            'id': row['id'],
            'answers': answers,
        }
    else:
        qa_item = {
            'question': row['問題'],
            'id': row['id'],
        }
    return qa_item


def create_article(qa_list, context):
    paragraph = [{
        'qas': qa_list,
        'context': context,
    }]
    # only one paragraph per article
    article = {
        'title': "",
        'paragraphs': paragraph,
    }
    return article


def create_data_dict(df):
    data_list = []
    qa_list = []
    last_context = None
    for _, row in df.iterrows():
        context = row['文章']
        if context != last_context:
            data_list.append(create_article(qa_list, last_context))
            qa_list.clear()
            last_context = context
        qa_list.append(create_qa_item(row))
    data_dict = {
        'version': 'v2.0',
        'data': data_list,
    }
    return data_dict


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    data_dict = create_data_dict(df)

    with open(os.path.join(data_dir, csv_name.split('.')[0] + '.json'), 'w') as f:
        json.dump(data_dict, f)
