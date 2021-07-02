import ast
import os
import json
import pandas as pd

DATA_DIR = 'data/'
categories = ['toys', 'kindle', 'movies']
data_file_names = ['Toys_and_Games', 'Kindle_Store', 'Movies_and_TV']

for category, file_name in zip(categories, data_file_names):
    print(f'Category: {category}')
    df = pd.read_csv(os.path.join(DATA_DIR, category + '_quality.tsv.gz'), sep='\t')

    filter_ids = []
    filter_ids.extend(df[(df['total_votes'] > 2) & (df['help_rate'] < 0.6)].index.tolist())
    filter_ids.extend(df[df['num_words'] < 6].index.tolist())
    filter_ids.extend(df[df['punct_dense'] > 0.1].index.tolist())
    filter_ids.extend(df[df['%misspell'] > 0.1]['review'].index.tolist())
    filter_ids = list(set(filter_ids))
    print(f'Number of filtered reviews: {len(filter_ids)}')
    print(f'% of filtered reviews: {100 * len(filter_ids) / len(df)}')
    print(f'Number of remained reviews: {len(df) - len(filter_ids)}')

    df = df.drop(index=filter_ids)
    df = df[['user_id', 'item_id', 'rating', 'review']]
    df.rename(columns={'user_id': 'reviewerID',
                       'item_id': 'asin',
                       'rating': 'overall',
                       'review': 'reviewText'},
              inplace=True)
    df['overall'] = df['overall'].astype(str)
    print(f'Total remained reviews: {len(df)}')

    print('Save to json file')
    output_json = ast.literal_eval(df.to_json(orient='records'))
    with open(os.path.join(DATA_DIR, 'filtered_reviews_' + file_name + '_5.json'), 'w') as f:
        for review in output_json:
            f.write(json.dumps(review) + '\n')
