'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd

TPS_DIR = '../../data/'
categories = ['kindle']  # ['kindle', 'movies', 'toys']
data_file_names = ['Kindle_Store']  # ['Kindle_Store', 'Movies_and_TV', 'Toys_and_Games']

for category, file_name in zip(categories, data_file_names):
    TP_file = os.path.join(TPS_DIR, file_name + '.json.gz')

    f = gzip.open(TP_file, 'r')
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    np.random.seed(2017)

    i = 0
    try:
        for line in f:
            js = json.loads(line)
            if 'reviewText' not in js:
                js['reviewText'] = ''
            if str(js['reviewerID']) == 'unknown':
                print("unknown")
                continue
            if str(js['asin']) == 'unknown':
                print("unknown2")
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))
            i += 1
            if i > 10000:
                break
    except Exception as e:
        print(e)

    data = pd.DataFrame({'user_id': pd.Series(users_id),
                         'item_id': pd.Series(items_id),
                         'ratings': pd.Series(ratings),
                         'reviews': pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]


    def get_count(tp, id):
        count = tp[[id, 'ratings']].groupby(id, as_index=False).size()
        return count


    usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
    unique_uid = usercount.user_id.unique()
    unique_sid = itemcount.item_id.unique()
    item2id = dict((sid, i + 1) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i + 1) for (i, uid) in enumerate(unique_uid))

    data['user_id'] = data['user_id'].apply(lambda x: user2id[x])
    data['item_id'] = data['item_id'].apply(lambda x: item2id[x])
    tp_rating = data[['user_id', 'item_id', 'ratings']]

    n_ratings = tp_rating.shape[0]
    test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_1 = tp_rating[test_idx]
    tp_train = tp_rating[~test_idx]

    data2 = data[test_idx]
    data = data[~test_idx]

    n_ratings = tp_1.shape[0]
    test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_test = tp_1[test_idx]
    tp_valid = tp_1[~test_idx]
    tp_train.to_csv(os.path.join(TPS_DIR, category, category + '_train.csv'), index=False, header=None)
    tp_valid.to_csv(os.path.join(TPS_DIR, category, category + '_valid.csv'), index=False, header=None)
    tp_test.to_csv(os.path.join(TPS_DIR, category, category + '_test.csv'), index=False, header=None)

    user_reviews = {}
    item_reviews = {}
    user_rid = {}
    item_rid = {}
    # Go over train data
    for i in data.values:
        if i[0] in user_reviews:
            user_reviews[i[0]].append(i[3])
            user_rid[i[0]].append(i[1])
        else:
            user_rid[i[0]] = [i[1]]
            user_reviews[i[0]] = [i[3]]
        if i[1] in item_reviews:
            item_reviews[i[1]].append(i[3])
            item_rid[i[1]].append(i[0])
        else:
            item_reviews[i[1]] = [i[3]]
            item_rid[i[1]] = [i[0]]

    # Go over test data
    for i in data2.values:
        if i[0] in user_reviews:
            continue
        else:
            user_rid[i[0]] = [0]
            user_reviews[i[0]] = ['0']
        if i[1] in item_reviews:
            continue
        else:
            item_reviews[i[1]] = [0]
            item_rid[i[1]] = ['0']

    pickle.dump(user_reviews, open(os.path.join(TPS_DIR, category, category + '_user_review'), 'wb'))
    pickle.dump(item_reviews, open(os.path.join(TPS_DIR, category, category + '_item_review'), 'wb'))
    pickle.dump(user_rid, open(os.path.join(TPS_DIR, category, category + '_user_rid'), 'wb'))
    pickle.dump(item_rid, open(os.path.join(TPS_DIR, category, category + '_item_rid'), 'wb'))

    # usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
    # print(np.sort(np.array(usercount.values)))
    # print(np.sort(np.array(itemcount.values)))
