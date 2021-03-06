import json
import os
import re

import gensim
import math
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# from sklearn.feature_extraction.text import TfidfVectorizer

stopwords_list = stopwords.words("english")
EMBEDDING_DIM = 200
MAX_SCORE = 5


class Sentences(object):

    def __init__(self, dirname):
        # Init the directory of documents to train the model
        self.dirname = dirname

    def __iter__(self):
        """
        Iterate over sentences from the corpus.
        Get file with the corpus documents. Each document is a sentence for the model.
        Build a list of sentences where each sentence converts to a list of words.
        For example: "The dog ate my HW" ---> [The, dog, ate, my, HW]
        """
        print("Sentences: iterate over file " + os.path.split(self.dirname)[1])
        # Each file line (sentence) is a single document
        for line in open(self.dirname).readlines():
            line = line.rstrip('\n').split(',')[2]
            yield line.split()


def clean_review_text(load_directory):
    for df in pd.read_csv(load_directory, chunksize=1000000, usecols=['user_id', 'business_id', 'text']):
        df['clean_review'] = df['text'].apply(lambda text: text.lower())
        # Remove non Ascii letters
        df['clean_review'] = df['clean_review'].apply(lambda text: "".join(c for c in text if ord(c) < 128))
        # Clean text from special characters
        df['clean_review'] = df['clean_review'].apply(lambda text: re.sub('[^A-Za-z0-9 ]+', ' ', text.strip()))
        # Remove stopwords
        df['clean_review'] = df['clean_review'].apply(
            lambda text: " ".join([w for w in word_tokenize(text) if w not in stopwords_list]))
        df[['user_id', 'business_id', 'clean_review']].to_csv('data/clean_reviews.csv', index=False, header=False,
                                                              mode='a')


def train_model(dirname):
    try:
        sentences = Sentences(dirname=dirname)
        model = gensim.models.Word2Vec(sentences=sentences, size=200, window=8, min_count=1, workers=10)
        model.train(sentences=sentences, total_examples=model.corpus_count, epochs=2)
        # Normalized vectors
        model.init_sims(replace=True)
        # Save model to directory
        print("train_model: Save model to directory")
        model_path = os.path.join('w2v_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_word2vec_format(os.path.join(model_path, 'w2v_yelp'), binary=True)

    except Exception as e:
        print("train_model: Exception: " + str(e))


def load_model():
    model_path = os.path.join('w2v_model', 'w2v_yelp')
    model = gensim.models.KeyedVectors.load_word2vec_format(fname=model_path, binary=True)  # , limit=700000)
    return model


def create_item_embedding():
    items_dict = {}
    for chunk in pd.read_csv('data/tfidf_vectors.tsv', chunksize=100000, sep='\t'):
        for _, row in chunk.iterrows():
            business_id = row['business_id']
            embedding = np.fromstring(row['embedding_vectors'].replace('\n', '').rstrip(']').lstrip('['), sep=' ')
            if business_id in items_dict:
                items_dict[business_id]['embedding_vectors'] += embedding
                items_dict[business_id]['count_reviews'] += 1
            else:
                items_dict[business_id] = {}
                items_dict[business_id]['embedding_vectors'] = embedding
                items_dict[business_id]['count_reviews'] = 1

    for business_id in items_dict:
        items_dict[business_id]['embedding_vectors'] /= items_dict[business_id]['count_reviews']

    embeddings = pd.DataFrame.from_dict(items_dict, orient='index').reset_index().rename(
        columns={'index': 'business_id'})
    items = pd.read_csv('data/yelp_business.csv', usecols=['business_id', 'stars'])
    items = pd.merge(items, embeddings, on='business_id', how='left')
    if len(items[items['embedding_vectors'].isnull()]) > 0:
        print('There are ' + str(len(items[items['embedding_vectors'].isnull()])) + ' businesses without reviews')
    items.to_csv('data/business_embedding.tsv', sep='\t', index=False)


def create_user_embedding():
    users = pd.read_csv('data/yelp_user.csv', usecols=['user_id', 'average_stars'])
    reviews = pd.read_csv('data/userTrainData.csv', usecols=['user_id', 'business_id', 'stars'])
    users2business = reviews.groupby('user_id').apply(
        lambda gb: {b: s for b, s in zip(gb['business_id'].tolist(), gb['stars'].tolist())})
    users2business = users2business.reset_index().rename(columns={0: 'business'})
    users2business = pd.merge(users2business, users, on='user_id', how='left')

    # Calculate weighted average of the rated items by the user. The weights are the item rating - avg rating of the user
    items_embedding = pd.read_csv('data/business_embedding.tsv', sep='\t')
    items_embedding['embedding_vectors'] = items_embedding['embedding_vectors'].apply(
        lambda x: np.fromstring(x.replace('\n', '').rstrip(']').lstrip('['), sep=' ') if not pd.isnull(x) else x)
    items_embedding = items_embedding[items_embedding['embedding_vectors'].isnull() == False].set_index('business_id')[
        'embedding_vectors'].to_dict()
    user_vectors = {}
    for _, row in users2business.iterrows():
        user_id = row['user_id']
        businesses_scores = row['business']
        user_average = row['average_stars']
        user_dict = {}
        for business, rating in businesses_scores.items():
            if business in items_embedding:
                user_dict[business] = {}
                user_dict[business]['embedding_vector'] = items_embedding[business]
                user_dict[business]['business_score'] = rating
            else:
                print(business + ' has no embedding')

        user_vectors[user_id] = {}
        user_vectors[user_id]['embedding_vector'] = calc_user_vector(user_dict, user_average)
        if user_dict == {}:
            print(user_id + ' has no embedding')

    embeddings = pd.DataFrame.from_dict(user_vectors, orient='index').reset_index().rename(
        columns={'index': 'user_id'})
    embeddings.to_csv('data/users_embedding.tsv', sep='\t', index=False)


def calc_user_vector(user_dict, user_average):
    positive_effect = np.zeros((EMBEDDING_DIM))
    negative_effect = np.zeros((EMBEDDING_DIM))
    liked_counter = 0
    unliked_counter = 0
    total_items = len(user_dict)
    for business in user_dict:
        if user_dict[business]['business_score'] >= 3:
            liked_counter += 1
            positive_effect += user_dict[business]['embedding_vector']
        else:
            unliked_counter += 1
            negative_effect += user_dict[business]['embedding_vector']

        # item_offset = (user_dict[business]['business_score'] - user_average + 1) / MAX_SCORE
    user_vector = (liked_counter / total_items) * positive_effect - (unliked_counter / total_items) * negative_effect
    return user_vector


def calc_similarity_between_user_items(top_rec=10, all_items=False):
    user_embeddings = pd.read_csv('data/users_embedding.tsv', sep='\t')
    user_embeddings['embedding_vector'] = user_embeddings['embedding_vector'].apply(
        lambda x: x.replace('\n', '').rstrip(']').lstrip('[') if not pd.isnull(x) else x)
    user_mat = user_embeddings.set_index('user_id').apply(lambda x: x['embedding_vector'].split(), axis=1,
                                                          result_type='expand')
    item_embeddings = pd.read_csv('data/business_embedding.tsv', sep='\t')
    item_embeddings = item_embeddings[item_embeddings['embedding_vectors'].isnull() == False]
    item_embeddings['embedding_vectors'] = item_embeddings['embedding_vectors'].apply(
        lambda x: x.replace('\n', '').rstrip(']').lstrip('[ ') if not pd.isnull(x) else x)
    item_mat = item_embeddings.set_index('business_id').apply(lambda x: x['embedding_vectors'].split(), axis=1,
                                                              result_type='expand')
    items_names = list(item_embeddings['business_id'])

    if all_items:
        # Calculate the similarity between user and all items
        with open('data/users_ratings.json', 'r') as f:
            users_history = json.load(f)
        users_recommendations = {}
        df_length = len(user_mat)
        batch_size = 10000
        num_batch = math.ceil(df_length / batch_size)
        for batch, chunk_user in enumerate(np.array_split(user_mat, num_batch)):
            print('Go over chunk number ' + str(batch + 1))
            similarity_mat = cosine_similarity(chunk_user.values, item_mat.values)
            for i, user_id in enumerate(chunk_user.index):
                similarities = similarity_mat[i]
                recommendations = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)[
                                  :top_rec + len(users_history[user_id])]
                recommendations = [items_names[i] for i in recommendations]
                for item in users_history[user_id]:
                    if item in recommendations:
                        recommendations.remove(item)
                users_recommendations[user_id] = recommendations[:top_rec]

        with open('data/users_recommendations.json', 'w') as f:
            json.dump(users_recommendations, f)
        # res = np.around(res, decimals=4)
        # df_mat = pd.DataFrame(res)
        # df_mat.index = user_embeddings['user_id']
        # df_mat.columns = item_embeddings['business_id']
        # df_mat.to_csv('data/user_business_similarity.csv')

    else:
        train_data = pd.read_csv('data/userTrainData.csv', usecols=['user_id', 'business_id'])
        test_data = pd.read_csv('data/userTestData.csv', usecols=['user_id', 'business_id'])
        train_data['sim'] = None
        test_data['sim'] = None

        for idx, row in train_data.iterrows():
            similarity = cosine_similarity(user_mat.loc[row['user_id']].values.reshape(1, -1),
                                           item_mat.loc[row['business_id']].values.reshape(1, -1))[0][0]
            train_data.loc[idx]['sim'] = similarity
        train_data.to_csv('data/userTrainDataSimilarity.csv', index=False)

        for idx, row in test_data.iterrows():
            if row['business_id'] in items_names:
                similarity = cosine_similarity(user_mat.loc[row['user_id']].values.reshape(1, -1),
                                               item_mat.loc[row['business_id']].values.reshape(1, -1))[0][0]
                test_data.loc[idx]['sim'] = similarity
        test_data.to_csv('data/userTestDataSimilarity.csv', index=False)


def build_tfidf_w2v_vectors(vector_dim=200):
    model = load_model()
    # Read each user-item review
    df = pd.read_csv('data/clean_reviews.csv', names=['user_id', 'business_id', 'clean_review'])
    # non_english_reviews = df[df['clean_review'].isnull()]
    # non_english_reviews[['user_id', 'business_id']].to_csv('data/non_english_reviews.csv', index=False)
    df = df[df['clean_review'].isnull() == False]
    # # Building TFIDF model and calculate TFIDF score
    # tfidf = TfidfVectorizer(analyzer='word', min_df=0)
    # tfidf.fit(df['clean_review'])
    # # Get the words from the TF-IDF model
    # tfidf_dict = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
    # tfidf_feature = tfidf.get_feature_names()

    df_length = len(df)
    batch_size = 10000
    num_batch = math.ceil(df_length / batch_size)
    # Get batch of sentence and create a text file
    for n, chunk in enumerate(np.array_split(df, num_batch)):
        print('Go over chunk number ' + str(n + 1))
        embedding_vectors = []
        for line in chunk['clean_review']:
            review_vec = np.zeros(vector_dim)
            # Num of words with a valid vector
            weight_sum = 0
            line_split = line.split()
            # tf_idf_scores = []
            delete_indices = []
            for i, word in enumerate(line_split):
                if word not in model.vocab:
                    delete_indices.append(i)
            #     if word in model.vocab and word in tfidf_feature:
            #         tf_idf = tfidf_dict[word] * (line_split.count(word) / len(line_split))
            #         tf_idf_scores.append(tf_idf)
            #     else:
            #         delete_indices.append(i)

            # # Remove words that don't exists in the TFIDF vectorizer / w2v model
            # w2v_vecs = np.delete(w2v_vecs, delete_indices, axis=0)

            # review_vec += np.sum(np.multiply(w2v_vecs, np.array(tf_idf_scores).reshape(len(tf_idf_scores), 1)), axis=0)
            # weight_sum += sum(tf_idf_scores)
            if len(delete_indices) > 0:
                for i in delete_indices:
                    line_split.pop(i)
            w2v_vecs = model[line_split]
            review_vec += np.sum(w2v_vecs, axis=0)
            weight_sum += len(line_split)
            if weight_sum != 0:
                review_vec /= weight_sum

            embedding_vectors.append(review_vec)

        chunk['embedding_vectors'] = embedding_vectors
        if os.path.exists('data/tfidf_vectors.tsv'):
            chunk[['user_id', 'business_id', 'embedding_vectors']].to_csv('data/tfidf_vectors.tsv', sep='\t',
                                                                          index=False, header=False, mode='a')
        else:
            chunk[['user_id', 'business_id', 'embedding_vectors']].to_csv('data/tfidf_vectors.tsv', sep='\t',
                                                                          index=False)


if __name__ == '__main__':
    load_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/userTrainData.csv')
    # clean_review_text(load_directory)
    # train_model('data/clean_reviews.csv')
    # build_tfidf_w2v_vectors()
    # create_item_embedding()
    # create_user_embedding()
    calc_similarity_between_user_items()
