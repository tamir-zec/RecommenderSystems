import os
import re

import gensim
import math
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords_list = stopwords.words("english")


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
    model = gensim.models.KeyedVectors.load_word2vec_format(fname=model_path, binary=True, limit=700000)
    return model


def calc_item_embedding():
    pass


def calc_user_embedding():
    pass


def build_tfidf_w2v_vectors(vector_dim=200):
    model = load_model()
    # Read each user-item review
    df = pd.read_csv('data/clean_reviews.csv', names=['user_id', 'business_id', 'clean_review'])
    non_english_reviews = df[df['clean_review'].isnull()]
    non_english_reviews[['user_id', 'business_id']].to_csv('data/non_english_reviews.csv', index=False)
    df = df[df['clean_review'].isnull() == False]
    # Building TFIDF model and calculate TFIDF score
    tfidf = TfidfVectorizer(analyzer='word')
    tfidf.fit(df['clean_review'])
    # Get the words from the TF-IDF model
    tfidf_dict = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
    tfidf_feature = tfidf.get_feature_names()

    df_length = len(df)
    batch_size = 10000
    num_batch = math.ceil(df_length / batch_size)
    # Get batch of sentence and create a text file
    for chunk in np.array_split(df, num_batch):
        embedding_vectors = []
        for line in chunk['clean_review']:
            review_vec = np.zeros(vector_dim)
            # Num of words with a valid vector
            weight_sum = 0
            line_split = line.split()
            for word in line_split:
                if word in model.vocab and word in tfidf_feature:
                    w2v_vec = model[word]
                    tf_idf = tfidf_dict[word] * (line_split.count(word) / len(line_split))
                    review_vec += (w2v_vec * tf_idf)
                    weight_sum += tf_idf

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
    build_tfidf_w2v_vectors()
