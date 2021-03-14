import os

import numpy as np
import pandas as pd
from scipy import sparse


class RecommenderSystem:
    def __init__(self, data_path, learning_rate=0.05, sgd_step_size=0.05, latent_factors=20, sgd_batch_size=100):
        self.learning_rate = learning_rate
        self.sgd_step_size = sgd_step_size
        self.latent_factors = latent_factors
        self.data_path = data_path
        self.sgd_batch_size = sgd_batch_size

    def Load(self):
        load_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.data_path)

        ratings = []
        users = []
        business = []

        for df in pd.read_csv(load_directory, chunksize=10000, usecols=['user_id', 'business_id', 'stars']):
            ratings.extend(df['stars'].values)
            users.extend(df['user_id'].values)
            business.extend(df['business_id'].values)

        users = np.array(users)
        business = np.array(business)
        self.total_users = len(np.unique(users))
        self.total_business = len(np.unique(business))
        self.user_bias = np.random.random(self.total_users)
        self.item_bias = np.random.random(self.total_business)
        self.total_ratings = len(ratings)
        self.avg_ratings = np.mean(ratings)
        self.users2id = {user: idx for idx, user in enumerate(np.unique(users))}
        self.business2id = {business: idx for idx, business in enumerate(np.unique(business))}
        map_users = lambda x: self.users2id[x]
        map_business = lambda x: self.business2id[x]

        users = np.array([map_users(user_i) for user_i in users])
        business = np.array([map_business(business_i) for business_i in business])

        self.ratings_matrix = sparse.csr_matrix((ratings, (np.array(users), np.array(business))),
                                                shape=(self.total_users, self.total_business))

    def rmse(self, ratings, predictions):
        return np.square(sum(np.power(ratings - predictions, 2)) / self.total_ratings)

    def TrainBaseModel(self):
        self.initialize_matrices()
        # split to train/validation
        train_ratings_matrix, val_rating_matrix = self.train_validation_split()
        # train iterations that do all
        return 0

    def initialize_matrices(self):
        self.users_matrix = np.random.rand((self.total_users, self.latent_factors))
        self.item_matrix = np.random.rand((self.total_business, self.latent_factors))

    def train_iterations(self, n_iter=10):
        # sample n entries - add a hyper parameter for batch size
        self.idx_row, self.idx_col = self.ratings_matrix.nonzero()
        # choose sample of batch size to calcuate sgd

        # calculate SGD for each user/item
        # move values towards the derivitive
        # self.ratings_matrix[] - self.users_matrix*self.item_matrix
        return 0

    def train_validation_split(self, validation_size=0.2):
        np.random.choice(range(self.idx_row), size=np.round(len(self.idx_row) * validation_size))
        return 0, 0

    def sgd_step(self, validation_idxs):
        for idx in validation_idxs:
            u = self.users_matrix[idx]
            i = self.item_matrix[idx]
            prediction = self.calc_rating(u, i)
            e = (self.ratings_matrix[u, i] - prediction)  # error
            # Update biases
            self.user_bias[u] += self.sgd_step_size * (e - self.learning_rate * self.user_bias[u])
            self.item_bias[i] += self.sgd_step_size * (e - self.learning_rate * self.item_bias[i])

            # Update latent factors
            self.item_matrix[i, :] += self.sgd_step_size * (e * self.users_matrix[u, :] - self.learning_rate * self.i)
            self.users_matrix[u, :] += self.sgd_step_size * (e * self.item_matrix[i, :] - self.learning_rate *
                                                             self.users_matrix[u, :])

    def calc_rating(self, user, item):
        return self.item_bias[item] + self.user_bias[user] + (user * item)



if __name__ == '__main__':
    recsys = RecommenderSystem('data/userTrainData.csv')
    recsys.Load()
