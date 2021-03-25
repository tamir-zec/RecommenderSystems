import os

import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict

class RecommenderSystem:

    def __init__(self, data_path, learning_rate=0.05, sgd_step_size=0.05, latent_factors=20, advance=False):

        self.learning_rate = learning_rate
        self.sgd_step_size = sgd_step_size
        self.latent_factors = latent_factors
        self.data_path = data_path
        self.advance = advance
        if self.advance:
            self.users2items = defaultdict(list)

    def Load(self):
        load_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.data_path)

        ratings, users, items = [], [], []
        for df in pd.read_csv(load_directory, chunksize=10000, usecols=['user_id', 'business_id', 'stars']):
            ratings.extend(df['stars'].values)
            users.extend(df['user_id'].values)
            items.extend(df['business_id'].values)
            if self.advance:
                self.users2items.update(zip(df['user_id'].values, df['business_id'].values))

        users = np.array(users)
        items = np.array(items)
        self.total_users = len(np.unique(users))
        self.total_items = len(np.unique(items))
        self.total_ratings = len(ratings)
        self.avg_ratings = np.mean(ratings)
        self.users2id = {user: idx for idx, user in enumerate(np.unique(users))}
        self.item2id = {item: idx for idx, item in enumerate(np.unique(items))}
        map_users = lambda x: self.users2id[x]
        map_items = lambda x: self.item2id[x]
        # Convert ids to indices
        users = np.array([map_users(user_i) for user_i in users])
        items = np.array([map_items(item_i) for item_i in items])
        # Create rating matrix as sparse matrix
        self.ratings_matrix = sparse.csr_matrix((ratings, (np.array(users), np.array(items))),
                                                shape=(self.total_users, self.total_items))

        self.initialize_data()

    def calc_rmse(self, ratings, predictions):
        N = len(predictions.nonzero()[0])
        result = (ratings - predictions).power(2)
        return np.square(result.sum() / N)

    def TrainBaseModel(self, n_iter=20):
        rmse = []
        # shuffle entries and calculate SGD for each user/item
        sgd_indices = np.arange(len(self.idx_row))
        for n in range(n_iter):
            np.random.shuffle(sgd_indices)
            self.sgd_step(sgd_indices)
            predictions = self.calc_predictions()
            rmse.append(self.calc_rmse(self.val_rating_matrix, predictions))
            # Stop rule
            if len(rmse) > 1 and rmse[-1] > rmse[-2]:
                break

        return rmse, n

    def TrainAdvancedModel(self, n_iter=20):
        rmse = []
        # shuffle entries and calculate SGD for each user/item
        sgd_indices = np.arange(len(self.idx_row))
        for n in range(n_iter):
            np.random.shuffle(sgd_indices)
            self.sgd_step(sgd_indices)
            predictions = self.calc_predictions()
            rmse.append(self.calc_rmse(self.val_rating_matrix, predictions))
            # Stop rule
            if len(rmse) > 1 and rmse[-1] > rmse[-2]:
                break

        return rmse, n

    def initialize_data(self):
        # Initialize bias vectors
        self.user_bias = np.random.random(self.total_users)
        self.item_bias = np.random.random(self.total_items)
        # initialize latent factors matrices
        self.users_matrix = np.random.rand(self.total_users, self.latent_factors)
        self.items_matrix = np.random.rand(self.latent_factors, self.total_items)
        # Keep the indices and values of the non-zero entries in the sparse matrix
        self.idx_row, self.idx_col, self.rating_val = sparse.find(self.ratings_matrix)
        # split to train/validation
        self.train_ratings_matrix, self.val_rating_matrix = self.train_validation_split()
        if self.advance:
            self.implicit_matrix = np.random.rand(self.latent_factors, self.total_items)


    def train_validation_split(self, validation_size=0.2):
        validation_indices = np.random.choice(range(len(self.idx_row)), size=int(len(self.idx_row) * validation_size))
        train_ratings_matrix = self.ratings_matrix.copy()
        # Place zeroes in the validation set entries and remove them from the matrix
        for u_id, i_id in zip(self.idx_row[validation_indices], self.idx_col[validation_indices]):
            train_ratings_matrix[u_id, i_id] = 0
        train_ratings_matrix.eliminate_zeros()
        # Create new matrix for the validation set
        val_rating_matrix = sparse.csr_matrix((self.rating_val[validation_indices],
                                               (self.idx_row[validation_indices], self.idx_col[validation_indices])),
                                              shape=(self.total_users, self.total_items))

        return train_ratings_matrix, val_rating_matrix

    def sgd_step(self, sgd_indices):
        for idx in sgd_indices:
            u = self.idx_row[idx]
            i = self.idx_col[idx]
            prediction = self.calc_rating(u, i)
            # Error
            e = (self.ratings_matrix[u, i] - prediction)
            # Update biases
            self.user_bias[u] += self.sgd_step_size * (e - self.learning_rate * self.user_bias[u])
            self.item_bias[i] += self.sgd_step_size * (e - self.learning_rate * self.item_bias[i])

            # Update latent factors
            self.items_matrix[:, i] += self.sgd_step_size * (
                    e * self.users_matrix[u, :] - self.learning_rate * self.items_matrix[:, i])
            self.users_matrix[u, :] += self.sgd_step_size * (
                    e * self.items_matrix[:, i] - self.learning_rate * self.users_matrix[u, :])

    def calc_rating(self, user, item):
        rating = self.avg_ratings + self.item_bias[item] + self.user_bias[user] + (
            np.dot(self.users_matrix[user, :], self.items_matrix[:, item]))
        if rating < 1:
            rating = 1
        elif rating > 5:
            rating = 5
        return rating

    def calc_predictions(self):
        row_idx, col_idx = self.val_rating_matrix.nonzero()
        data = []
        for u_id, i_id in zip(row_idx, col_idx):
            prediction = self.calc_rating(u_id, i_id)
            data.append(prediction)

        predictions = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.total_users, self.total_items))
        return predictions
