import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse


class RecommenderSystem:

    def __init__(self, data_path, learning_rate=0.05, sgd_step_size=0.05, implicit_lrate=0.05, latent_factors=20,
                 rand_const=0.2, advanced=False, content=False, train_mode=True):
        self.learning_rate = learning_rate
        self.sgd_step_size = sgd_step_size
        self.latent_factors = latent_factors
        self.rand_const = rand_const
        self.data_path = data_path
        self.advanced = advanced
        if self.advanced:
            self.users2items = defaultdict(list)
            self.implicit_learning_rate = implicit_lrate
        self.content = content
        self.train_mode = train_mode

    def Load(self):
        load_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.data_path, 'userTrainData.csv')

        ratings, users, items = [], [], []
        for df in pd.read_csv(load_directory, chunksize=100000, usecols=['user_id', 'business_id', 'stars']):
            ratings.extend(df['stars'].values)
            users.extend(df['user_id'].values)
            items.extend(df['business_id'].values)
            if self.advanced:
                for user, item in zip(df['user_id'].values, df['business_id'].values):
                    self.users2items[user].append(item)

        users = np.array(users)
        items = np.array(items)
        self.total_users = len(np.unique(users))
        self.total_items = len(np.unique(items))
        self.total_ratings = len(ratings)
        self.avg_ratings = np.mean(ratings)
        self.user2idx = {user: idx for idx, user in enumerate(np.unique(users))}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        self.item2idx = {item: idx for idx, item in enumerate(np.unique(items))}
        map_users = lambda x: self.user2idx[x]
        map_items = lambda x: self.item2idx[x]
        # Convert ids to indices
        users = np.array([map_users(user_i) for user_i in users])
        items = np.array([map_items(item_i) for item_i in items])
        # Create rating matrix as sparse matrix
        self.ratings_matrix = sparse.csr_matrix((ratings, (np.array(users), np.array(items))),
                                                shape=(self.total_users, self.total_items))

        if not self.train_mode:
            self.missing_items = []
            # Initialize test rating matrix
            load_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.data_path,
                                          'userTestData.csv')
            missing_business = []
            missing_users = []
            test_ratings, test_users, test_items = [], [], []
            for df in pd.read_csv(load_directory, chunksize=100000, usecols=['user_id', 'business_id', 'stars']):
                for _, row in df.iterrows():
                    if row['user_id'] in self.user2idx and row['business_id'] in self.item2idx:
                        u_id = self.user2idx[row['user_id']]
                        i_id = self.item2idx[row['business_id']]
                        test_ratings.append(row['stars'])
                        test_users.append(u_id)
                        test_items.append(i_id)
                    else:
                        if row['user_id'] not in self.user2idx:
                            missing_users.append(row['user_id'])
                        if row['business_id'] not in self.item2idx:
                            missing_business.append(row['business_id'])
                        self.missing_items.append((row['user_id'], row['business_id'], row['stars']))
                        continue

            self.test_ratings_matrix = sparse.csr_matrix((test_ratings, (np.array(test_users), np.array(test_items))),
                                                         shape=(self.total_users, self.total_items))
            # if len(missing_users) > 0:
            #     pd.DataFrame(missing_users).to_csv('data/missing_users.csv', index=False, header=False)
            # if len(missing_business) > 0:
            #     pd.DataFrame(missing_business).to_csv('data/missing_business.csv', index=False, header=False)

        self.initialize_data()

    def calc_rmse(self, ratings, predictions):
        N = len(predictions.nonzero()[0])
        result = (ratings - predictions).power(2)
        result_sum = result.sum()
        if not self.train_mode:
            missing_items_preds = self.calc_test_missing()
            N += len(missing_items_preds)
            result_sum += self.calc_missing_rmse_part(missing_items_preds).sum()

        return np.sqrt(result_sum / N)

    def calc_missing_rmse_part(self, missing_items_preds):
        ratings = np.array([item[2] for item in self.missing_items])
        missing_preds = np.array(missing_items_preds)
        return np.power((ratings - missing_preds), 2)

    def calc_mae(self, ratings, predictions):
        N = len(predictions.nonzero()[0])
        result = np.abs(ratings - predictions)
        result_sum = result.sum()
        if not self.train_mode:
            missing_items_preds = self.calc_test_missing()
            N += len(missing_items_preds)
            result_sum += self.calc_missing_mae_part(missing_items_preds).sum()

        return result_sum / N

    def calc_missing_mae_part(self, missing_items_preds):
        ratings = np.array([item[2] for item in self.missing_items])
        missing_preds = np.array(missing_items_preds)
        return np.abs(ratings - missing_preds)

    def TrainBaseModel(self, n_iter=20):
        rmse = []
        mae = []
        # Shuffle entries and calculate SGD for each user/item
        if self.train_mode:
            sgd_indices = np.arange(len(self.train_idx_row))
        else:
            sgd_indices = np.arange(len(self.idx_row))
        for n in range(n_iter):
            np.random.shuffle(sgd_indices)
            self.sgd_step(sgd_indices)
            if self.train_mode:
                predictions = self.calc_predictions()
                rmse.append(self.calc_rmse(self.val_rating_matrix, predictions))
                mae.append(self.calc_mae(self.val_rating_matrix, predictions))
                # Stop rule
                if len(rmse) > 1 and (rmse[-1] > rmse[-2] or mae[-1] > mae[-2]):
                    break

        if not self.train_mode:
            predictions = self.calc_predictions()
            rmse.append(self.calc_rmse(self.test_ratings_matrix, predictions))
            mae.append(self.calc_mae(self.test_ratings_matrix, predictions))

        return rmse, mae, n

    def TrainAdvancedModel(self, n_iter=20):
        rmse = []
        mae = []
        # shuffle entries and calculate SGD for each user/item
        if self.train_mode:
            sgd_indices = np.arange(len(self.train_idx_row))
        else:
            sgd_indices = np.arange(len(self.idx_row))
        for n in range(n_iter):
            np.random.shuffle(sgd_indices)
            self.sgd_step(sgd_indices)
            if self.train_mode:
                predictions = self.calc_predictions()
                rmse.append(self.calc_rmse(self.val_rating_matrix, predictions))
                mae.append(self.calc_mae(self.val_rating_matrix, predictions))
                # Stop rule
                if len(rmse) > 1 and (rmse[-1] > rmse[-2] or mae[-1] > mae[-2]):
                    break

        if not self.train_mode:
            predictions = self.calc_predictions()
            rmse.append(self.calc_rmse(self.test_ratings_matrix, predictions))
            mae.append(self.calc_mae(self.test_ratings_matrix, predictions))

        return rmse, mae, n

    def TrainContentModel(self, top_rec=10):
        pass

    def PredictRating(self):
        model_predictions = self.calc_predictions()
        missing_item_ratings = []
        if not self.train_mode:
            missing_item_ratings = self.calc_test_missing()

        return model_predictions, missing_item_ratings

    def initialize_data(self):
        # Initialize bias vectors
        self.user_bias = self.rand_const * np.random.random(self.total_users)
        self.item_bias = self.rand_const * np.random.random(self.total_items)
        # initialize latent factors matrices
        self.users_matrix = self.rand_const * np.random.rand(self.total_users, self.latent_factors)
        self.items_matrix = self.rand_const * np.random.rand(self.latent_factors, self.total_items)
        # Keep the indices and values of the non-zero entries in the sparse matrix
        self.idx_row, self.idx_col, self.rating_val = sparse.find(self.ratings_matrix)
        # split to train/validation
        if self.train_mode:
            self.train_ratings_matrix, self.val_rating_matrix = self.train_validation_split()
            self.train_idx_row, self.train_idx_col, _ = sparse.find(self.train_ratings_matrix)
        if self.advanced:
            self.implicit_matrix = self.rand_const * np.random.rand(self.latent_factors, self.total_items)

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

    def sgd_step_base(self, sgd_indices):
        for idx in sgd_indices:
            if self.train_mode:
                u = self.train_idx_row[idx]
                i = self.train_idx_col[idx]
            else:
                u = self.idx_row[idx]
                i = self.idx_col[idx]
            prediction = self.calc_rating(u, i)
            # Error
            if self.train_mode:
                e = (self.train_ratings_matrix[u, i] - prediction)
            else:
                e = (self.ratings_matrix[u, i] - prediction)
            # Update biases
            self.user_bias[u] += self.sgd_step_size * (e - self.learning_rate * self.user_bias[u])
            self.item_bias[i] += self.sgd_step_size * (e - self.learning_rate * self.item_bias[i])
            # Update latent factors
            self.items_matrix[:, i] += self.sgd_step_size * (
                    e * self.users_matrix[u, :] - self.learning_rate * self.items_matrix[:, i])
            self.users_matrix[u, :] += self.sgd_step_size * (
                    e * self.items_matrix[:, i] - self.learning_rate * self.users_matrix[u, :])

    def sgd_step_advanced(self, sgd_indices):
        for idx in sgd_indices:
            if self.train_mode:
                u = self.train_idx_row[idx]
                i = self.train_idx_col[idx]
            else:
                u = self.idx_row[idx]
                i = self.idx_col[idx]
            prediction = self.calc_rating(u, i)
            # Error
            if self.train_mode:
                e = (self.train_ratings_matrix[u, i] - prediction)
            else:
                e = (self.ratings_matrix[u, i] - prediction)
            # Update biases
            self.user_bias[u] += self.sgd_step_size * (e - self.learning_rate * self.user_bias[u])
            self.item_bias[i] += self.sgd_step_size * (e - self.learning_rate * self.item_bias[i])
            # Update latent factors - new rule with implicit data
            self.items_matrix[:, i] += self.sgd_step_size * (e * (
                    self.users_matrix[u, :] + self.get_implicit_weights_user(
                u)) - self.implicit_learning_rate * self.items_matrix[:, i])
            self.users_matrix[u, :] += self.sgd_step_size * (
                    e * self.items_matrix[:, i] - self.implicit_learning_rate * self.users_matrix[u, :])
            # Update implicit matrix
            self.update_implicit_matrix(u, i, e)

    def sgd_step(self, sgd_indices):
        if self.advanced:
            return self.sgd_step_advanced(sgd_indices)
        else:
            return self.sgd_step_base(sgd_indices)

    def update_implicit_matrix(self, user_idx, item_idx, error):
        item_indices = []
        user_name = self.idx2user[user_idx]
        for item_name in self.users2items[user_name]:
            item_indices.append(self.item2idx[item_name])
        N = len(item_indices)
        implicit_weight = 1 / np.sqrt(N)
        implicit_total_update = np.transpose(np.tile(error * implicit_weight * self.items_matrix[:, item_idx], (N, 1)))
        self.implicit_matrix[:, item_indices] += self.sgd_step_size * (
                implicit_total_update - self.implicit_learning_rate * self.implicit_matrix[:, item_indices])

    def calc_rating_base(self, user, item):
        factor_product = np.multiply(self.users_matrix[user, :], self.items_matrix[:, item])
        rating = self.avg_ratings + self.item_bias[item] + self.user_bias[user]
        for factor_mul in factor_product:
            rating = rating + factor_mul
            if rating < 1:
                rating = 1
            elif rating > 5:
                rating = 5
        return rating

    def calc_rating_advanced(self, user, item):
        implicit_weight = self.get_implicit_weights_user(user)
        new_Pu = self.users_matrix[user, :] + implicit_weight
        factor_product = np.multiply(new_Pu, self.items_matrix[:, item])
        rating = self.avg_ratings + self.item_bias[item] + self.user_bias[user]
        for factor_mul in factor_product:
            rating = rating + factor_mul
            if rating < 1:
                rating = 1
            elif rating > 5:
                rating = 5
        return rating

    def calc_rating_content(self, user, item):
        # todo: implement

        with open('data/users_recommendations.json', 'r') as f:
            users_recommendations = json.load(f)
        pass

    def calc_rating(self, user, item):
        if self.advanced:
            return self.calc_rating_advanced(user, item)
        elif self.content:
            return self.calc_rating_content(user, item)
        else:
            return self.calc_rating_base(user, item)

    def get_implicit_weights_user(self, user_idx):
        item_indices = []
        user_name = self.idx2user[user_idx]
        for item_name in self.users2items[user_name]:
            item_indices.append(self.item2idx[item_name])
        N_u_implicit = len(item_indices)
        # Calculate rating by new formula (15) from koren 2008
        sum_imp_weight = np.sum(self.implicit_matrix[:, item_indices], axis=1)
        return 1 / np.sqrt(N_u_implicit) * sum_imp_weight

    def calc_predictions(self):
        if self.train_mode:
            return self.calc_train_predictions()
        else:
            return self.calc_test_predictions()

    def calc_train_predictions(self):
        row_idx, col_idx = self.val_rating_matrix.nonzero()
        data = []
        for u_id, i_id in zip(row_idx, col_idx):
            prediction = self.calc_rating(u_id, i_id)
            data.append(prediction)

        predictions = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.total_users, self.total_items))
        return predictions

    def calc_test_predictions(self):
        row_idx, col_idx = self.test_ratings_matrix.nonzero()
        data = []
        for u_id, i_id in zip(row_idx, col_idx):
            prediction = self.calc_rating(u_id, i_id)
            data.append(prediction)
        predictions = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(self.total_users, self.total_items))
        return predictions

    def calc_test_missing(self):
        # Missing items - a list of tuple of user id, business id, and original rating
        missing_items_predictions = []
        for user_id, _, _ in self.missing_items:
            missing_items_predictions.append(self.give_missing_item_pred(user_id))
        return missing_items_predictions

    def give_missing_item_pred(self, user_id):
        user_index = self.user2idx[user_id]
        user_average = self.ratings_matrix[user_index, :].data.mean()
        return user_average
