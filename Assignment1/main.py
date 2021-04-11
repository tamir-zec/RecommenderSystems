import os

import numpy as np
import pandas as pd

from SVD_MF import RecommenderSystem

RANDOM_SEED = 4
HYBRID = False
TRAIN_MODE = True
LEARNING_RATE = 0.03
SGD_STEP_SIZE = 0.03
IMPLICIT_LEARNING_RATE = 0.05
RAND_CONST = 0.05
LATENT_FACTORS = 50
ITERATIONS = 4


def TrainHybridModel():
    # Initialize basic model
    recsys1 = RecommenderSystem('data', learning_rate=0.03, sgd_step_size=0.03, latent_factors=50,
                                rand_const=0.1, advanced=False, content=False, train_mode=TRAIN_MODE)
    recsys1.Load()
    recsys1.initialize_data()
    recsys1.TrainBaseModel(n_iter=5)
    base_model_predictions, _ = recsys1.PredictRating()

    # Initialize advanced model
    recsys2 = RecommenderSystem('data', learning_rate=0.03, sgd_step_size=0.03, implicit_lrate=0.05,
                                latent_factors=50, rand_const=0.05, advanced=True, content=False,
                                train_mode=TRAIN_MODE)
    recsys2.Load()
    recsys2.initialize_data()
    recsys2.TrainAdvancedModel(n_iter=4)
    advanced_model_predictions, _ = recsys2.PredictRating()

    # Initialize content model
    recsys3 = RecommenderSystem('data', advanced=False, content=True, train_mode=TRAIN_MODE)
    recsys3.Load()
    recsys3.TrainContentModel()
    content_model_predictions, _ = recsys3.PredictRating()

    weights = [0.5, 0.4, 0.1]
    hybrid_predictions = (weights[0] * base_model_predictions +
                          weights[1] * advanced_model_predictions +
                          weights[2] * content_model_predictions)
    if TRAIN_MODE:
        rmse = recsys1.calc_rmse(recsys1.val_rating_matrix, hybrid_predictions)
        mae = recsys1.calc_mae(recsys1.val_rating_matrix, hybrid_predictions)
    else:
        rmse = recsys1.calc_rmse(recsys1.test_ratings_matrix, hybrid_predictions)
        mae = recsys1.calc_mae(recsys1.test_ratings_matrix, hybrid_predictions)

    return rmse, mae, weights


if __name__ == '__main__':

    np.random.seed(RANDOM_SEED)
    if HYBRID:
        rmse, mae, _ = TrainHybridModel()
        print(f'Hybrid Model: RMSE {rmse}, MAE {mae}')
        res = pd.DataFrame({'RMSE': rmse,
                            'MAE': mae
                            },
                           index=[0])
        save_result_path = os.path.join('results', 'hybrid_model_results.csv')
        if os.path.exists(save_result_path):
            res.to_csv(save_result_path, header=False, mode='a', index=False)
        else:
            res.to_csv(save_result_path, index=False)

    else:
        recsys = RecommenderSystem('data', advanced=False, content=True, train_mode=TRAIN_MODE)
        recsys.Load()
        recsys.learning_rate = LEARNING_RATE
        recsys.sgd_step_size = SGD_STEP_SIZE
        recsys.latent_factors = LATENT_FACTORS
        recsys.rand_const = RAND_CONST
        if recsys.advanced:
            recsys.implicit_learning_rate = IMPLICIT_LEARNING_RATE
            recsys.initialize_data()
            rmse_results, mae_results, n = recsys.TrainAdvancedModel(ITERATIONS)
        elif recsys.content:
            rmse_results, mae_results, n = recsys.TrainContentModel()
        else:
            recsys.initialize_data()
            rmse_results, mae_results, n = recsys.TrainBaseModel(ITERATIONS)

        print(f'{n} iterations: RMSE {rmse_results[-2]}, MAE {mae_results[-2]}')
        res = pd.DataFrame({'learning rate': LEARNING_RATE,
                            'sgd step size': SGD_STEP_SIZE,
                            'latent factors': LATENT_FACTORS,
                            'iterations': n,
                            'last RMSE': rmse_results[-2],
                            'RMSE list': [rmse_results],
                            'last MAE': mae_results[-2],
                            'MAE list': [mae_results]
                            },
                           index=[0])

        if not os.path.exists(os.path.join('results')):
            os.makedirs(os.path.join('results'))
        save_result_path = os.path.join('results', 'basic_model_results.csv')
        if recsys.advanced:
            save_result_path = os.path.join('results', 'advanced_model_results.csv')
        elif recsys.content:
            save_result_path = os.path.join('results', 'content_model_results.csv')
        if os.path.exists(save_result_path):
            res.to_csv(save_result_path, header=False, mode='a', index=False)
        else:
            res.to_csv(save_result_path, index=False)
