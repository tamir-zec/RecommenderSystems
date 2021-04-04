import os

import numpy as np
import pandas as pd

from SVD_MF import RecommenderSystem

RANDOM_SEED = 4
TRAIN_MODE = False


def TrainHybridModel():
    recsys1 = RecommenderSystem('data', learning_rate=0.05, sgd_step_size=0.05, latent_factors=50,
                                rand_const=0.2, advanced=False, content=False, train_mode=TRAIN_MODE)
    recsys1.Load()
    recsys1.TrainBaseModel()
    base_model_predictions, _ = recsys1.PredictRating()

    recsys2 = RecommenderSystem('data', learning_rate=0.05, sgd_step_size=0.05, implicit_lrate=0.05,
                                latent_factors=50, rand_const=0.2, advanced=True, content=False, train_mode=TRAIN_MODE)
    recsys2.Load()
    recsys2.TrainAdvancedModel()
    advanced_model_predictions, _ = recsys2.PredictRating()

    recsys3 = RecommenderSystem('data', advanced=False, content=True, train_mode=TRAIN_MODE)
    recsys3.Load()
    recsys3.TrainContentModel()
    content_model_predictions, _ = recsys3.PredictRating()

    hybrid_predictions = base_model_predictions + advanced_model_predictions + content_model_predictions / 3
    rmse = recsys1.calc_rmse(recsys1.ratings_matrix, hybrid_predictions)
    mae = recsys1.calc_mae(recsys1.ratings_matrix, hybrid_predictions)

    print(f'Hybrid Model: RMSE {rmse}, MAE {mae}')


if __name__ == '__main__':

    np.random.seed(RANDOM_SEED)
    TrainHybridModel()

    recsys = RecommenderSystem('data', advanced=False, content=False, train_mode=TRAIN_MODE)
    recsys.Load()
    for learning_rate in [0.05, 0.04, 0.03]:
        for sgd_step_size in [0.05, 0.04, 0.03]:
            for latent_factors, rand_const in [(50, 0.1), (100, 0.05)]:
                for n_iter in [1]:
                    recsys.learning_rate = learning_rate
                    recsys.sgd_step_size = sgd_step_size
                    recsys.latent_factors = latent_factors
                    recsys.rand_const = rand_const
                    recsys.initialize_data()
                    if recsys.advanced:
                        rmse_results, mae_results, n = recsys.TrainAdvancedModel(n_iter)
                    else:
                        rmse_results, mae_results, n = recsys.TrainBaseModel(n_iter)

                    print(f'{n} iterations: RMSE {rmse_results[-2]}, MAE {mae_results[-2]}')
                    res = pd.DataFrame({'learning rate': learning_rate,
                                        'sgd step size': sgd_step_size,
                                        'latent factors': latent_factors,
                                        'iterations': n,
                                        'last RMSE': rmse_results[-2],
                                        'RMSE list': [rmse_results],
                                        'last MAE': mae_results[-2],
                                        'MAE list': [mae_results]},
                                       index=[0])

                    if not os.path.exists(os.path.join('results')):
                        os.makedirs(os.path.join('results'))
                    save_result_path = os.path.join('results', 'basic_model_results.csv')
                    if recsys.advanced:
                        save_result_path = os.path.join('results', 'advanced_model_results.csv')
                    if os.path.exists(save_result_path):
                        res.to_csv(save_result_path, header=False, mode='a', index=False)
                    else:
                        res.to_csv(save_result_path, index=False)
