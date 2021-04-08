import os

import numpy as np
import pandas as pd

from SVD_MF import RecommenderSystem

RANDOM_SEED = 4
TRAIN_MODE = False


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

    for weights in [[0.5, 0.3, 0.2], [0.5, 0.4, 0.1], [0.4, 0.4, 0.2]]:
        hybrid_predictions = (weights[0] * base_model_predictions +
                              weights[1] * advanced_model_predictions +
                              weights[2] * content_model_predictions) / 3
        rmse = recsys1.calc_rmse(recsys1.ratings_matrix, hybrid_predictions)
        mae = recsys1.calc_mae(recsys1.ratings_matrix, hybrid_predictions)

        print(f'Hybrid Model: weights{weights}, RMSE {rmse}, MAE {mae}')
        res = pd.DataFrame({'weights': [weights],
                            'RMSE': rmse,
                            'MAE': mae
                            },
                           index=[0])
    save_result_path = os.path.join('results', 'hybrid_model_results.csv')
    if os.path.exists(save_result_path):
        res.to_csv(save_result_path, header=False, mode='a', index=False)
    else:
        res.to_csv(save_result_path, index=False)


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    TrainHybridModel()

    # recsys = RecommenderSystem('data', advanced=False, content=True, train_mode=TRAIN_MODE)
    # recsys.Load()
    # for learning_rate in [0.03]:
    #     for sgd_step_size in [0.03]:
    #         # for implicit_lr in [0.05]:
    #         #     for rand_const in [0.05]:
    #         latent_factors = 50
    #         n_iter = 4
    #         recsys.learning_rate = learning_rate
    #         recsys.sgd_step_size = sgd_step_size
    #         recsys.latent_factors = latent_factors
    #         # recsys.rand_const = rand_const
    #         # recsys.implicit_learning_rate = implicit_lr
    #         if recsys.advanced:
    #             recsys.initialize_data()
    #             rmse_results, mae_results, n = recsys.TrainAdvancedModel(n_iter)
    #         elif recsys.content:
    #             rmse_results, mae_results, n = recsys.TrainContentModel()
    #         else:
    #             recsys.initialize_data()
    #             rmse_results, mae_results, n = recsys.TrainBaseModel(n_iter)
    #
    #         print(f'{n} iterations: RMSE {rmse_results}, MAE {mae_results}')
    #         # print(f'{n} iterations: RMSE {rmse_results[-2]}, MAE {mae_results[-2]}')
    #         res = pd.DataFrame({'learning rate': learning_rate,
    #                             'sgd step size': sgd_step_size,
    #                             'latent factors': latent_factors,
    #                             # 'implicit learning rate': implicit_lr,
    #                             # 'random const': rand_const,
    #                             'iterations': n,
    #                             'last RMSE': rmse_results[0],  # rmse_results[-2],
    #                             'RMSE list': [rmse_results],
    #                             'last MAE': mae_results[0],
    #                             'MAE list': [mae_results]
    #                             },
    #                            index=[0])
    #
    #         if not os.path.exists(os.path.join('results')):
    #             os.makedirs(os.path.join('results'))
    #         save_result_path = os.path.join('results', 'basic_model_results.csv')
    #         if recsys.advanced:
    #             save_result_path = os.path.join('results', 'advanced_model_results.csv')
    #         elif recsys.content:
    #             save_result_path = os.path.join('results', 'content_model_results.csv')
    #         if os.path.exists(save_result_path):
    #             res.to_csv(save_result_path, header=False, mode='a', index=False)
    #         else:
    #             res.to_csv(save_result_path, index=False)
