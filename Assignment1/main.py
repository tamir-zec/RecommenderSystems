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
    learning_rate = 0.03
    sgd_step_size = 0.03
    implicit_lr = 0.05
    rand_const_advanced = 0.05
    rand_const_basic = 0.1
    latent_factors = 50
    n_iter = 500
    recsys_svd = RecommenderSystem('data', advanced=False, content=False, train_mode=True,learning_rate= learning_rate,
                                   sgd_step_size=sgd_step_size ,rand_const=rand_const_basic,latent_factors=latent_factors)
    recsys_svd.Load()
    recsys_svd.initialize_data()
    rmse_results_val, mae_results_val, rmse_results_train, mae_results_train, n = recsys_svd.TrainBaseModel(n_iter)

    recsys_svdpp = RecommenderSystem('data', advanced=True, content=False, train_mode=True, learning_rate=learning_rate,
                                   sgd_step_size=sgd_step_size, implicit_lrate=implicit_lr, rand_const=rand_const_advanced,
                                   latent_factors=latent_factors)

    recsys_svdpp.Load()
    recsys_svdpp.initialize_data()
    rmse_results_val_pp, mae_results_val_pp, rmse_results_train_pp, mae_results_train_pp, n_pp = recsys_svd.TrainAdvancedModel(n_iter)

    base_res_train = pd.DataFrame({'learning rate': learning_rate,
                        'sgd step size': sgd_step_size,
                        'latent factors': latent_factors,
                        'random const': rand_const_basic,
                        'iterations': n,
                        'last RMSE': rmse_results_train[-2],
                        'RMSE list': [rmse_results_train],
                        'last MAE': mae_results_train[-2],
                        'MAE list': [mae_results_train]
                        },
                       index=[0])
    base_res_val = pd.DataFrame({'learning rate': learning_rate,
                                   'sgd step size': sgd_step_size,
                                   'latent factors': latent_factors,
                                   'random const': rand_const_basic,
                                   'iterations': n,
                                   'last RMSE': rmse_results_val[-2],
                                   'RMSE list': [rmse_results_val],
                                   'last MAE': mae_results_val[-2],
                                   'MAE list': [mae_results_val]
                                   },
                                  index=[0])

    advanced_res_train = pd.DataFrame({'learning rate': learning_rate,
                                   'sgd step size': sgd_step_size,
                                   'latent factors': latent_factors,
                                   'implicit learning rate': implicit_lr,
                                   'random const': rand_const_advanced,
                                   'iterations': n,
                                   'last RMSE': rmse_results_train_pp[-2],
                                   'RMSE list': [rmse_results_train_pp],
                                   'last MAE': mae_results_train_pp[-2],
                                   'MAE list': [mae_results_train_pp]
                                   },
                                  index=[0])

    advanced_res_val = pd.DataFrame({'learning rate': learning_rate,
                                       'sgd step size': sgd_step_size,
                                       'latent factors': latent_factors,
                                       'implicit learning rate': implicit_lr,
                                       'random const': rand_const_advanced,
                                       'iterations': n,
                                       'last RMSE': rrmse_results_val_pp[-2],
                                       'RMSE list': [rrmse_results_val_pp],
                                       'last MAE': mae_results_val_pp[-2],
                                       'MAE list': [mae_results_val_pp]
                                       },
                                      index=[0])

    if not os.path.exists(os.path.join('results')):
        os.makedirs(os.path.join('results'))

    train_save_result_path = os.path.join('results', 'basic_model_results_train.csv')
    val_save_result_path = os.path.join('results', 'basic_model_results_train.csv')

    trainAdv_save_result_path = os.path.join('results', 'advanced_model_results_train.csv')
    valAdv_save_result_path = os.path.join('results', 'advanced_model_results_train.csv')

    paths = [train_save_result_path, val_save_result_path, trainAdv_save_result_path, valAdv_save_result_path]
    results = [base_res_train, base_res_val, advanced_res_train, advanced_res_val]
    for path, result in zip(paths, results):
        if os.path.exists(path):
            result.to_csv(path, header=False, mode='a', index=False)
    else:
            result.to_csv(path, index=False)


    # elif recsys.content:
    #   save_result_path = os.path.join('results', 'content_model_results.csv')
    # if os.path.exists(save_result_path):
    #     res.to_csv(save_result_path, header=False, mode='a', index=False)
    # else:
    #     res.to_csv(save_result_path, index=False)
