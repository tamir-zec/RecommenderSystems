import os

import numpy as np
import pandas as pd

from SVD_MF import RecommenderSystem

RANDOM_SEED = 4

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    recsys = RecommenderSystem('data/userTrainData.csv')
    recsys.Load()
    for learning_rate in [0.075, 0.05, 0.025, 0.01, 0.005]:
        for sgd_step_size in [0.075, 0.05, 0.025, 0.01, 0.005]:
            for latent_factors in [5, 10, 20]:
                for n_iter in [20]:
                    recsys.learning_rate = learning_rate
                    recsys.sgd_step_size = sgd_step_size
                    recsys.latent_factors = latent_factors
                    recsys.initialize_data()
                    rmse_results, n = recsys.TrainBaseModel(n_iter)

                    res = pd.DataFrame({'learning rate': learning_rate,
                                        'sgd step size': sgd_step_size,
                                        'latent factors': latent_factors,
                                        'iterations': n,
                                        'last RMSE': rmse_results[-2],
                                        'RMSE list': [rmse_results]}, index=[0])


                    if not os.path.exists(os.path.join('results')):
                        os.makedirs(os.path.join('results'))
                    save_result_path = os.path.join('results', 'basic_model_results.csv')
                    if os.path.exists(save_result_path):
                        res.to_csv(save_result_path, header=False, mode='a', index=False)
                    else:
                        res.to_csv(save_result_path, index=False)
