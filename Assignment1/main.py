import numpy as np

from SVD_MF import RecommenderSystem

RANDOM_SEED = 4

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    recsys = RecommenderSystem('data/userTrainData.csv')
    recsys.Load()
    recsys.TrainBaseModel()
