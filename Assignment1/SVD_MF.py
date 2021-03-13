import pandas as pd
import numpy as np
import os


class RecommenderSystem:
    def __init__(self, ratings, learning_rate=0.05, sgd_step=0.05, latent_factors=20, item_bias=0.0, user_bias=0.0):
        self.learning_rate = learning_rate
        self.sgd_step = sgd_step
        self.latent_factors = latent_factors
        self.item_bias = item_bias
        self.user_bias = user_bias


    def Load(self,fileName):

        load_directory = os.path.join()

