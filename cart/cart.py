import numpy as np


class ClassificationAndRegressionTree:
    ''''''

    def __init__(self,
                 criterion = 'gini',
                 splitter = 'best',
                 max_depth = None,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 min_weight_fraction_leaf = 0.0,
                 max_features = None,
                 random_state = None,
                 max_leaf_nodes = None,
                 min_impurity_decrease = 0.0,
                 min_impurity_split = None,
                 class_weight = None,
                 # presort = False
                 ):

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


