from cart.sampler import BootstrapSampler, SimpleSampler
from cart.cart import CartClassifier
from cart.cart import CartRegressor
import numpy as np
import pandas as pd
from random import Random


class RandomForestClassifier():
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 random_state=None,
                 class_weight=None):
        self.n_estimators = n_estimators,
        self.criterion = criterion,
        self.max_depth = max_depth,
        self.min_samples_split = min_samples_split,
        self.min_samples_leaf = min_samples_leaf,
        self.max_features = max_features,
        self.max_leaf_nodes = max_leaf_nodes,
        self.min_impurity_decrease = min_impurity_decrease,
        self.min_impurity_split = min_impurity_split,
        self.bootstrap = bootstrap,
        self.random_state = random_state,
        self.class_weight = class_weight

        if self.bootstrap:
            self.sample_sampler = BootstrapSampler(self.random_state, quantity='all')
        else:
            self.sample_sampler = SimpleSampler(self.random_state, quantity='all')

        self.estimators = []

    def fit(self, X:np.ndarray, y:np.ndarray):

        row_indexes = list(range(len(X)))

        estimators = []
        random = Random(self.random_state)
        for i in range(self.n_estimators):
            est = CartClassifier(criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 max_features=self.max_features,
                                 random_state=random.randint(0,10000000),
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 min_impurity_split=self.min_impurity_split,
                                 class_weight=self.class_weight)
            sampled_row_indexes = self.sample_sampler.sample(row_indexes)
            sampled_X = X[sampled_row_indexes]
            sampled_y = y[sampled_row_indexes]
            est.fit(sampled_X, sampled_y)
            estimators.append(est)

        self.estimators = estimators
        return self

    def predict(self, X: np.ndarray):
        predict_value = self.predict_proba(X)

        return predict_value.argmax(axis=1)

    def predict_proba(self, X: np.ndarray):

        predict_proba_value = []

        for e in X:
            predict_proba_sample = []

            for est in self.estimators:
                predict_proba_sample.append(est.predict_proba(e))

            predict_proba_value.append(np.array(predict_proba_sample).mean(axis=0))

        return np.array(predict_proba_value)


class RandomForestRegressor:
    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 # min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 # oob_score=False,
                 random_state=None):
        self.n_estimators = n_estimators,
        self.criterion = criterion,
        self.max_depth = max_depth,
        self.min_samples_split = min_samples_split,
        self.min_samples_leaf = min_samples_leaf,
        # self.min_weight_fraction_leaf = min_weight_fraction_leaf,
        self.max_features = max_features,
        self.max_leaf_nodes = max_leaf_nodes,
        self.min_impurity_decrease = min_impurity_decrease,
        self.min_impurity_split = min_impurity_split,
        self.bootstrap = bootstrap,
        # self.oob_score = oob_score,
        self.random_state = random_state,
        # self.verbose = verbose,
        # self.warm_start = warm_start

        if self.bootstrap:
            self.sample_sampler = BootstrapSampler(self.random_state, quantity='all')
        else:
            self.sample_sampler = SimpleSampler(self.random_state, quantity='all')

        self.estimators = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        row_indexes = list(range(len(X)))

        estimators = []
        random = Random(self.random_state)
        for i in range(self.n_estimators):
            est = CartRegressor(criterion=self.criterion,
                                 max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 max_features=self.max_features,
                                 random_state=random.randint(0, 10000000),
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 min_impurity_split=self.min_impurity_split,
                                 class_weight=self.class_weight)
            sampled_row_indexes = self.sample_sampler.sample(row_indexes)
            sampled_X = X[sampled_row_indexes]
            sampled_y = y[sampled_row_indexes]
            est.fit(sampled_X, sampled_y)
            estimators.append(est)

        self.estimators = estimators
        return self

    def predict(self, X: np.ndarray):
        predict_value = []

        for e in X:
            predict_sample = []

            for est in self.estimators:
                predict_sample.append(est.predict_proba(e))

            predict_value.append(np.array(predict_sample).mean())

        return np.array(predict_value)

