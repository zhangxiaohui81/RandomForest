import numpy as np
from functools import partial

class Criteria:

    def calculate(self, y:np.ndarray):
        raise NotImplemented


class GiniCriteria(Criteria):

    def __init__(self):
        self.name = 'gini'

    def calculate(self, y:np.ndarray):
        unique, counts = np.unique(y.astype(np.int32), return_counts=True)
        gini = 1 - np.sum((counts.astype(np.float) / len(y)) ** 2)

        return gini

class EntropyCriteria(Criteria):

    def __init__(self):
        self.name = 'entropy'

    def calculate(self, y:np.ndarray):
        unique, counts = np.unique(y.astype(np.int32), return_counts=True)
        p = counts.astype(np.float) / len(y)
        entropy = - np.sum(p * np.log2(p))

        return entropy


class LambdaCriteria(Criteria):

    def __init__(self, func=None):
        self.name = 'lambda'
        self.func = func

    def calculate(self, y:np.ndarray):
        avg = y.mean()
        return sum(map(y, partial(self.func,avg)))


class MSECriteria(LambdaCriteria):

    def __init__(self):
        self.name = 'MSE'
        self.func = lambda y_pred, y: (y_pred - y) ** 2

class MAECriteria(LambdaCriteria):

    def __init__(self):
        self.name = 'MAE'
        self.func = lambda y_pred, y: abs(y_pred - y)

# class RMAECriteria(LambdaCriteria):
#
#     def __init__(self):
#         self.name = 'RMAE'
        # self.func = lambda


if __name__ == '__main__':
    c = np.array([3,4,5,1,2,0,8,1,1])
    x = GiniCriteria().calculate(c)
    print(x)

    y = EntropyCriteria().calculate(c)
    print(y)
