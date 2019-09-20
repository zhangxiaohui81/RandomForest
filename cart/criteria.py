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
        p = counts.astype(np.float) / len(y)
        gini = 1 - np.sum(p ** 2)

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
        lam = sum(map(partial(self.func, avg), y)) / len(y)
        return lam


class MSECriteria(LambdaCriteria):

    def __init__(self):
        self.name = 'MSE'
        self.func = lambda y_pred, y: (y - y_pred) ** 2

class MAECriteria(LambdaCriteria):

    def __init__(self):
        self.name = 'MAE'
        self.func = lambda y_pred, y: abs(y_pred - y)

# class RMAECriteria(LambdaCriteria):
#
#     def __init__(self):
#         self.name = 'RMAE'
        # self.func = lambda


# if __name__ == '__main__':
#     c = np.array([3,4,5,1,2,0,8,1,1])
#     x = GiniCriteria().calculate(c)
#     print('x',x)
#
#     y = EntropyCriteria().calculate(c)
#     print('y',y)
#
#     me = c.mean()
#     s = sum(((c - me) ** 2))
#     print('s',s)
#
#
#     z = MSECriteria().calculate(c)
#     print('z',z)
#
#     k = MAECriteria().calculate(c)
#     print('k', k)

