from cart.cart import CartRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)
from time import *

# 训练糖尿病回归
col = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
data = pd.DataFrame(load_diabetes().data, columns=col)
data['target'] = load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# 回归决策树结果预测
our_model = CartRegression(max_depth=8,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=42,
                            max_leaf_nodes=6,
                            min_impurity_decrease=1e-4,)

our_begin_time_fit = time()
our_model.fit(X_train.values, y_train.values)
our_end_time_fit = time()

our_begin_time_predict = time()
our_predict_result = our_model.predict(X_test.values)
our_end_time_predict = time()

# sklearn回归决策树预测
skl_model = DecisionTreeRegressor(max_depth=8,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=42,
                            max_leaf_nodes=6,
                            min_impurity_decrease=1e-4,)
skl_begin_time_fit = time()
skl_model.fit(X_train, y_train)
skl_end_time_fit = time()

skl_begin_time_predict = time()
skl_predict_result = skl_model.predict(X_test)
skl_end_time_predict = time()

# 各自预测结果输出对比
our_predict_result_variance = sum(((y_test - our_predict_result) ** 2)) / len(y_test)
skl_predict_result_variance = sum(((y_test - skl_predict_result) ** 2)) / len(y_test)

our_train_variance = sum(((y_train - our_model.predict(X_train.values)) ** 2)) / len(y_train)
skl_train_variance = sum(((y_train - skl_model.predict(X_train.values)) ** 2)) / len(y_train)

our_fit_time = our_end_time_fit - our_begin_time_fit
skl_fit_time = skl_end_time_fit - skl_begin_time_fit

our_predict_time = our_end_time_predict - our_begin_time_predict
skl_predict_time = skl_end_time_predict - skl_begin_time_predict

df = [[our_predict_result_variance, skl_predict_result_variance],
      [our_train_variance, skl_train_variance],
      [our_fit_time, skl_fit_time],
      [our_predict_time, skl_predict_time]]
col = ['our_value', 'skl_value']
index = ['predict_result_variance', 'train_variance', 'fit_time', 'predict_time']
result = pd.DataFrame(df, columns=col, index=index)

print(result)