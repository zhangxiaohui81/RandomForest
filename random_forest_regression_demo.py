from random_forest.random_forest import RandomForestRegressor as my_RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor as skl_RandomForestRegressor
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)

# 训练糖尿病回归
col = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
data = pd.DataFrame(load_diabetes().data, columns=col)
data['target'] = load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# 回归决策树结果预测
regression = my_RandomForestRegressor(n_estimators=20,
                                      criterion="mse",
                                      max_depth=None,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features="auto",
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.,
                                      min_impurity_split=None,
                                      bootstrap=True,
                                      random_state=42)
regression.fit(X_train.values, y_train.values)
our_predict_result = regression.predict(X_test.values)

# sklearn回归决策树预测
reg = skl_RandomForestRegressor(n_estimators=20,
                                criterion="mse",
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.,
                                max_features="auto",
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=None,
                                random_state=42,
                                verbose=0,
                                warm_start=False)
reg.fit(X_train, y_train)
skl_predict_result = reg.predict(X_test)

# 各自预测结果输出对比
our_predict_result_variance = sum(((y_test - our_predict_result) ** 2)) / len(y_test)
skl_predict_result_variance = sum(((y_test - skl_predict_result) ** 2)) / len(y_test)

our_train_variance = sum(((y_train - regression.predict(X_train.values)) ** 2)) / len(y_train)
skl_train_variance = sum(((y_train - reg.predict(X_train.values)) ** 2)) / len(y_train)

df = [[our_predict_result_variance, skl_predict_result_variance], [our_train_variance, skl_train_variance]]

col = ['our_value', 'skl_value']
index = ['predict_result_variance', 'train_variance']
result = pd.DataFrame(df, columns=col, index=index)
print(result)
