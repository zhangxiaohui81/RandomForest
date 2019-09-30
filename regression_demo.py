from cart.cart import CartRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)

# 训练糖尿病回归
col = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
data = pd.DataFrame(load_diabetes().data, columns=col)
data['target'] = load_diabetes().target
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# 回归决策树结果预测
regression = CartRegressor(max_depth=8,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features=None,
                            random_state=42,
                            max_leaf_nodes=6,
                            min_impurity_decrease=2e-7)
regression.fit(X_train.values, y_train.values)
our_predict_result = regression.predict(X_test.values)

# sklearn回归决策树预测
reg = DecisionTreeRegressor(max_depth=8,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features=None,
                            random_state=42,
                            max_leaf_nodes=6,
                            min_impurity_decrease=1e-4,)
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
