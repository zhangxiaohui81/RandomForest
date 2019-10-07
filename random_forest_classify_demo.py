from random_forest.random_forest import RandomForestClassifier as my_RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier as skl_RandomForestClassifier
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)
from time import *

# 训练鸢尾花分类
# 获取数据
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal width']
data = pd.DataFrame(load_iris().data, columns=col)
data['target'] = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# 分类决策树结果预测
classifier = my_RandomForestClassifier(n_estimators=10,
                                       criterion="gini",
                                       max_depth=None,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       max_features="auto",
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.,
                                       min_impurity_split=None,
                                       bootstrap=True,
                                       random_state=42,
                                       class_weight=None)

our_begin_time_fit = time()
classifier.fit(X_train.values, y_train.values)
our_end_time_fit = time()

our_begin_time_predict = time()
our_predict_result = classifier.predict(X_test.values)
our_end_time_predict = time()

# sklearn分类决策树预测
clf = skl_RandomForestClassifier(n_estimators=10,
                                 criterion="gini",
                                 max_depth=None,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features='auto',
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 bootstrap=True,
                                 oob_score=False,
                                 n_jobs=1,
                                 random_state=42,
                                 verbose=0,
                                 warm_start=False,
                                 class_weight=None)

skl_begin_time_fit = time()
clf.fit(X_train,y_train)
skl_end_time_fit = time()

skl_begin_time_predict = time()
skl_predict_result = clf.predict(X_test)
skl_end_time_predict = time()

# 各自预测结果输出对比
df = list(zip(our_predict_result, y_test, skl_predict_result))
col = ['our_predict_result', 'y_test', 'skl_predict_result']
result = pd.DataFrame(df, columns=col).T

print(result)
