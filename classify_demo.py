from cart.cart import CartClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)

# 训练鸢尾花分类
# 获取数据
col = ['sepal_length', 'sepal_width', 'petal_length', 'petal width']
data = pd.DataFrame(load_iris().data, columns=col)
data['target'] = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# 分类决策树结果预测
classifier = CartClassifier(criterion="gini",
                            max_depth=8,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features=None,
                            random_state=42,
                            max_leaf_nodes=3,
                            min_impurity_decrease=1e-4,
                            class_weight=None)
classifier.fit(X_train.values, y_train.values)
our_predict_result = classifier.predict(X_test.values)

# sklearn分类决策树预测
clf = DecisionTreeClassifier(criterion="gini",
                             max_depth=8,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features=None,
                             random_state=42,
                             max_leaf_nodes=3,
                             min_impurity_decrease=1e-4,
                             class_weight=None)
clf.fit(X_train,y_train)
skl_predict_result = clf.predict(X_test)

# 各自预测结果输出对比
df = list(zip(our_predict_result, y_test, skl_predict_result))
col = ['our_predict_result', 'y_test', 'skl_predict_result']
result = pd.DataFrame(df, columns=col).T
print(result)
