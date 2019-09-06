import numpy as np
from .tree import Walker, Node, Tree
from .splitter import Splitter
from .criteria import Criteria, GiniCriteria, EntropyCriteria, LambdaCriteria, MSECriteria, MAECriteria

class ClassificationAndRegressionTree:
    ''''''

    def __init__(self,
                 criterion = 'gini',
                 # splitter = 'best',
                 max_depth = 8,
                 min_samples_split = 2,
                 min_samples_leaf = 1,
                 min_weight_fraction_leaf = 0.0,
                 max_features = None,
                 random_state = 42,
                 max_leaf_nodes = 20,
                 min_impurity_decrease = 1e-4,
                 min_impurity_split = 2e-4,
                 # class_weight = None,
                 # presort = False
                 ):

        if criterion == 'gini':
            splitter_criteria = GiniCriteria()
        elif criterion == 'entropy':
            splitter_criteria = EntropyCriteria()
        elif criterion == 'mse':
            splitter_criteria = MSECriteria()
        elif criterion == 'mae':
            splitter_criteria = MAECriteria()
        else:
            raise NotImplemented

        self.max_depth = max_depth or 8
        self.min_samples_split = min_samples_split or 2
        self.min_samples_leaf = min_samples_leaf or 1
        self.min_weight_fraction_leaf = min_weight_fraction_leaf or 0.0
        self.max_features = max_features
        self.random_state = random_state or 42
        self.max_leaf_nodes = max_leaf_nodes or 20
        self.min_impurity_decrease = min_impurity_decrease or 1e-4
        self.min_impurity_split = min_impurity_split or 2e-4
        # self.class_weight = class_weight

        self.splitter = Splitter(splitter_criteria, self.min_impurity_decrease, self.min_impurity_split,
                                 self.min_samples_split, self.min_samples_leaf)

    def should_check_leaf(self, leaf:Node) -> bool:
        if leaf.level >= self.max_depth - 1:
            return False
        elif len(leaf.row_indexes) <= self.min_samples_split:
            return False
        elif leaf.impurity <= self.min_impurity_split:
            return False
        else:
            return True

    def should_split_hold(self,node:Node, left:Node, right:Node) -> bool:
        if len(left.row_indexes) < self.min_samples_leaf or len(right.row_indexes) < self.min_samples_leaf:
            return False
        elif node.impurity - left.impurity - right.impurity < self.min_impurity_decrease:
            return False
        else:
            return True

    def fit(self, X:np.ndarray, y:np.ndarray):
        assert len(X)==len(y), "sample counts of X and y do not match"
        tr = Tree()
        tr.add_root(row_indexes=list(range(X.shape[0])), feature_indexes=list(range(X.shape[1])),
                    impurity=self.splitter.criteria.calculate(y))
        leaves_metric = {}
        while tr.leaf_count()<self.max_leaf_nodes:
            for leaf in tr.leaves:
                if self.should_check_leaf(leaf) and id(leaf) not in leaves_metric:
                    fidx, cut_point, parts = self.splitter.split(X, y, feature_indexes=leaf.feature_indexes,
                                                                 row_indexes=leaf.row_indexes, whole_impurity=leaf.impurity)

                    left = Node(parent=id(leaf), level=leaf.level + 1, row_indexes=parts[0][0],
                                feature_indexes=leaf.feature_indexes, impurity=parts[0][1])
                    right = Node(parent=id(leaf), level=leaf.level + 1, row_indexes=parts[1][0],
                                 feature_indexes=leaf.feature_indexes, impurity=parts[1][1])

                    leaves_metric[id(leaf)] = (leaf, fidx, cut_point, (left,right), leaf.impurity-parts[0][1]-parts[1][1])



            leaves_metric = { leaf_id : metric
                              for leaf_id,metric in leaves_metric.items()
                              if self.should_split_hold(metric[0], metric[3][0], metric[3][2]) }

            if len(leaves_metric)==0:
                break

            leaf, fidx, cut_point, parts, _ = max(leaves_metric.values(), key=lambda x: x[4])
            walker = Walker(feature_index=fidx, dividing_line=cut_point)
            tr.add_children_for_node(node=leaf, left=left, right=right, walker=walker)

        self.tree = tr
        self.feature_count = X.shape[1]

    def fitted(self) -> bool:
        return hasattr(self, "tree")


    # def predict(self, X):
    #     pass
    #
    # def predict_proba(self, X):
    #     pass

class CartClassifier(ClassificationAndRegressionTree):

    def __init__(self,
                 criterion='gini',
                 max_depth=8,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=None,
                 random_state=42,
                 max_leaf_nodes=20,
                 min_impurity_decrease=1e-4,
                 min_impurity_split=2e-4,
                 class_weight = None
                 ):
        assert criterion in ['gini', 'entropy']
        super(CartClassifier, self).__init__(criterion = criterion,
                                             max_depth = max_depth,
                                             min_samples_split = min_samples_split,
                                             min_samples_leaf = min_samples_leaf,
                                             min_weight_fraction_leaf = min_weight_fraction_leaf,
                                             max_features = max_features,
                                             random_state = random_state,
                                             max_leaf_nodes = max_leaf_nodes,
                                             min_impurity_decrease = min_impurity_decrease,
                                             min_impurity_split = min_impurity_split,)
        self.class_weight = class_weight

    def fit(self, X:np.ndarray, y:np.ndarray):
        super(CartClassifier, self).fit(X, y)
        unique, counts = np.unique(y.astype(np.int32), return_counts=True)
        self.class_count = len(unique)
        total = len(y)

        if self.class_weight is None:
            class_weight_value = [1] * self.class_count
        elif self.class_weight == 'balanced':
            class_weight_value = [x / total for x in counts]
        elif isinstance (self.class_weight, dict):
            assert set(self.class_weight.keys()) == set(unique)
            class_weight_value = [i[1] for i in sorted(list(self.class_weight.items()), key=lambda x:x[0])]
        else:
            raise Exception("The value of class_weight(%s) is not supported" %str(self.class_weight))

        class_dic = {unique[i]: 0 for i in range(len(unique))}
        for leaf in self.tree.leaves:
            leaf_p = class_dic.copy()
            unique_leaf, counts_leaf = np.unique(y[leaf.row_indexes].astype(np.int32), return_counts=True)
            leaf_dic = {unique_leaf[i]:counts_leaf[i] for i in range(len(unique_leaf))}
            leaf_p.update(leaf_dic)
            class_weight_leaf = [i[1] for i in sorted(list(leaf_p.items()), key=lambda x: x[0])]
            weight = [w1 / w2 for w1, w2 in zip(class_weight_leaf, class_weight_value)]
            weight_sum = sum(weight)
            normalized = [i / weight_sum for i in weight]
            leaf.value = np.array(normalized)


    def predict(self, X:np.ndarray):
        assert self.fitted()

        predict_value = self.predict_proba(X)

        return np.eye(self.class_count)[predict_value.argmax(axis=1)]



    def predict_proba(self, X:list):
        assert self.fitted()

        predict_proba_value = []

        n = self.tree.root

        for e in X:
            while not n.is_leaf():

                if n.walker.should_go_left(e):
                    n=n.left
                else:
                    n=n.right

            predict_proba_value.append(n.value)

        return np.ndarray(predict_proba_value)


    # 特征重要性