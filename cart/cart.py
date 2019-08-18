import numpy as np
from .tree import Walker, Node, Tree
from .splitter import Splitter
from .criteria import Criteria, GiniCriteria, EntropyCriteria, LambdaCriteria, MSECriteria, MAECriteria

class ClassificationAndRegressionTree:
    ''''''

    def __init__(self,
                 criterion = 'gini',
                 # splitter = 'best',
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

    def

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


