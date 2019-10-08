import numpy as np
from .tree import Walker, Node, Tree, CleanVisitor
from .splitter import Splitter
from .criteria import Criteria, GiniCriteria, EntropyCriteria, LambdaCriteria, MSECriteria, MAECriteria
from .sampler import SimpleSampler


class ClassificationAndRegressionTree:
    """Base class for decision trees.
    """

    def __init__(self,
                 criterion="gini",
                 max_depth=8,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=42,
                 max_leaf_nodes=20,
                 min_impurity_decrease=1e-4,
                 min_impurity_split=2e-7):
        self.max_depth = max_depth or 8
        self.min_samples_split = min_samples_split or 2
        self.min_samples_leaf = min_samples_leaf or 1
        self.max_features = max_features
        self.random_state = random_state or 42
        self.max_leaf_nodes = max_leaf_nodes or 20
        self.min_impurity_decrease = min_impurity_decrease or 1e-4
        self.min_impurity_split = min_impurity_split or 2e-4
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

        self.feature_sampler = SimpleSampler(self.random_state, quantity=max_features)

        self.splitter = Splitter(splitter_criteria, self.min_impurity_decrease, self.min_impurity_split,
                                 self.min_samples_split, self.min_samples_leaf)

    def should_check_leaf(self, leaf: Node) -> bool:
        if leaf.level >= self.max_depth - 1:
            return False
        elif len(leaf.row_indexes) <= self.min_samples_split:
            return False
        elif leaf.impurity <= self.min_impurity_split:
            return False
        else:
            return True

    def should_split_hold(self, node: Node, left: Node, right: Node) -> bool:
        if len(left.row_indexes) < self.min_samples_leaf or len(right.row_indexes) < self.min_samples_leaf:
            return False
        elif node.impurity - left.impurity - right.impurity < self.min_impurity_decrease:
            return False
        else:
            return True

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Constructing decision tree.
        """
        assert len(X) == len(y), "sample counts of X and y do not match"
        tr = Tree()
        tr.add_root(row_indexes=list(range(X.shape[0])), feature_indexes=list(range(X.shape[1])),
                    impurity=self.splitter.criteria.calculate(y))

        leaves_metric = {}
        while tr.leaf_count() < self.max_leaf_nodes:

            for leaf in tr.leaves:
                if self.should_check_leaf(leaf) and id(leaf) not in leaves_metric:
                    features_to_consider = self.feature_sampler.sample(leaf.feature_indexes)
                    fidx, cut_point, parts, impu_dec = self.splitter.split(X, y, feature_indexes=features_to_consider,
                                                                           row_indexes=leaf.row_indexes,
                                                                           whole_impurity=leaf.impurity)

                    left = Node(parent=id(leaf), level=leaf.level + 1, row_indexes=parts[0][0],
                                feature_indexes=leaf.feature_indexes, impurity=parts[0][1])
                    right = Node(parent=id(leaf), level=leaf.level + 1, row_indexes=parts[1][0],
                                 feature_indexes=leaf.feature_indexes, impurity=parts[1][1])

                    leaves_metric[id(leaf)] = (leaf, fidx, cut_point, (left, right), impu_dec * len(leaf.row_indexes))

            if len(leaves_metric) == 0:
                break

            leaf, fidx, cut_point, parts, _ = max(leaves_metric.values(), key=lambda x: x[4])
            walker = Walker(feature_index=fidx, dividing_line=cut_point)
            tr.add_children_for_node(node=leaf, left=parts[0], right=parts[1], walker=walker)
            del leaves_metric[id(leaf)]

        self.tree = tr
        self.feature_count = X.shape[1]

    def fitted(self) -> bool:
        return hasattr(self, "tree")


class CartClassifier(ClassificationAndRegressionTree):
    """A decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth : int, optional (default=8)
        The maximum depth of the tree.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    max_features ： int, float, string or None, optional (default="None")
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then max_features is a percentage
              and int(max_features * n_features) features are considered at each split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            - If "all", then `max_features=n_features`.

    random_state ： int, optional (default=42)
        If int, random_state is the seed used by the random number generator;

    max_leaf_nodes : int, optional (default=20)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.

    min_impurity_decrease : float, optional (default=1e-4)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N * (whole_impurity - N_R / N_t * right_impurity - N_L / N * left_impurity)

        where ``N`` is the number of samples at the current node,
        ``N_L`` is the number of samples in the left child,
        and ``N_R`` is the number of samples in the right child.

        ``N``, ``N_R`` and ``N_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_impurity_split : float, (default=2e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    class_weight ： dict, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" model uses the value of y to automatically adjust
        weights, and the frequency of categories in the input data is inversely
        proportional to the total sample size::
        ``np.bincount(y) / n_samples``
    """

    def __init__(self,
                 criterion="gini",
                 max_depth=8,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=42,
                 max_leaf_nodes=20,
                 min_impurity_decrease=1e-4,
                 min_impurity_split=2e-7,
                 class_weight=None):
        assert criterion in ["gini", "entropy"]

        super(CartClassifier, self).__init__(criterion=criterion,
                                             max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf,
                                             max_features=max_features,
                                             random_state=random_state,
                                             max_leaf_nodes=max_leaf_nodes,
                                             min_impurity_decrease=min_impurity_decrease,
                                             min_impurity_split=min_impurity_split, )
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        super(CartClassifier, self).fit(X, y)
        unique, counts = np.unique(y.astype(np.int32), return_counts=True)
        self.class_count = len(unique)
        total = len(y)

        if self.class_weight is None:
            class_weight_value = [1] * self.class_count
        elif self.class_weight == 'balanced':
            class_weight_value = [x / total for x in counts]
        elif isinstance(self.class_weight, dict):
            assert set(self.class_weight.keys()) == set(unique)
            class_weight_value = [i[1] for i in sorted(list(self.class_weight.items()), key=lambda x: x[0])]
        else:
            raise Exception("The value of class_weight(%s) is not supported" % str(self.class_weight))

        class_dic = {unique[i]: 0 for i in range(len(unique))}
        for leaf in self.tree.leaves:
            leaf_p = class_dic.copy()
            unique_leaf, counts_leaf = np.unique(y[leaf.row_indexes].astype(np.int32), return_counts=True)
            leaf_dic = {unique_leaf[i]: counts_leaf[i] for i in range(len(unique_leaf))}
            leaf_p.update(leaf_dic)
            class_weight_leaf = [i[1] for i in sorted(list(leaf_p.items()), key=lambda x: x[0])]
            weight = [w1 / w2 for w1, w2 in zip(class_weight_leaf, class_weight_value)]
            weight_leaf_sum = sum(weight)
            normalized = [i / weight_leaf_sum for i in weight]
            leaf.value = np.array(normalized)

        self.tree.accept(CleanVisitor())

    def predict(self, X: np.ndarray):
        """Predict class or regression value for X.

        Parameters
        ----------
        x : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to ``dtype=np.float64``.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted values.
        """
        assert self.fitted()

        predict_value = self.predict_proba(X)

        return predict_value.argmax(axis=1)

    def predict_proba(self, X: np.ndarray):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        x : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes].
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute class.
        """
        assert self.fitted()

        predict_proba_value = []

        for e in X:
            n = self.tree.root
            while not n.is_leaf():

                if n.walker.should_go_left(e):
                    n = n.left
                else:
                    n = n.right

            predict_proba_value.append(n.value)

        return np.array(predict_proba_value)


class CartRegressor(ClassificationAndRegressionTree):

    def __init__(self,
                 criterion='mse',
                 max_depth=8,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=42,
                 max_leaf_nodes=20,
                 min_impurity_decrease=1e-4,
                 min_impurity_split=2e-7):
        assert criterion in ['mse', 'mae']
        super(CartRegressor, self).__init__(criterion=criterion,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=random_state,
                                            max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_impurity_split=min_impurity_split)

    def fit(self, X: np.ndarray, y: np.ndarray):

        super(CartRegressor, self).fit(X, y)

        for leaf in self.tree.leaves:
            leaf.value = y[leaf.row_indexes].mean()

        self.tree.accept(CleanVisitor())

    def predict(self, X: np.ndarray):
        assert self.fitted()

        predict_value = []

        for e in X:
            n = self.tree.root
            while not n.is_leaf():

                if n.walker.should_go_left(e):
                    n = n.left
                else:
                    n = n.right

            predict_value.append(n.value)

        return np.array(predict_value)
