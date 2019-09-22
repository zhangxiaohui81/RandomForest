from .utils import Visitor


class Walker:

    def __init__(self, feature_index, dividing_line, go_left_when_equal=True):
        self.feature_index = feature_index
        self.dividing_line = dividing_line #
        self.go_left_when_equal = go_left_when_equal

    def should_go_left(self, x):
        if self.dividing_line < x[self.feature_index]:
            return False
        elif self.dividing_line > x[self.feature_index]:
            return True
        else:
            return self.go_left_when_equal


class Node:

    def __init__(self, parent: int=-1, left: 'Node'=None, right: 'Node'=None,
                 level: int=0, row_indexes: [int]=None, feature_indexes: [int]=None, impurity: float=-1):
        self.parent = parent
        self.left = left
        self.right = right
        self.level = level
        self.row_indexes = row_indexes
        self.walker = None
        self.feature_indexes = feature_indexes
        self.impurity = impurity
        self.value = None

    def is_leaf(self):
        return self.left is None and self.right is None


class Tree:

    def __init__(self):
        self.root = None
        self.leaves = []
        self.id_to_node = {}

    def add_root(self, row_indexes: [int], feature_indexes: [int], impurity):
        self.root = Node(row_indexes=row_indexes, feature_indexes=feature_indexes, impurity=impurity)
        self.leaves = [self.root, ]
        self.id_to_node[id(self.root)] = self.root

    def leaf_count(self):
        return len(self.leaves)

    def height(self):
        return max([leaf.level for leaf in self.leaves]) + 1

    def empty(self):
        return self.root is None

    def add_children_for_node(self, node: Node, left: Node, right: Node, walker: Walker):
        left.parent = id(node)
        right.parent = id(node)
        node.left = left
        node.right = right
        node.walker = walker
        self.leaves.remove(node)
        self.leaves.extend([left, right])
        self.id_to_node[id(left)] = left
        self.id_to_node[id(right)] = right

    def nodes(self):
        ns=[self.root]
        while ns:
            n = ns[0]
            del ns[0]
            if n.left is not None:
                ns.append(n.left)
            if n.right is not None:
                ns.append(n.right)
            yield n

    def accept(self, visitor: Visitor):
        for n in self.nodes():
            visitor.visit(node=n)


class CleanVisitor(Visitor):
    def visit(self, node: Node):
        self.clean(node)

    def clean(self, node: Node):
        del node.row_indexes
        del node.impurity
        del node.feature_indexes
