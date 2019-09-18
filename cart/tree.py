
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

    def __init__(self, parent:int=-1, left:'Node'=None, right:'Node'=None,
                 level:int=0, row_indexes:[int]=None, feature_indexes:[int]=None, impurity:float=-1):
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

    def add_root(self, row_indexes:[int], feature_indexes:[int], impurity):
        self.root = Node(row_indexes=row_indexes, feature_indexes=feature_indexes, impurity=impurity)
        self.leaves = [self.root, ]
        self.id_to_node[id(self.root)] = self.root

    def leaf_count(self):
        return len(self.leaves)

    def height(self):
        return max([leaf.level for leaf in self.leaves])+1

    def empty(self):
        return self.root is None

    def add_children_for_node(self, node:Node, left:Node, right:Node, walker:Walker):
        left.parent = id(node)
        right.parent = id(node)
        node.left = left
        node.right = right
        node.walker = walker
        self.leaves.remove(node)
        self.leaves.extend([left, right])
        self.id_to_node[id(left)] = left
        self.id_to_node[id(right)] = right


