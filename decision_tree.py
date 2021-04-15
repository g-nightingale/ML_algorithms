import numpy as np


class Split:
    """
    Stores decision tree split information.
    """

    def __init__(self, column, value, column_rf=None):
        self.column = column
        self.value = value
        self.column_rf = column_rf

    def match(self, x):
        return x[:, self.column] >= self.value

    def match_row(self, x):
        if self.column_rf is not None:
            return x[self.column_rf] >= self.value
        else:
            return x[self.column] >= self.value

    def __repr__(self):
        # Helper method to print the split info
        condition = ">="
        return "Is %s %s %s?" % (self.column, condition, str(self.value))


class Leaf:
    """
    Stores decision tree leaf node information.
    """

    def __init__(self, y):
        self.predictions = round(y.sum() / y.shape[0])
        self.probability = y.sum() / y.shape[0]
        self.volume = y.shape[0]


class DecisionNode:
    """
    Store split criteria and reference to the child nodes.
    """

    def __init__(self,
                 split,
                 true_branch,
                 false_branch):
        self.split = split
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree:
    """
    Implementation of a decision tree.
    """

    def __init__(self, max_depth=100, min_samples_leaf=0, use_random_splits=False, column_ids=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.use_random_splits = use_random_splits
        self.column_ids = column_ids
        self.tree = None
        self._current_depth = -1

    def fit(self, x, y):
        """
        Fit decision tree.
        """
        self.tree = None
        self._current_depth = -1
        self.tree = self._build_tree(x, y)

    def predict(self, x):
        """
        Predict on new data.
        """
        return [self._predict_row(r, self.tree) for r in x]

    def print_tree(self):
        """
        Print the decision tree.
        """
        self._print_tree(self.tree)

    def _build_tree(self, x, y):
        """
        Builds the decision tree.
        """
        self._current_depth += 1

        # Find best split at this node
        gain, split = self._find_best_split(x, y)

        # Return leaf when no further information gain
        if gain == 0 or self._current_depth >= self.max_depth:
            return Leaf(y)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        x_true, x_false, y_true, y_false = self._partition(x, y, split)
        if x_true.shape[0] < self.min_samples_leaf or x_false.shape[0] < self.min_samples_leaf:
            return Leaf(y)

            # Recursively build the true branch
        true_branch = self._build_tree(x_true, y_true)

        # Recursively build the false branch
        false_branch = self._build_tree(x_false, y_false)

        # Return a Decision Node
        return DecisionNode(split, true_branch, false_branch)

    def _class_counts(self, y):
        """Return class counts"""
        counts = {}
        for val in np.unique(y):
            counts[val] = np.sum(y == val)
        return counts

    def _partition(self, x, y, split):
        """Create lists based on split criteria"""

        idx = split.match(x)

        x_true = x[idx]
        x_false = x[~idx]

        y_true = y[idx]
        y_false = y[~idx]

        return x_true, x_false, y_true, y_false

    def _gini(self, y):
        """Return gini impurity"""
        counts = self._class_counts(y)
        impurity = 1
        for cls in counts:
            prob_of_cls = np.sum(y == cls) / y.shape[0]
            impurity -= prob_of_cls ** 2
        return impurity

    def _information_gain(self, y_left, y_right, current_gini):
        """Return information gain"""
        p = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
        return current_gini - p * self._gini(y_left) - (1 - p) * self._gini(y_right)

    def _find_best_split(self, x, y):
        """Find the best variable and value to split on"""
        best_gain = 0
        best_split = None
        current_gini = self._gini(y)

        for col in range(x.shape[1]):
            values = np.unique(x[:, col])
            if self.use_random_splits is True:
                val = random.choice(values)
                if self.column_ids is None:
                    split = Split(col, val)
                else:
                    split = Split(col, val, self.column_ids[col])

                x_true, x_false, y_true, y_false = self._partition(x, y, split)

                # Skip split if it doesn't divide the dataset
                if len(x_true) == 0 or len(x_false) == 0:
                    continue

                gain = self._information_gain(y_true, y_false, current_gini)
                if gain >= best_gain:
                    best_gain, best_split = gain, split
            else:
                for val in values:
                    if self.column_ids is None:
                        split = Split(col, val)
                    else:
                        split = Split(col, val, self.column_ids[col])

                    x_true, x_false, y_true, y_false = self._partition(x, y, split)

                    # Skip split if it doesn't divide the dataset
                    if len(x_true) == 0 or len(x_false) == 0:
                        continue

                    gain = self._information_gain(y_true, y_false, current_gini)
                    if gain >= best_gain:
                        best_gain, best_split = gain, split

        return best_gain, best_split

    def _print_tree(self, node, spacing=""):
        """Tree printing helper function"""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            print(spacing + "Probability", node.probability)
            print(spacing + "Volume", node.volume)
            return

        # Print the split criteria at this node
        print(spacing + str(node.split))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self._print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self._print_tree(node.false_branch, spacing + "  ")

    def _predict_row(self, row, node):
        """Predict for a single observation"""

        # Base case: leaf node
        if isinstance(node, Leaf):
            return node.predictions

        # Decide whether to follow the true or false branch.
        # Compare the feature / value stored in the node to the current observation.
        if node.split.match_row(row):
            return self._predict_row(row, node.true_branch)
        else:
            return self._predict_row(row, node.false_branch)
