import numpy as np

class Split: 
    """
    Stores decision tree split information.
    """
    def __init__(self, column, value, column_name=None, column_rf=None):
        self.column = column
        self.column_name = column_name
        self.value = value
        self.column_rf = column_rf
        self.is_object_type = False
        
    def match(self, x):
        if x.iloc[:, self.column].dtype == 'object':
            self.is_object_type = True
            return x.iloc[:, self.column] == self.value
        else:
            return x.iloc[:, self.column] >= self.value

    def match_row(self, row):
        if self.is_object_type:
            if self.column_rf is not None:
                return row[self.column_rf] == self.value
            else:
                return row[self.column] == self.value
        else:
            if self.column_rf is not None:
                return row[self.column_rf] >= self.value
            else:
                return row[self.column] >= self.value

    def __repr__(self):
        # Helper method to print the split info
        if self.is_object_type:
            qualifier = "=="
        else:
            qualifier = ">=" 

        if self.column_name is None:
            return f"{self.column} {qualifier} {str(self.value)}" 
        else:
            return f"{self.column_name} {qualifier} {str(self.value)}"
    
    
class Leaf: 
    """
    Stores decision tree leaf node information.
    """
    def __init__(self, y):
        self.prediction = round(np.mean(y))
        self.probability = np.mean(y)
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
    Implementation of a CART decision tree.
    """
    def __init__(self, max_depth=100, min_samples_leaf=0, column_names=None, column_ids=None,
                classification=True, minimum_gain=0.01, alpha=0.0, max_splits=20, verbose=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.column_names = column_names
        self.column_ids = column_ids
        self.classification = classification
        self.minimum_gain = minimum_gain
        self.alpha = alpha
        self.max_splits = max_splits
        self.verbose = verbose
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
        return [self._predict_row(r, self.tree) for r in x.values.tolist()]
    
    
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

        if self.verbose:
            print(f"gain: {gain} split: {split} \n")

        # Return leaf when no further information gain
        if gain <= self.minimum_gain  or self._current_depth >= self.max_depth:
            return Leaf(y)

        # If we reach here, we have found a useful feature / value to partition on
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
        """Calculate class counts."""
        counts = {}
        for val in np.unique(y):
            counts[val] = np.sum(y==val)
        return counts


    def _gini(self, y, complexity_pruning=False): 
        """Calculate gini impurity."""
        counts = self._class_counts(y)
        gini = 1
        for cls in counts:
            prob_of_cls = np.sum(y==cls)/y.shape[0]
            gini -= prob_of_cls**2
        # Cost complexity adjustment
        if complexity_pruning:
            gini += self.alpha * (2**self._current_depth)
        return gini

    
    def _information_gain(self, y_left, y_right, current_gini): 
        """Calculate information gain."""
        p = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
        return current_gini - p * self._gini(y_left, True) - (1-p) * self._gini(y_right, True)
    

    def _partition(self, x, y, split): 
        """Create lists based on split criteria."""
        idx = split.match(x)

        x_true = x[idx]
        x_false = x[~idx]
        
        y_true = y[idx]
        y_false = y[~idx]
        
        return x_true, x_false, y_true, y_false


    def _mse(self, y, complexity_pruning=False):
        """Calculate the mean squared error."""
        avg = np.mean(y)
        mse = 1/y.shape[0] * np.sum((y - avg)**2)
        # Cost complexity adjustment
        if complexity_pruning:
            mse += self.alpha * (2**self._current_depth)
        return mse


    def _variance_reduction(self, y_left, y_right, current_mse): 
        """Calculate the variance reduction for regression."""
        p = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
        return current_mse - p * self._mse(y_left, True) - (1-p) * self._mse(y_right, True)


    def _reduce_splits(self, x):
        """Reduce the number of splits to speed-up calculations on continuous variables."""
        if x.shape[0] > self.max_splits:
            c = int(x.shape[0]/self.max_splits)
            x_reduced = np.array([t for i, t in enumerate(x) if i%c == 0])
            return x_reduced
        else:
            return x


    def _find_best_split(self, x, y):
        """Find the best variable and value to split on."""
        best_gain = -np.inf
        best_split = None
        if self.classification:
            current_metric = self._gini(y)
        else:
            current_metric = self._mse(y)

        if self.verbose:
            print(f"current metric: {current_metric}")

        for col in range(x.shape[1]):

            if self.verbose:
                if self.column_names is not None:
                    print(f"Finding best split for column {self.column_names[col]}")
                else:
                    print(f"Finding best split for column {self.column_ids[col]}")

            values = np.unique(x.iloc[:, col])

            # Reduce the number of splits to check for continuous variables
            if x.iloc[:, col].dtype != 'object' and self.max_splits is not None:
                values = self._reduce_splits(values)

            for val in values:
                if self.column_names is not None:
                    split = Split(col, val, column_name=self.column_names[col])
                else:
                    split = Split(col, val)

                x_true, x_false, y_true, y_false = self._partition(x, y, split)

                # Skip split if we already have purity
                if len(x_true) == 0 or len(x_false) == 0:
                    if self.verbose:
                        print("No further split as node has purity")
                    continue

                if self.classification:
                    gain = self._information_gain(y_true, y_false, current_metric)
                else:
                    gain = self._variance_reduction(y_true, y_false, current_metric)

                if self.verbose:
                    print(f"value: {val} gain: {gain}")
                if gain >= best_gain:
                    best_gain, best_split = gain, split
            
            if self.verbose:
                print()

        return best_gain, best_split 


    def _print_tree(self, node, spacing=""):
        """Tree printing helper function."""
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "__Leaf Node__")
            if self.classification:
                print(spacing + "predicted class:", node.prediction)
                print(spacing + "class probability:", round(node.probability, 4))
            else:
                print(spacing + "predicted value:", round(node.probability, 2))
            print(spacing + "support:", node.volume)
            return

        # Print the split criteria at this node
        print(spacing + str(node.split))

        # Call this function recursively on the true branch
        print(spacing + '+ True:')
        self._print_tree(node.true_branch, spacing + "|   ")

        # Call this function recursively on the false branch
        print(spacing + '+ False:')
        self._print_tree(node.false_branch, spacing + "    ")   
        
        
    def _predict_row(self, row, node):
        """Predict for a single observation."""

        # Base case: leaf node
        if isinstance(node, Leaf):
            return node.prediction

        # Decide whether to follow the true or false branch.
        # Compare the feature / value stored in the node to the current observation.
        if node.split.match_row(row):
            return self._predict_row(row, node.true_branch)
        else:
            return self._predict_row(row, node.false_branch)

