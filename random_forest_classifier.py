import numpy as np
from scipy.stats import mode


class RandomForestClassifier:
    """
    Implementation of a random forest classifier.
    - Uses DecisionTree class
    """

    def __init__(self, n_estimators=100, max_depth=100, min_samples_leaf=1, col_percent=0.7, row_percent=1.0,
                 use_random_splits=False, random_seed=42):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.col_percent = col_percent
        self.row_percent = row_percent
        self.use_random_splits = use_random_splits
        self.random_seed = random_seed
        self.estimators = []

    def fit(self, x, y):
        """
        Fit the random forest model
        """
        random_seed = self.random_seed
        for estimator_i in range(self.n_estimators):
            # Set random seed
            np.random.seed(random_seed)

            # Create datasets
            x_sample, y_sample, col_ids = self.create_samples(x, y)

            # Fit estimator
            estimator = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     use_random_splits=self.use_random_splits, column_ids=col_ids)
            estimator.fit(x_sample, y_sample)
            self.estimators.append(estimator)

            random_seed += 1

    def create_samples(self, x, y):
        """
        Create random samples
        """

        p = np.random.permutation(x.shape[0])
        x = x[p]
        y = y[p]

        n_cols_sample = int(x.shape[1] * self.col_percent)
        sample_cols = np.random.choice(x.shape[1], n_cols_sample, replace=False)
        sample_rows = int((x.shape[0] * self.row_percent))

        return x[:sample_rows, sample_cols], y[:sample_rows], sample_cols

    def predict(self, x):
        """
        Predict on new data
        """

        predictions = np.zeros((x.shape[0], len(self.estimators)), dtype=np.int8)

        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(x)

        return mode(predictions, axis=1)[0].flatten()
