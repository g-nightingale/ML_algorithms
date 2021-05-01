import numpy as np


class LogisticRegression:
    """
    Class to perform logistic regression
    """

    def __init__(self, num_iterations=1000, learning_rate=0.01, verbose=False, verbose_n=100):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.verbose_n = verbose_n
        self.w = None
        self.b = None

    def fit(self, x, y):
        """
        Fit logistic regression model
        """

        # Handle rank 1 arrays
        x = self._reshape(x)
        y = self._reshape(y)

        # Initialise weights
        self.w = np.zeros((x.shape[1], 1))
        self.b = 0

        # Gradient descent
        self.gradient_descent(x, y, self.w, self.b)

    def predict(self, x):
        """"
        Predict on new data
        """

        # Handle rank 1 arrays
        x = self._reshape(x)
        y_prob = self._sigmoid(np.dot(x, self.w) + self.b)

        return y_prob.round().flatten()

    def propagate(self, x, y, w, b):
        """
        Perform forward and backward propagation
        """

        m = x.shape[1]

        # Forward prop
        A = self._sigmoid(np.dot(x, w) + b)
        cost = 1.0 / m * -(np.dot(y.T, np.log(A)) + np.dot((1 - y).T, np.log(1 - A)))

        # Backprop
        dz = A - y
        dw = 1.0 / m * np.dot(dz.T, x)
        db = 1.0 / m * np.sum(dz)

        cost = np.squeeze(cost)
        return dw, db, cost

    def gradient_descent(self, x, y, w, b):
        """
        Gradient descent algorithm
        """

        learning_rate = self.learning_rate
        for i in range(self.num_iterations):

            # Calculate gradients and cost
            dw, db, cost = self.propagate(x, y, w, b)

            # Update weights
            w = w - (learning_rate * dw)
            b = b - (learning_rate * db)

            # Print cost
            if self.verbose and i % self.verbose_n == 0:
                print(f"Cost after iteration {i}: {cost}")

        self.w = w
        self.b = b

    def _reshape(self, z):
        """
        Handle rank 1 arrays
        """

        if z.ndim == 1:
            z = z[:, None]
        return z

    def _sigmoid(self, z):
        """
        Sigmoid activation function
        """

        s = 1 / (1 + np.exp(-z))
        return s


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Function to return a labelled confusion matrix
    def labelled_confusion_matrix(y, y_pred, prop=False):
        """Returns a labelled confusion matrix"""
        matrix = pd.DataFrame(confusion_matrix(y, y_pred))
        matrix.columns = ['Predicted:0', 'Predicted:1']
        matrix['Total'] = matrix['Predicted:0'] + matrix['Predicted:1']
        matrix = matrix.append(matrix.sum(), ignore_index=True)
        matrix.index = ['Actual:0', 'Actual:1', 'Total']

        if prop is True:
            matrix = round(matrix / matrix.iloc[2, 2], 4)

        return matrix

    # Create some random data
    x = np.random.uniform(0, 1, 50)
    e = np.random.uniform(-0.3, 0.3, 50)  # Add noise to data
    y = (x + e).round()

    # Fit a logistic regression model and get predictions
    lr = LogisticRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)

    # Compute accuracy and confusion matrix
    print(f'Accuracy: {sum(y_pred == y) / len(y)} \n')
    confusion_matrix = labelled_confusion_matrix(y, y_pred)
    print(confusion_matrix)
