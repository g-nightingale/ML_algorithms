import numpy as np


class LinearRegression:
    """
    Class to perform linear regression
    """

    def __init__(self, alpha=0.1, epochs=100, reg=None, lambd=1.0):
        self.alpha = alpha
        self.epochs = epochs
        self.reg = reg
        self.lambd = lambd
        self.w = None

    def fit(self, x, y):
        """
        Fit linear regression model using gradient descent

        Excellent article on l1 and l2 regularisation:
        https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
        """

        # Handle rank 1 arrays
        x = self._reshape(x)
        y = self._reshape(y)

        # Add bias term to x
        x = self._add_bias(x)

        # Initialise weights
        w = np.zeros((x.shape[1], 1))

        # Gradient descent
        for epoch in range(self.epochs):
            # Recap of matrix shapes during linear algebra operations:
            # (n * m) . (((m * n ) . (n * 1)) - (m * 1))
            # = (n * m) . (m * 1)
            # = (n * 1)
            if self.reg is None:
                dw = np.dot(x.T, (np.dot(x, w) - y))
            elif self.reg == 'l2':
                dw = np.dot(x.T, (np.dot(x, w) - y)) + 2 * self.lambd * w
            else:
                raise ValueError('Only l2 regularisation supported')

            w = w - self.alpha * dw

        # Store weights
        self.w = w

    def predict(self, x):
        """
        Predict on new data
        """

        x = self._reshape(x)
        x = self._add_bias(x)
        return np.dot(x, self.w)

    def _reshape(self, z):
        """
        Handle rank 1 arrays
        """

        if z.ndim == 1:
            z = z[:, None]
        return z

    def _add_bias(self, x):
        """
        Add bias term to the x vector
        """

        x_w_bias = np.zeros((x.shape[0], x.shape[1] + 1))
        x_w_bias[:, 0] = 1.0
        x_w_bias[:, 1:] = x
        return x_w_bias


# Example: Fit a linear regression model to some data
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create some random data
    x = np.random.randn(50)
    y = x + np.random.randn(50) / 2.0

    x = np.append(x, 3.0)
    y = np.append(y, 8.0)

    # Fit a linear regression model and get predictions
    lr = LinearRegression(epochs=100, alpha=0.01)
    lr.fit(x, y)
    y_pred = lr.predict(x)

    # Fit another model using l2 regularisation (ridge regression)
    lr_l2 = LinearRegression(epochs=100, alpha=0.01, reg='l2', lambd=10.0)
    lr_l2.fit(x, y)
    y_pred_l2 = lr_l2.predict(x)

    # Plot original data and predictions
    plt.figure(figsize=(12, 10))
    plt.title('Linear regression example')
    plt.scatter(x, y)
    plt.plot(x, y_pred, label='pred', color='red')
    plt.plot(x, y_pred_l2, label='pred l2', color='green')
    plt.legend(loc='upper left')
    plt.show()
