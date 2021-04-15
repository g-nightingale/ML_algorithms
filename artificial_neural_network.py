import numpy as np
from matplotlib import pyplot as plt


class ANN:
    """
    Implementation of an Artifical Neural Network using numpy.
    """

    def __init__(self, ann_dims, alpha=0.01, epochs=100, batch_size=None, l2_lambda=0, use_adam=False,
                 early_stopping=False, early_stopping_rounds=100, verbose=False, verbose_update=100, random_seed=42):

        # Internal attributes
        self.ann_dims = ann_dims
        self.layers = len(ann_dims) - 1
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.use_adam = use_adam
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.verbose_update = verbose_update
        self.random_seed = random_seed
        self.parameters = {}
        self.v_grads = {}
        self.s_grads = {}
        self.train_costs = []
        self.val_costs = []

        # Set random seed
        np.random.seed(self.random_seed)

    def fit(self, x, y, x_val=None, y_val=None):
        """
        Fit the Neural Network to some data x and y.
        """
        # Reshape the data
        x = x.T
        y = y.reshape((1, len(y)))

        if x_val is not None and y_val is not None:
            x_val = x_val.T
            y_val = y_val.reshape((1, len(y_val)))

        if self.batch_size is None:
            self.batch_size = x.shape[1]

        self.m = y.shape[1]
        counter = 1
        rounds_no_improvement = 0
        min_val_cost = np.inf
        seed = 0

        # Reset costs
        self.train_costs = []
        self.val_costs = []

        # Initialise parameters
        parameters = self.init_params()

        # Gradient descent
        for i in range(self.epochs):

            # Create batches
            seed = seed + 1
            batches = self.create_batches(x, y, seed)

            for batch in batches:
                x_batch, y_batch = batch

                y_hat, caches = self.forward_prop(x_batch, parameters)
                cost = self.cost_function(y_hat, y_batch)
                gradients = self.back_prop(y_hat, y_batch, caches)
                parameters = self.update_parameters(parameters, gradients, counter)
                counter += 1

                self.train_costs.append(cost)

                # Collect data for cost curves and early stopping
                if x_val is not None and y_val is not None:
                    y_hat_val, _ = self.forward_prop(x_val, parameters)
                    val_cost = self.cost_function(y_hat_val, y_val)
                    self.val_costs.append(val_cost)

                    if val_cost < min_val_cost:
                        min_val_cost = val_cost
                        rounds_no_improvement = 0
                    else:
                        rounds_no_improvement += 1

                # Early stopping
                if self.early_stopping and rounds_no_improvement > self.early_stopping_rounds:
                    if self.verbose:
                        print('Early stop!')
                        self.plot_cost(self.train_costs, self.val_costs)
                    self.parameters = parameters
                    return

            # Print costs
            if x_val is not None and y_val is not None and self.verbose and i % self.verbose_update == 0:
                print(f'Iteration {i}, cost: {round(cost, 6)}, val_cost: {round(val_cost, 6)}')
            elif self.verbose and i % self.verbose_update == 0:
                print(f'Iteration {i}, cost: {round(cost, 6)}')

        # Update parameters attribute
        self.parameters = parameters

        # Plot costs
        if self.verbose:
            if x_val is not None and y_val is not None:
                self.plot_cost(self.train_costs, self.val_costs)
            else:
                self.plot_cost(self.train_costs)

    def init_params(self):
        """
        Initialise network parameters using He initialisation.
        """
        for l in range(1, len(self.ann_dims)):
            self.parameters['W' + str(l)] = np.random.randn(self.ann_dims[l], self.ann_dims[l - 1]) * np.sqrt(
                2 / (self.ann_dims[l - 1]))
            self.parameters['b' + str(l)] = np.zeros((self.ann_dims[l], 1))

        if self.use_adam:
            for l in range(len(self.ann_dims) - 1):
                self.v_grads["dW" + str(l + 1)] = np.zeros((self.parameters['W' + str(l + 1)].shape[0],
                                                            self.parameters['W' + str(l + 1)].shape[1]))
                self.v_grads["db" + str(l + 1)] = np.zeros((self.parameters['b' + str(l + 1)].shape[0],
                                                            self.parameters['b' + str(l + 1)].shape[1]))
                self.s_grads["dW" + str(l + 1)] = np.zeros((self.parameters['W' + str(l + 1)].shape[0],
                                                            self.parameters['W' + str(l + 1)].shape[1]))
                self.s_grads["db" + str(l + 1)] = np.zeros((self.parameters['b' + str(l + 1)].shape[0],
                                                            self.parameters['b' + str(l + 1)].shape[1]))

        return self.parameters

    def create_batches(self, x, y, seed):
        """
        Create minibatches.
        """
        np.random.seed(
            seed)  # Update the seed on each iteration so that the batches are different
        batches = []

        # Shuffle datasets
        p = list(np.random.permutation(self.m))
        shuffled_x = x[:, p]
        shuffled_y = y[:, p].reshape((1, self.m))

        # Create complete batches
        n_complete_batches = self.m // self.batch_size

        for k in range(0, n_complete_batches):
            batch_x = shuffled_x[:, k * self.batch_size: (k + 1) * self.batch_size]
            batch_y = shuffled_y[:, k * self.batch_size: (k + 1) * self.batch_size]
            batch = (batch_x, batch_y)
            batches.append(batch)

        # Create the remaining batch if applicable
        if (self.m % self.batch_size) != 0:
            z = self.m % self.batch_size
            batch_x = shuffled_x[:, self.m - z: self.m]
            batch_y = shuffled_y[:, self.m - z: self.m]
            batch = (batch_x, batch_y)
            batches.append(batch)

        return batches

    def forward_prop(self, x, parameters):
        """
        Forward propagation.
        """
        caches = []
        A = x

        # Hidden layers (RELU)
        for l in range(1, self.layers):
            A_prev = A

            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            Z_cache = (A_prev, W, b)

            A = self.relu(Z)
            cache = (Z_cache, Z)
            caches.append(cache)

        # Output layer (Sigmoid)
        W = parameters['W' + str(self.layers)]
        b = parameters['b' + str(self.layers)]
        Z = np.dot(W, A) + b
        Z_cache = (A, W, b)

        AL = self.sigmoid(Z)
        cache = (Z_cache, Z)
        caches.append(cache)

        return AL, caches

    def sigmoid(self, Z):
        """
        Computes the Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z, forward=True):
        """
        Computes the Relu activation function.
        """
        return np.maximum(0, Z)

    def cost_function(self, AL, y):
        """
        Computes the binary cross-entropy loss function.
        """
        m = y.shape[1]

        # Binary cross entropy
        logprobs = np.multiply(np.log(AL), y) + np.multiply(np.log(1 - AL), 1 - y)
        cross_entropy = 1.0 / m * - np.sum(logprobs)
        cross_entropy = np.squeeze(cross_entropy)

        # L2 regularisation
        weight_square_sum = 0
        for l in range(1, len(self.ann_dims)):
            weight_square_sum += (np.sum(np.square(self.parameters['W' + str(l)])))

        l2_regularisation = 1.0 / m * self.l2_lambda / 2 * weight_square_sum

        cost = cross_entropy + l2_regularisation

        return cross_entropy

    def back_prop(self, AL, y, caches):
        """
        Backpropagation.
        """
        gradients = {}
        y = y.reshape(AL.shape)

        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
        cache_l = caches[self.layers - 1]
        s = 1 / (1 + np.exp(-cache_l[1]))
        dZ = dAL * s * (1 - s)
        A_prev, W, b = cache_l[0]
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T) + (self.l2_lambda / m * W)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        gradients["dA" + str(self.layers - 1)] = dA_prev
        gradients["dW" + str(self.layers)] = dW
        gradients["db" + str(self.layers)] = db

        for l in reversed(range(self.layers - 1)):
            cache_l = caches[l]
            dZ = gradients["dA" + str(l + 1)]
            dZ[cache_l[1] <= 0] = 0

            A_prev, W, b = cache_l[0]
            m = A_prev.shape[1]

            dW = 1 / m * np.dot(dZ, A_prev.T) + (self.l2_lambda / m * W)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

            gradients["dA" + str(l)] = dA_prev
            gradients["dW" + str(l + 1)] = dW
            gradients["db" + str(l + 1)] = db

        return gradients

    def update_parameters(self, parameters, gradients, c, b1=0.9, b2=0.999, e=1e-6):
        """
        Use gradient descent to update parameters.
        """
        if self.use_adam:
            v_grads_corrected = {}
            s_grads_corrected = {}

            for l in range(self.layers):
                # Exponential moving average of the gradients
                self.v_grads["dW" + str(l + 1)] = (b1 * self.v_grads["dW" + str(l + 1)]) + (
                            (1 - b1) * gradients["dW" + str(l + 1)])
                self.v_grads["db" + str(l + 1)] = (b1 * self.v_grads["db" + str(l + 1)]) + (
                            (1 - b1) * gradients["db" + str(l + 1)])
                v_grads_corrected["dW" + str(l + 1)] = self.v_grads["dW" + str(l + 1)] / (1 - b1 ** c)
                v_grads_corrected["db" + str(l + 1)] = self.v_grads["db" + str(l + 1)] / (1 - b1 ** c)

                # Exponential moving average of the squared gradients
                self.s_grads["dW" + str(l + 1)] = (b2 * self.s_grads["dW" + str(l + 1)]) + (
                            (1 - b2) * gradients["dW" + str(l + 1)] ** 2)
                self.s_grads["db" + str(l + 1)] = (b2 * self.s_grads["db" + str(l + 1)]) + (
                            (1 - b2) * gradients["db" + str(l + 1)] ** 2)
                s_grads_corrected["dW" + str(l + 1)] = self.s_grads["dW" + str(l + 1)] / (1 - b2 ** c)
                s_grads_corrected["db" + str(l + 1)] = self.s_grads["db" + str(l + 1)] / (1 - b2 ** c)

                # Update parameters
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.alpha * v_grads_corrected[
                    "dW" + str(l + 1)] / (np.sqrt(s_grads_corrected["dW" + str(l + 1)]) + e)
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.alpha * v_grads_corrected[
                    "db" + str(l + 1)] / (np.sqrt(s_grads_corrected["db" + str(l + 1)]) + e)

        else:
            for l in range(self.layers):
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.alpha * gradients["dW" + str(l + 1)]
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.alpha * gradients["db" + str(l + 1)]

        return parameters

    def plot_cost(self, train_costs, val_costs=None):
        """
        Plot cost curves.
        """
        plt.plot(train_costs, label='train')
        if val_costs is not None:
            plt.plot(val_costs, label='val')
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.show()

    def predict(self, x):
        """
        Use trained model to make predictions on new data.
        """
        x = x.T
        m = x.shape[1]
        n = len(self.parameters) // 2
        preds = np.zeros((1, m), dtype=int)

        # Forward propagation
        probas, _ = self.forward_prop(x, self.parameters)

        # Create predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                preds[0, i] = 1
            else:
                preds[0, i] = 0

        return preds
