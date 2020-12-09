'''Simple Perceptron'''

import numpy as np


class SimplePerceptron():
    """
    Perceptron class

    Args:
        inputs (array): input values
        outputs (list): output values
    """

    def __init__(self, inputs, outputs, epochs=1000, learn_rate=1, bias=1):
        """
        The weights for each of the inputs and the bias are initialized.
        """
        self.bias = bias
        self.inputs = []
        for i in inputs:
            l = i.tolist()
            l.insert(0, self.bias)
            self.inputs.append(l)
        self.output = outputs
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.weights = np.random.uniform(-1.0, 1.0, len(self.inputs[0]))
        self.number_imputs = len(self.inputs)

    def net(self, features):
        """
        Calculate the sum of the products of the inputs by their weights,
        the bias is included.

        Args:
            features (list): data input (bias included)
        Returns:
            (float): net_input
        """
        return np.dot(features, self.weights)

    def predict(self, features):
        """
        Calculate the label taking into account the activation function

        Args:
            features (list): data input
        Returns:
            (float): label (-1.0 or 1.0)
        """
        return np.where(self.net(features) >= 0.0, 1, 0)

    def train(self):
        for e in range(self.epochs):
            error = False
            for features, expected in zip(self.inputs, self.output):
                pred = self.predict(features)
                if pred != expected:
                    if expected == 0:
                        expected = -1.0
                    self.weights += self.learn_rate * expected * np.asarray(features)
                    error = True
            if error is False:
                print('------------------------')
                print(('Epoch:', e))
                print('------------------------')
                print(('Weights:', self.weights))
                print('------------------------\n')
                return self.weights
        if error:
            print('------------------------')
            print(
                "The maximum number of epochs has been reached. " +
                "You may not have a linear solution.")
            print('------------------------\n')
