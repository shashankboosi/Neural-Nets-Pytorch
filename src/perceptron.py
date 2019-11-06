#!/usr/bin/env python3
"""
perceptron.py

Perceptron from scratch built using numpy

"""
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


class LinearModel:
    def __init__(self, num_inputs, learning_rate):
        """
        Model is very similar to the Perceptron
        1) the bias is indexed by w(n+1) rather than w(0), and
        (2) the activation function is a (continuous) sigmoid rather than a (discrete) step function.

        x1 ----> * w1 ----\
        x2 ----> * w2 -----\
        x3 ----> * w3 ------\
        ...
                             \
        xn ----> * wn -------+--> s --> activation ---> z
        1  ----> * w(n+1) --/
        """
        self.num_inputs = num_inputs
        self.lr = learning_rate
        self.weights = np.asarray([1.0, -1.0, 0.0])

    def activation(self, x):
        """
        Sigmoid activation function that accepts a float and returns
        Raises a Value error if a boolean, list or numpy array is passed in
        """
        if type(x) is bool or type(x) is list or type(x) is np.ndarray:
            raise ValueError('Wrong data type! Please check the input')
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        Inputs a numpy array. The bias term is the last element in self.weights.
        """
        return self.activation(self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.weights[2])

    @staticmethod
    def loss(prediction, label):
        """
        Cross entropy loss for the given prediction and label
        """
        return - (label * np.log(prediction) + (1 - label) * np.log(1 - prediction))

    @staticmethod
    def error(prediction, label):
        """
        Error between the label and the prediction
        For example, if label= 1 and the prediction was 0.8, return 0.2
                     if label= 0 and the prediction was 0.43 return -0.43
        """
        return label - prediction

    def backward(self, inputs, diff):
        """
        Gradient descent

        We take advantage of the simplification to compute the gradient
        directly from the differential or difference dE/ds = z - t
        (which is passed in as diff)

        The resulting weight update should look essentially the same as for the
        Perceptron Learning Rule except that
        the error can take on any continuous value between -1 and +1,
        rather than being restricted to the integer values -1, 0 or +1.
        """
        inputs = np.append(inputs, 1)
        self.weights = self.weights + self.lr * inputs * diff

    def plot(self, inputs, marker):
        """
        Plot the data and the decision boundary
        """
        xmin = inputs[:, 0].min() * 1.1
        xmax = inputs[:, 0].max() * 1.1
        ymin = inputs[:, 1].min() * 1.1
        ymax = inputs[:, 1].max() * 1.1

        x = np.arange(xmin * 1.3, xmax * 1.3, 0.1)
        plt.scatter(inputs[:25, 0], inputs[:25, 1], c="C0", edgecolors='w', s=100)
        plt.scatter(inputs[25:, 0], inputs[25:, 1], c="C1", edgecolors='w', s=100)

        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))
        plt.plot(x, -(self.weights[0] * x + self.weights[2]) / self.weights[1], marker, alpha=0.6)
        plt.title("Data and decision boundary")
        plt.xlabel("x1")
        plt.ylabel("x2").set_rotation(0)


def main():
    inputs, labels = pkl.load(open("../data/CNN/binary_classification_data.pkl", "rb"))

    epochs = 400
    model = LinearModel(num_inputs=inputs.shape[1], learning_rate=0.01)

    for i in range(epochs):
        num_correct = 0
        cost = 0
        for x, y in zip(inputs, labels):
            # Get prediction
            output = model.forward(x)

            # Calculate loss
            cost += model.loss(output, y)

            # Calculate difference or differential
            diff = model.error(output, y)

            # Update the weights
            model.backward(x, diff)

            # Record accuracy
            preds = output > 0.5  # 0.5 is midline of sigmoid
            num_correct += int(preds == y)

        print(f" Cost: {cost / len(inputs):.2f} Accuracy: {num_correct / len(inputs) * 100:.2f}%")
        model.plot(inputs, "C2--")
    model.plot(inputs, "k")
    plt.show()


if __name__ == "__main__":
    main()
