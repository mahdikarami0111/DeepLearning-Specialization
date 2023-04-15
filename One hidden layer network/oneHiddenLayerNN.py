import numpy as np
import sklearn
import matplotlib
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases_v2 import *
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


class Model:
    def __init__(self, input_size, hidden_size, output_size, train_set, test_set=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.train_set = train_set
        self.test_set = test_set
        self.parameters = self.initialize_parameters()

    def test(self):
        for param in self.parameters:
            print(param)

    def initialize_parameters(self):
        np.random.seed(2)
        W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01
        b1 = np.zeros((self.hidden_size, 1))
        W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        b2 = np.zeros((self.output_size, 1))

        return W1, b1, W2, b2

    def forward_propagate(self):
        W1, b1, W2, b2 = self.parameters
        X = self.train_set[0]
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        return Z1, A1, Z2, A2

    def set_parameters(self, parameters):
        self.parameters = parameters

    def compute_cost(self, A2):
        Y = self.train_set[1]
        m = Y.shape[1]
        loss = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
        cost = (-1/m) * np.sum(loss)

        cost = float(np.squeeze(cost))
        assert (isinstance(cost, float))
        return cost

    def backward_propagate(self, cache):
        W1, b1, W2, b2 = self.parameters
        X, Y = self.train_set
        Z1, A1, Z2, A2 = cache
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) * (1/m)
        db2 = np.sum(dZ2, axis=1, keepdims=True) * (1/m)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) * (1/m)
        db1 = np.sum(dZ1, axis=1, keepdims=True) * (1/m)

        return dW1, db1, dW2, db2

    def update_parameters(self, grads, learning_rate):
        W1, b1, W2, b2 = self.parameters
        dW1, db1, dW2, db2 = grads

        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        b1 = b1 - learning_rate * db1
        b2 = b2 - learning_rate * db2

        return W1, b1, W2, b2

    def train(self, learning_rate, num_iterations=10000, print_cost=False):
        for i in range(num_iterations):
            cache = self.forward_propagate()
            cost = self.compute_cost(cache[3])
            grads = self.backward_propagate(cache)
            self.parameters = self.update_parameters(grads, learning_rate)
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        return self.parameters



# X, Y = load_planar_dataset()
X, Y = nn_model_test_case()
print(X.shape)
model = Model(2, 4, 1, (X, Y))
model.train(1.02, 10000, True)

parameters = model.parameters
W1 = parameters[0]
b1 = parameters[1]
W2 = parameters[2]
b2 = parameters[3]

print("W1 = " + str(parameters[0]))
print("b1 = " + str(parameters[1]))
print("W2 = " + str(parameters[2]))
print("b2 = " + str(parameters[3]))