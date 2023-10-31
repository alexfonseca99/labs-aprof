#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        y_hat_i = self.predict(x_i)
        if y_hat_i != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat_i, :] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=1):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        self.W -= learning_rate * log_regression_gradient(x_i, y_i, self.W)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        h1 = self.relu(np.dot(self.W1, X.T) + self.b1[:, None])
        z2 = np.dot(self.W2, h1) + self.b2[:, None]
        return np.argmax(self.softmax(z2), axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for (x_i, y_i) in zip(X, y):
            z1 = np.dot(self.W1, x_i) + self.b1
            h1 = self.relu(z1)
            z2 = np.dot(self.W2, h1) + self.b2
            h2 = self.softmax(z2)
            dL_dz = self.cross_entropy_gradient(h2, y_i)
            prev_grad_h = np.dot(self.W2.T, dL_dz)
            prev_grad_z = np.multiply(prev_grad_h, self.relu_gradient(z1))
            self.W2 -= learning_rate * np.outer(dL_dz, h1)
            self.b2 -= learning_rate * dL_dz
            self.W1 -= learning_rate * np.outer(prev_grad_z, x_i)
            self.b1 -= learning_rate * prev_grad_z

    def relu(self, x):
        return x * (x > 0) # Most efficient way to calculate relu according to a poster on stackoverflow

    def softmax(self, x):
        z = x - x.max()
        return np.exp(z) / np.sum(np.exp(z))

    def cross_entropy_gradient(self, h, y):
        one_hot = np.zeros(self.W2.shape[0])
        one_hot[y] = 1
        return h - one_hot

    def relu_gradient(self, x):
        return 1 * (x > 0)


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def log_regression_gradient(x, y, W):
    z = np.dot(W, x)
    z -= np.max(z)
    probabilities = np.exp(z) / np.sum(np.exp(z))
    one_hot = np.zeros(np.shape(W)[0])
    one_hot[y] = 1
    return np.outer((probabilities - one_hot), x.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        # model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
