import numpy as np


def normalize(data, x):
    return (data - np.mean(x)) / np.std(x)


def cost(Xe, y, beta):
    return ((np.dot(Xe, beta) - y).T.dot(np.dot(Xe, beta) - y)) / len(Xe)


def gradient(Xe, y, alpha):
    b = np.array([0, 0, 0, 0, 0, 0, 0])
    gradient = []
    for i in range(1000):
        b = b - (np.dot((alpha * Xe.T), ((np.dot(Xe, b)) - y)))
        gradient.append([i, cost(Xe, y, b)])
    return gradient, b


def gradientDescent(alpha, iteration, X, y):
    beta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    gd = []
    for i in range(iteration):
        beta = beta - (np.dot(((alpha / len(X)) * X.T), ((sigmoid(np.dot(X, beta))) - y)))
        gd.append([i, costFunction(len(X), y, X, beta)])
    return gd, beta


def sigmoid(x):
    return (np.e ** x) / ((np.e ** x) + 1)


def costFunction(n, y, x, beta):
    return (-1 / n) * (
            np.dot(y.T, np.log(sigmoid(np.dot(x, beta)))) + np.dot((1 - y).T, np.log(1 - sigmoid(np.dot(x, beta)))))
