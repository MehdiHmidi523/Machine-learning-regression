import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from A2_utils import costFunction, sigmoid


def gradientDescent(alpha, iteration, X, y):
    beta = np.array([0, 0, 0, 0, 0, 0])
    gd = []
    for i in range(iteration):
        beta = beta - (np.dot(((alpha / len(X)) * X.T), ((sigmoid(np.dot(X, beta))) - y)))
        gd.append([i, costFunction(len(X), y, X, beta)])
    return gd, beta


def mapFeature(X1, X2, D, eins):
    if eins:
        Xe = np.c_[np.ones([len(X1), 1]), X1, X2]
    else:
        Xe = np.c_[X1, X2]
    for i in range(2, D + 1):
        for j in range(0, i + 1):
            Xe = np.append(Xe, (X1 ** (i - j) * X2 ** j).reshape(-1, 1), 1)
    return Xe


def mesh_grid_plot(X, y, i):
    h = .05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x1, x2 = xx.ravel(), yy.ravel()

    xy_mesh = mapFeature(x1, x2, i, False)
    classes = result.predict(xy_mesh)
    clz_mesh = classes.reshape(xx.shape)

    light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.pcolormesh(xx, yy, clz_mesh, cmap=light)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', cmap=points)


# Plot the data in X and y
data_ = np.genfromtxt('A2_DataSets/microchips.csv', delimiter=',')
X = data_[:, [0, 1]]
y = data_[:, 2]
ok = data_[data_[:, 2] == 1]
fail = data_[data_[:, 2] == 0]
plt.figure(1)
plt.plot(ok[:, 0], ok[:, 1], 'b+', markersize=3)
plt.plot(fail[:, 0], fail[:, 1], 'r+', markersize=3)
plt.show()

# find β in the case of a quadratic model. // no need for normalization
Xe = np.c_[np.ones((len(data_), 1)),
           X[:, 0],
           X[:, 1],
           pow(X[:, 0], 2),
           X[:, 0] * X[:, 1],
           pow(X[:, 1], 2)]

print('Parameters:\n α = ' +
      str(0.1) + ', N_iteration = ' + str(20000))

# The cost function J(β) as a function over iterations
gd, beta = gradientDescent(0.1, 20000, Xe, y)
gd_arr = np.asarray(gd)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.xlabel('Iteration')
plt.ylabel('Cost J(β)')
plt.plot(gd_arr[:, 0], gd_arr[:, 1])
# The corresponding decision boundary
h = .01
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
x1, x2 = xx.ravel(), yy.ravel()

XXe = mapFeature(x1, x2, 2, True)
classes = sigmoid(np.dot(XXe, beta)) > 0.5
clz_mesh = classes.reshape(xx.shape)

light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, clz_mesh, cmap=light)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', cmap=points)
plt.show()

# a function that takes two features X1, X2 and a degree d as input and outputs
# all combinations of polynomial terms of degree <= to d of X1 and X2.
Xe = np.c_[X[:, 0],
           X[:, 1],
           pow(X[:, 0], 2),
           X[:, 0] * X[:, 1],
           pow(X[:, 1], 2)]
result = LogisticRegression(solver='lbfgs', C=1000, tol=1e-6, max_iter=100000)
result.fit(Xe, y)

plt.figure(3)
mesh_grid_plot(X, y, 2)
plt.title('microchips.csv, In sci-kit-learn logistic regression')
plt.show()

plt.figure(4)
for i in range(9):
    Xe = mapFeature(X[:, 0], X[:, 1], i + 1, False)
    result.fit(Xe, y)
    plt.subplot(3, 3, i + 1)
    mesh_grid_plot(X, y, i + 1)
    plt.title("D = " + str(i + 1) + ", Train errors = " + str(np.sum(result.predict(Xe) != y)))
plt.show()
