import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from A2_utils import cost


data_ = np.genfromtxt('A2_DataSets/GPUbenchmark.csv', delimiter=',')
X = data_[:, [0, 1, 2, 3, 4, 5]]
y = data_[:, 6]
result = LinearRegression()
feat = [0, 1, 2, 3, 4, 5]
best = []
for i in range(6):
    costs = []
    for j in feat:
        aux = X[:, best + [j]]
        result.fit(aux, y)
        costs.append(cost(aux, result.predict(aux), result.coef_))
    best.append(feat[np.argmin(costs)])
    feat = np.setdiff1d(feat, feat[np.argmin(costs)])

print('Best: ' + str(best))
x_best = np.array([X[:, best[0]],
                   X[:, best[1]],
                   X[:, best[2]],
                   X[:, best[3]],
                   X[:, best[4]],
                   X[:, best[5]]]).T

# 3-fold cross-validation to find the best model among all Mi, i = 1,...,6.
features = []
for i in range(6):
    plt.subplot(2, 3, i + 1)
    aux = x_best[:, features + [i]]
    features.append(i)

    result.fit(aux, y)
    prediction = result.predict(aux)
    mse = (np.square(prediction - y)).mean()

    plt.title("Features: " + str(i + 1) + ", MSE: " + str(np.around(mse, 1)))
    plt.plot(y, "ro", markersize=3)
    plt.plot(prediction, "black")

print("Feature 4 is not far away from feature 5 and 6 in error. the best choice is the simpler one. Occam's razor, "
      "which states that if there are many competing models to explain or fit the same data, then that with the "
      "fewest assumptions or parameters should be selected. The issue of over-fitting is relevant to many applications "
      "such as curve-fitting, regression in general, and in this case that would be the model with 4 features.")
plt.show()

