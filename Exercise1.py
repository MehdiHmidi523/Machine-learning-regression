import matplotlib.pyplot as plt
import numpy as np
from A2_utils import normalize, gradient, cost

# dataset GPUBenchmark.csv
data_ = np.genfromtxt('A2_DataSets/GPUbenchmark.csv', delimiter=',')
X = data_[:, [0, 1, 2, 3, 4, 5]]
y = data_[:, 6]
cuda = np.array(X[:, 0])
base = np.array(X[:, 1])
boost = np.array(X[:, 2])
speed = np.array(X[:, 3])
config = np.array(X[:, 4])
bandwidth = np.array(X[:, 5])

# Normalize X using Xn = (X − μ)/σ.
Xn = np.c_[normalize(cuda, cuda),
           normalize(base, base),
           normalize(boost, boost),
           normalize(speed, speed),
           normalize(config, config),
           normalize(bandwidth, bandwidth)]

# Multivariate dataset plot for Xi vs y
plt.figure(1)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(Xn[:, i], y, 'ro', markersize=2)
plt.show()
# Compute β using the normal equation
Xe = np.c_[np.ones((len(data_), 1)), Xn]
beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
featureValues = [2432, 1607, 1683, 8, 8, 256]
normalized_fv = [1, normalize(featureValues[0], cuda),
                 normalize(featureValues[1], base),
                 normalize(featureValues[2], boost),
                 normalize(featureValues[3], speed),
                 normalize(featureValues[4], config),
                 normalize(featureValues[5], bandwidth)]
print("The predicted benchmark result is: {}".format(np.dot(beta, normalized_fv)))

# cost function J(β)
print("The cost function J(β) result is: {}".format(cost(Xe, y, beta)))

# 1% of the final cost for the normal equation.
gradient, beta = gradient(Xe, y, 0.01)
gd_arr = np.asarray(gradient)
plt.figure(2)
plt.plot(gd_arr[:, 0], gd_arr[:, 1])
plt.show()

# Predicted benchmark result
print("The Predicted benchmark result is: {}".format(np.dot(beta, normalized_fv)))

