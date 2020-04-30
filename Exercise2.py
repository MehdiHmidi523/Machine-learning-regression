import numpy as np
import matplotlib.pyplot as plt


# Plot polynomial
def fit(X, b, degree):
    if degree == 1:
        plt.plot(X, b[0] + X * b[1], 'b')
    elif degree == 2:
        plt.plot(X, b[0] + X * b[1] + (pow(X, 2) * b[2]), 'b')
    elif degree == 3:
        plt.plot(X, b[0] + X * b[1] + (pow(X, 2) * b[2]) + (pow(X, 3) * b[3]), 'b')
    elif degree == 4:
        plt.plot(X, b[0] + X * b[1] + (pow(X, 2) * b[2]) + (pow(X, 3) * b[3]) + (pow(X, 4) * b[4]), 'b')


# Data in the matrix housing_price_index.
data_ = np.genfromtxt('A2_DataSets/housing_price_index.csv', delimiter=',')
plt.figure(1)
plt.plot(data_[:, 0], data_[:, 1], 'r+', markersize=2)
plt.show()

# f(X) = β0 + β1X + β2X2 + . . . + βdXd for degrees d ∈ [1, 4]
# Subplots Degree 4 follows the data points tightly
X = data_[:, [0]]
y = data_[:, 1]
Xe = np.c_[np.ones((len(X), 1)), X[:, 0], pow(X[:, 0], 2), pow(X[:, 0], 3), pow(X[:, 0], 4)]
beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

plt.subplot(2, 2, 1)
plt.plot(data_[:, 0], data_[:, 1], 'r+', markersize=2)
fit(X, beta, 1)
plt.subplot(2, 2, 2)
plt.plot(data_[:, 0], data_[:, 1], 'r+', markersize=2)
fit(X, beta, 2)
plt.subplot(2, 2, 3)
plt.plot(data_[:, 0], data_[:, 1], 'r+', markersize=2)
fit(X, beta, 3)
plt.subplot(2, 2, 4)
plt.plot(data_[:, 0], data_[:, 1], 'r+', markersize=2)
fit(X, beta, 4)

# Using the best fit 4th degree polynomial and the index increasing, Jonas Nordqvist should expect a return of 3.1M SEK
years = 2022 - 1975
print(beta[0] + beta[1] * years + beta[2] * (years ** 2) + beta[3] * (years ** 3) + beta[4] * (years ** 4))
plt.show()
