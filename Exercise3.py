import numpy as np
import matplotlib.pyplot as plt
from A2_utils import sigmoid, normalize, gradientDescent


# Train a linear logistic regression model
def linearReg(dataSet, dataType):
    normalized_data = np.c_[np.ones((len(dataSet), 1)), normalize(dataSet[:, 0]),
                            normalize(dataSet[:, 1]), normalize(dataSet[:, 2]),
                            normalize(dataSet[:, 3]), normalize(dataSet[:, 4]),
                            normalize(dataSet[:, 5]), normalize(dataSet[:, 6]),
                            normalize(dataSet[:, 7]), normalize(dataSet[:, 8])]

    gd, beta = gradientDescent(0.01, 5000, normalized_data, dataSet[:, 9])
    gd_arr = np.asarray(gd)

    plt.figure(1)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(β)')
    if dataType == 1:
        plt.plot(gd_arr[:, 0], gd_arr[:, 1], label='Training data')
    elif dataType == 2:
        plt.plot(gd_arr[:, 0], gd_arr[:, 1], label='Test data')
    plt.legend()

    pred = np.round(sigmoid(np.dot(normalized_data, beta)))
    dataAccuracy = 100 - np.round((np.mean(pred != dataSet[:, 9]) * 100), 2)
    print("Incorrect classifications: " + str(
        int(np.around(len(dataSet[:, 9]) * np.mean(pred != dataSet[:, 9])))))
    print("Accuracy: " + str(dataAccuracy) + "%\n")


# shuffle rows in the raw data matrix
data_ = np.genfromtxt('A2_DataSets/breast_cancer.csv', delimiter=',')
np.random.shuffle(data_)
for d in data_:
    if d[9] == 2:
        d[9] = 0
    if d[9] == 4:
        d[9] = 1

# with less training data, your parameter estimates have greater variance.
# With less testing data, your performance statistic will have greater variance.
# In the lecture we are told to be concerned with dividing data such that neither variance is too high
trainingSet = np.array(data_[:546])
testSet = np.array(data_[546:])

print("Training data:")
linearReg(trainingSet, 1)

print("Test data:")
linearReg(testSet, 2)

plt.show()
# Reruns result in very small differences. The training set is be within 96 and 98% accuracy. The test set is within
# 97 and 100% accuracy. Depending on how we choose to divide the data, the amount of errors will change but the
# accuracy will be approximately the same.
