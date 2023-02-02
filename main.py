import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import environ

def h(x, w):
    return np.matmul(x, w)

# Load the data
def load_data(file):
    data = pd.read_csv(file)
    data = np.array(data)
    return data

# Compute the weight vector given: (X'*X)^-1*(X'Y)
def computeWeights(x, y):
    w = np.dot(np.linalg.matrix_power(np.dot(designMatrix.transpose(), designMatrix), -1), np.dot(designMatrix.transpose(), trainingy))
    return w

trainingx = load_data('hw1xtr.dat.csv')
trainingy = load_data('hw1ytr.dat.csv')

plt.scatter(trainingx, trainingy, label='Training', color='blue')
plt.show()

testingx = load_data('hw1xte.dat.csv')
testingy = load_data('hw1yte.dat.csv')

plt.scatter(testingx, testingy, label='Testing', color='orange')
plt.show()

# Compute the regression line
designMatrix = np.c_[ trainingx, np.ones(39) ]
weights = computeWeights(designMatrix, trainingy)
regressionLine = weights[0]*trainingx + weights[1]

plt.scatter(trainingx, trainingy, label='Training', color='blue')
plt.plot(trainingx, regressionLine, color='red')
plt.show()

plt.scatter(testingx, testingy, label='Testing', color='orange')
plt.plot(trainingx, regressionLine, color='red')
plt.show()




