#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearRegression:
    """
    Author: Suraj Regmi
    Date: 31st May, 2018
    Description: Simple LinearRegression class that builts the linear
    regression model and then predicts the output from input values.
    Also, validates the model using root mean square error and coefficient
    of determination.
    """

    def __init__(self, x, y):
        """
        Initialize the model with training data
        :param x: Numpy array of training input
        :param y: Numpy array of labelled output

        Size of matrix x: number of training examples * number of features of training examples
        Size of matrix y: number of training examples * 1
        """

        # Splitting the x and y matrix into train and test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=20)
        self.m, self.n = x.shape

        # Initialize the weights and biases
        self.w = np.random.randn(self.n, 1)
        self.c = np.random.randn(1, 1)

    def train(self, lambd, alpha):
        """
        One iteration of training
        :param lambd: regularization parameter
        :param alpha: learning rate
        """
        self.w = self.w - alpha*(np.matmul(self.x_train.T, np.matmul(self.x_train, self.w)
                                           + self.c-self.y_train)/self.m+lambd*self.w/self.m)
        self.c = self.c - alpha*np.sum(np.matmul(self.x_train, self.w)+self.c-self.y_train)/self.m

    def fit(self, alpha, no_of_iterations, lambd):
        """
        Fits the training data to the linear model.
        :param alpha: learning rate
        :param no_of_iterations: no of iterations of gradient descent algorithm
        :param lambd: regularization parameter
        """

        # Training process to the given number of iterations
        for _ in range(no_of_iterations):
            self.train(lambd, alpha)

    def predict(self, x):
        """
        Predicts the value of y on the basis of given value of x.
        :param x: input value of x, independent variable
        :return: value of predicted or dependent variable
        """
        return np.matmul(x, self.w) + self.c

    def validate(self):
        """
        Evaluates the performance the model by calculating and printing coefficient
        of determination and root mean squared error on the test set
        """

        y_test_pred = self.predict(self.x_test)
        y_test_mean = np.average(y_test_pred)

        rmse = np.sqrt(np.average((self.y_test-y_test_pred)**2))
        coef_of_determination = 1-np.sum((self.y_test-y_test_pred)**2)/np.sum((self.y_test-y_test_mean)**2)

        print("Coefficient of determination: {}".format(coef_of_determination))
        print("Root mean squared error: {}".format(rmse))


def main():

    # Costruct training matrices
    x = np.array([np.linspace(0, 10, 100), 2*np.linspace(0, 10, 100)]).T
    y = np.sum(x, axis=1).reshape(-1, 1)

    # Train the model and predict
    linear_regression = LinearRegression(x, y)
    linear_regression.fit(0.001, 10000, 0)
    y_pred = linear_regression.predict(x)

    # Print weights, biases and the plot and also print the performance estimators of the model
    print("Weights: {}\nBiases: {}".format(linear_regression.w, linear_regression.c))
    plt.plot(y, y_pred)
    plt.show()
    linear_regression.validate()


if __name__ == "__main__":
    main()
