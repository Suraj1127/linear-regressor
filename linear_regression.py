import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=20)
        self.m, self.n = x.shape
        self.w = np.random.randn(self.n,1)
        self.c = np.random.randn(1,1)

    def train(self, lambd, alpha):
        self.w = self.w - alpha*(np.matmul(self.x_train.T, np.matmul(self.x_train, self.w)+self.c-self.y_train)/self.m+lambd*self.w/self.m)
        self.c = self.c - alpha*np.sum(np.matmul(self.x_train, self.w)+self.c-self.y_train)/self.m

    def fit(self, alpha, no_of_iterations, lambd):
        for i in range(no_of_iterations):
            self.train(lambd, alpha)

    def predict(self, x):
        print(self.c.shape)
        return np.matmul(x, self.w) + self.c


def main():

    x = np.linspace(0, 10, 100).reshape(-1, 1)
    print(x.shape)
    y = 1*x + np.random.randn(x.shape[0], x.shape[1])

    linear_regression = LinearRegression(x, y)
    linear_regression.fit(0.1, 100, 0)
    y_pred = linear_regression.predict(x)
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()


if __name__ == "__main__":
    main()