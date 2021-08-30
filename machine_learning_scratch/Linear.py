import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn import datasets

def r2_Score(y_true, y_pred):
    corr = np.corrcoef(y_true, y_pred)
    corr = corr[0,1]
    return corr ** 2 

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


class LinearRegression:

    def __init__(self, n_iters= 1000, learning_rate = 0.01):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iters):
            predictions = np.dot(X,self.weights) + self.bias

            
            dw = (1/n_samples) * np.dot(X.T,(predictions-y))
            db = (1/n_samples) * np.sum((predictions-y))

            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        predictions = np.dot(X,self.weights) + self.bias
        return predictions



if __name__ == '__main__':
    X,y = datasets.make_regression(n_samples=1000, n_features=1, noise=25, random_state=5)

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=24,test_size=0.2)

    regression = LinearRegression(learning_rate=0.01, n_iters=1000)
    regression.fit(X_train,y_train)
    predict = regression.predict(X_test)

    msee = mse(y_test,predict)
    acc = r2_Score(y_test,predict)
    print(f"mse {msee}, r2 score {acc}")

    
    
    plt.scatter(X_test, y_test, s=10)
    plt.plot(X_test, predict, color="black", linewidth=3, label="Prediction")
    plt.show()

