import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class KNN:

    def __init__(self,k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        distances = [euclidean_distance(x,x_train) for x_train in self.x_train]
        k_index = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_index]

        most_common = Counter(k_neighbor_labels).most_common(1)

        return most_common[0][0]

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1234,test_size=0.2)

    clf= KNN(k=5)
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    acc = accuracy(y_test,prediction)

    print(f"acc {acc}")